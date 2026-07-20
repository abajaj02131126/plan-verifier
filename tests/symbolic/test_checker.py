"""Extensive unit tests for the symbolic verifier [SOUNDNESS-CRITICAL],
including edge cases: empty plans, actions with no preconditions, negative
preconditions, unknown actions/objects, arity/type errors, and resource
floor/cap violations at interior prefixes.
"""

import pytest

from verifier.domains.blocksworld import build_domain
from verifier.schema import (
    ActionSchema,
    Domain,
    Literal,
    Parameter,
    PredicateAtom,
    PredicateDefinition,
    Problem,
    ResourceDimension,
    TypedObject,
)
from verifier.schema.state import GroundAction
from verifier.symbolic import verify


@pytest.fixture()
def bw_problem() -> Problem:
    domain = build_domain(energy_cap=6.0)
    return Problem(
        name="checker-fixture",
        domain=domain,
        objects=[TypedObject(name=b, type="block") for b in ("b1", "b2", "b3")],
        init=[
            PredicateAtom(predicate="on-table", args=["b1"]),
            PredicateAtom(predicate="on-table", args=["b2"]),
            PredicateAtom(predicate="on", args=["b3", "b2"]),
            PredicateAtom(predicate="clear", args=["b1"]),
            PredicateAtom(predicate="clear", args=["b3"]),
            PredicateAtom(predicate="crane-empty", args=[]),
        ],
        goal=[Literal.pos("on", "b1", "b3"), Literal.pos("on-table", "b2")],
    )


def _ga(name: str, *args: str) -> GroundAction:
    return GroundAction(schema_name=name, args=args)


def test_valid_plan_passes_everything(bw_problem):
    v = verify([_ga("pick-up", "b1"), _ga("stack", "b1", "b3")], bw_problem)
    assert v.is_consistent and v.is_goal_complete and v.is_resource_feasible
    assert v.overall_valid
    assert v.explanation.startswith("VALID")


def test_empty_plan_fails_goal_only(bw_problem):
    v = verify([], bw_problem)
    assert v.is_consistent
    assert v.is_resource_feasible
    assert not v.is_goal_complete
    # on-table(b2) already holds in init, so only on(b1, b3) is unmet
    assert len(v.unmet_goals) == 1
    assert "on(b1, b3)" in v.unmet_goals[0]
    assert not v.overall_valid


def test_precondition_violation_reported_with_step_number(bw_problem):
    v = verify([_ga("pick-up", "b3")], bw_problem)  # b3 not on table
    assert not v.is_consistent
    assert any("step 1" in viol and "on-table(b3)" in viol for viol in v.consistency_violations)


def test_unknown_action_and_object(bw_problem):
    v = verify([_ga("levitate", "b1"), _ga("pick-up", "b99")], bw_problem)
    assert not v.is_consistent
    assert any("unknown action 'levitate'" in x for x in v.consistency_violations)
    assert any("unknown object 'b99'" in x for x in v.consistency_violations)


def test_arity_mismatch(bw_problem):
    v = verify([_ga("stack", "b1")], bw_problem)
    assert not v.is_consistent
    assert any("arity mismatch" in x for x in v.consistency_violations)


def test_resource_floor_violation_at_interior_prefix(bw_problem):
    # 3x pick-up/put-down = 6 energy (hits exactly 0, legal), then one more
    # pick-up drives energy to -1 at step 7 — a floor violation mid-plan even
    # though nothing else is wrong and the goal check happens later.
    plan = [_ga("pick-up", "b1"), _ga("put-down", "b1")] * 3 + [_ga("pick-up", "b1"), _ga("stack", "b1", "b3")]
    v = verify(plan, bw_problem)
    assert v.is_consistent
    assert v.is_goal_complete
    assert not v.is_resource_feasible
    viol = v.resource_violations[0]
    assert viol.resource == "energy" and viol.step == 7 and viol.kind == "floor"
    assert not v.overall_valid


def test_exactly_at_floor_is_legal(bw_problem):
    # 6 energy consumed exactly: 3x(pick-up + put-down) = 6, ends at 0.0
    plan = [_ga("pick-up", "b1"), _ga("put-down", "b1")] * 3
    v = verify(plan, bw_problem)
    assert v.is_resource_feasible


def test_input_errors_count_against_consistency(bw_problem):
    v = verify(
        [_ga("pick-up", "b1"), _ga("stack", "b1", "b3")],
        bw_problem,
        input_errors=["unknown action 'move' (not in domain 'blocksworld')"],
    )
    assert not v.is_consistent
    assert any("unstructured step" in x for x in v.consistency_violations)
    assert not v.overall_valid  # plan can't be executed as written


def test_negative_precondition_and_no_precondition_action():
    """A hand-built domain exercising negated preconditions and an action with
    no preconditions at all — neither exists in the shipped domains."""
    domain = Domain(
        name="toy",
        types=["item"],
        predicates=[
            PredicateDefinition(name="broken", arg_types=["item"]),
            PredicateDefinition(name="polished", arg_types=["item"]),
        ],
        action_schemas=[
            ActionSchema(
                name="polish",
                parameters=[Parameter(name="x", type="item")],
                preconditions=[Literal.neg("broken", "x")],  # must NOT be broken
                add_effects=[PredicateAtom(predicate="polished", args=["x"])],
                del_effects=[],
                resource_deltas={},
            ),
            ActionSchema(
                name="drop",
                parameters=[Parameter(name="x", type="item")],
                preconditions=[],  # no preconditions
                add_effects=[PredicateAtom(predicate="broken", args=["x"])],
                del_effects=[],
                resource_deltas={},
            ),
        ],
        resource_dimensions=[],
    )
    problem = Problem(
        name="toy-1",
        domain=domain,
        objects=[TypedObject(name="cup", type="item")],
        init=[],
        goal=[Literal.pos("polished", "cup")],
    )

    # polish then done: negated precondition holds (cup not broken)
    v = verify([_ga("polish", "cup")], problem)
    assert v.overall_valid

    # drop (no preconditions, always applicable) then polish: now broken
    v2 = verify([_ga("drop", "cup"), _ga("polish", "cup")], problem)
    assert not v2.is_consistent
    assert any("must be false" in x for x in v2.consistency_violations)


def test_replenishing_resource_cap_violation():
    """A resource that regenerates (positive delta) must respect its cap."""
    domain = Domain(
        name="battery",
        types=["robot"],
        predicates=[PredicateDefinition(name="idle", arg_types=["robot"])],
        action_schemas=[
            ActionSchema(
                name="recharge",
                parameters=[Parameter(name="r", type="robot")],
                preconditions=[Literal.pos("idle", "r")],
                add_effects=[],
                del_effects=[],
                resource_deltas={"charge": +4.0},
            ),
        ],
        resource_dimensions=[ResourceDimension(name="charge", initial=8.0, cap=10.0)],
    )
    problem = Problem(
        name="battery-1",
        domain=domain,
        objects=[TypedObject(name="r1", type="robot")],
        init=[PredicateAtom(predicate="idle", args=["r1"])],
        goal=[Literal.pos("idle", "r1")],
    )
    v = verify([_ga("recharge", "r1")], problem)  # 8 + 4 = 12 > cap 10
    assert not v.is_resource_feasible
    assert v.resource_violations[0].kind == "cap"


def test_verdict_agrees_with_oracle_labeler_on_fixtures(bw_problem):
    """In-process cross-check on hand-built plans of every flaw type: the
    Phase 2 labeler and this verifier are independent implementations and
    must produce identical per-dimension verdicts."""
    from verifier.generation.labeler import label_plan
    from verifier.generation.parser import parse_plan

    fixture_plans = [
        "Step 1: pick-up(b1)\nStep 2: stack(b1, b3)",  # valid
        "Step 1: pick-up(b3)",  # precondition slip
        "Step 1: pick-up(b1)\nStep 2: put-down(b1)",  # goal-incomplete
        "Step 1: move(b1, b3)",  # hallucinated action
        ("Step 1: pick-up(b1)\nStep 2: put-down(b1)\n" * 3)
        + "Step 7: pick-up(b1)\nStep 8: stack(b1, b3)",  # resource overrun
        "",  # empty
    ]
    for text in fixture_plans:
        parse = parse_plan(text, bw_problem.domain)
        labels = label_plan(bw_problem, parse)
        verdict = verify(parse.actions, bw_problem, input_errors=parse.errors)
        assert labels.is_consistent == verdict.is_consistent, text
        assert labels.is_goal_complete == verdict.is_goal_complete, text
        assert labels.is_resource_feasible == verdict.is_resource_feasible, text
        assert labels.overall_valid == verdict.overall_valid, text
