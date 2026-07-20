"""Adversarial tests for the oracle labeler [SOUNDNESS-CRITICAL]: one
hand-constructed plan per flaw type, plus a fully valid plan, on a fixture
blocksworld problem small enough to verify by hand.

Fixture: 3 blocks. init: b1 on table (clear), b2 on table, b3 on b2 (clear),
crane empty. Goal: on(b1, b3) AND on-table(b2). Energy cap 6.
Hand-checked valid plan: pick-up(b1) [1] ; stack(b1, b3) [2] => total 3 <= 6.
"""

import pytest

from verifier.domains.blocksworld import build_domain
from verifier.generation.labeler import label_plan
from verifier.generation.parser import parse_plan
from verifier.schema import Literal, PredicateAtom, Problem, TypedObject


@pytest.fixture()
def problem() -> Problem:
    domain = build_domain(energy_cap=6.0)
    return Problem(
        name="labeler-fixture",
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


def _label(problem: Problem, text: str):
    return label_plan(problem, parse_plan(text, problem.domain))


def test_fully_valid_plan(problem):
    labels = _label(problem, "Step 1: pick-up(b1)\nStep 2: stack(b1, b3)")
    assert labels.is_consistent
    assert labels.is_goal_complete
    assert labels.is_resource_feasible
    assert labels.overall_valid
    assert labels.consistency_violations == []
    assert labels.unmet_goals == []
    assert labels.resource_violations == []


def test_precondition_slip_is_inconsistent(problem):
    # b3 is on b2, so pick-up(b3) (table-only) violates on-table(b3)
    labels = _label(problem, "Step 1: pick-up(b3)\nStep 2: stack(b3, b1)")
    assert not labels.is_consistent
    assert any("on-table" in v for v in labels.consistency_violations)
    assert not labels.overall_valid


def test_goal_omission_is_goal_incomplete(problem):
    # valid actions but achieves nothing toward on(b1, b3)
    labels = _label(problem, "Step 1: pick-up(b1)\nStep 2: put-down(b1)")
    assert labels.is_consistent
    assert not labels.is_goal_complete
    assert any("on(b1, b3)" in g for g in labels.unmet_goals)
    assert not labels.overall_valid


def test_resource_overrun_is_infeasible(problem):
    # 4 x (pick-up + put-down) = 8 energy > cap 6, then solve the goal anyway.
    text = """\
Step 1: pick-up(b1)
Step 2: put-down(b1)
Step 3: pick-up(b1)
Step 4: put-down(b1)
Step 5: pick-up(b1)
Step 6: put-down(b1)
Step 7: pick-up(b1)
Step 8: stack(b1, b3)
"""
    labels = _label(problem, text)
    assert labels.is_consistent
    assert labels.is_goal_complete  # goal IS reached...
    assert not labels.is_resource_feasible  # ...but the energy budget is blown
    assert any("energy" in v for v in labels.resource_violations)
    assert not labels.overall_valid


def test_hallucinated_action_is_inconsistent(problem):
    labels = _label(problem, "Step 1: move(b1, b3)")
    assert not labels.is_consistent
    assert any("unknown action 'move'" in v for v in labels.consistency_violations)
    assert not labels.overall_valid


def test_wrong_arity_is_inconsistent(problem):
    labels = _label(problem, "Step 1: stack(b1)")
    assert not labels.is_consistent
    assert not labels.overall_valid


def test_unknown_object_is_inconsistent(problem):
    labels = _label(problem, "Step 1: pick-up(b9)")
    assert not labels.is_consistent
    assert any("unknown object 'b9'" in v for v in labels.consistency_violations)


def test_empty_plan_labels(problem):
    labels = _label(problem, "")
    assert labels.is_consistent  # no steps => nothing inconsistent
    assert not labels.is_goal_complete  # but the goal is not reached
    assert labels.is_resource_feasible
    assert not labels.overall_valid


def test_early_slip_does_not_mask_later_goal_check(problem):
    # Step 1 has a precondition violation (crane not empty is fine here —
    # instead: pick up non-clear b2). Lenient progression still applies
    # effects, so the goal check reflects the whole plan, not just the prefix.
    text = "Step 1: pick-up(b2)\nStep 2: put-down(b2)\nStep 3: pick-up(b1)\nStep 4: stack(b1, b3)"
    labels = _label(problem, text)
    assert not labels.is_consistent  # b2 was not clear
    assert labels.is_goal_complete  # but the goal atoms do end up true
    assert not labels.overall_valid


def test_multiple_flaws_all_reported(problem):
    text = "Step 1: move(b1, b3)\nStep 2: pick-up(b3)"
    labels = _label(problem, text)
    assert len(labels.consistency_violations) >= 2
    assert not labels.is_goal_complete
    assert not labels.overall_valid
