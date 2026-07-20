"""Direct sanity checks for the BFS gold planner against hand-built problems."""

from verifier.domains.blocksworld import build_domain
from verifier.domains.planner import bfs_plan
from verifier.schema import Literal, PredicateAtom, Problem, TypedObject


def test_bfs_returns_empty_plan_when_goal_already_holds():
    domain = build_domain()
    problem = Problem(
        name="already-solved",
        domain=domain,
        objects=[TypedObject(name="b1", type="block")],
        init=[
            PredicateAtom(predicate="on-table", args=["b1"]),
            PredicateAtom(predicate="clear", args=["b1"]),
            PredicateAtom(predicate="crane-empty", args=[]),
        ],
        goal=[Literal.pos("on-table", "b1")],
    )
    assert bfs_plan(problem) == []


def test_bfs_finds_optimal_two_step_stack_plan():
    domain = build_domain()
    problem = Problem(
        name="two-step",
        domain=domain,
        objects=[TypedObject(name="b1", type="block"), TypedObject(name="b2", type="block")],
        init=[
            PredicateAtom(predicate="on-table", args=["b1"]),
            PredicateAtom(predicate="on-table", args=["b2"]),
            PredicateAtom(predicate="clear", args=["b1"]),
            PredicateAtom(predicate="clear", args=["b2"]),
            PredicateAtom(predicate="crane-empty", args=[]),
        ],
        goal=[Literal.pos("on", "b1", "b2")],
    )
    plan = bfs_plan(problem)
    assert plan is not None
    # pick-up(b1); stack(b1, b2) is the unique shortest solution
    assert len(plan) == 2
    assert plan[0].schema_name == "pick-up"
    assert plan[1].schema_name == "stack"


def test_bfs_returns_none_for_unreachable_goal():
    domain = build_domain()
    problem = Problem(
        name="unreachable",
        domain=domain,
        objects=[TypedObject(name="b1", type="block")],
        init=[
            PredicateAtom(predicate="on-table", args=["b1"]),
            PredicateAtom(predicate="clear", args=["b1"]),
            PredicateAtom(predicate="crane-empty", args=[]),
        ],
        # b1 can never be "on" itself
        goal=[Literal.pos("on", "b1", "b1")],
    )
    assert bfs_plan(problem, max_depth=5) is None