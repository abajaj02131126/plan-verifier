"""Tool-use domain: generation determinism, gold-plan validity, and
auto-labeling of the tool-specific flaw taxonomy via the shared labeler."""

from verifier.domains.tools import build_domain, generate_problem
from verifier.generation.labeler import label_plan
from verifier.generation.parser import parse_plan
from verifier.schema import goal_satisfied, initial_state, simulate


def test_generation_is_deterministic_under_fixed_seed():
    p1, plan1 = generate_problem(seed=42, index=0)
    p2, plan2 = generate_problem(seed=42, index=0)
    assert p1.model_dump() == p2.model_dump()
    assert plan1 == plan2


def test_gold_plans_are_valid_and_resource_feasible():
    for i in range(15):
        problem, plan = generate_problem(seed=7, index=i)
        assert 3 <= len(plan) <= 8
        final = simulate(problem.domain, initial_state(problem), plan)
        assert goal_satisfied(problem, final)
        for dim in problem.domain.resource_dimensions:
            value = final.resource_dict()[dim.name]
            assert dim.floor <= value <= dim.cap


def _fresh_problem():
    """A problem with nothing pre-granted, for hand-written flaw fixtures."""
    for i in range(50):
        p, plan = generate_problem(seed=99, index=i)
        if not any(a.predicate in ("authed", "flight-options", "hotel-options") for a in p.init):
            return p, plan
    raise AssertionError("no fully-fresh problem found in 50 tries")


def test_missing_prerequisite_call_is_labeled_inconsistent():
    problem, _ = _fresh_problem()
    # book_flight without ever calling search_flights (and without auth)
    labels = label_plan(problem, parse_plan(
        "Step 1: authenticate(flights-scope)\nStep 2: book_flight(trip1, flights-scope)",
        problem.domain,
    ))
    assert not labels.is_consistent
    assert any("flight-options" in v for v in labels.consistency_violations)


def test_invalid_call_ordering_is_labeled_inconsistent():
    problem, _ = _fresh_problem()
    # charge before booking
    labels = label_plan(problem, parse_plan(
        "Step 1: authenticate(payments-scope)\nStep 2: charge_card(trip1, payments-scope)",
        problem.domain,
    ))
    assert not labels.is_consistent
    assert any("flight-booked" in v for v in labels.consistency_violations)


def test_wrong_arg_type_is_labeled_inconsistent():
    problem, _ = _fresh_problem()
    labels = label_plan(problem, parse_plan(
        "Step 1: authenticate(trip1)", problem.domain  # a trip is not a scope
    ))
    assert not labels.is_consistent
    assert any("type" in v for v in labels.consistency_violations)


def test_quota_exceeded_is_labeled_infeasible():
    domain = build_domain(api_quota=2.0)  # only 2 calls allowed
    problem, _ = _fresh_problem()
    tight = problem.model_copy(update={"domain": domain})
    labels = label_plan(tight, parse_plan(
        "Step 1: authenticate(flights-scope)\n"
        "Step 2: search_flights(trip1, flights-scope)\n"
        "Step 3: book_flight(trip1, flights-scope)",
        tight.domain,
    ))
    assert not labels.is_resource_feasible
    assert any("api-quota" in v for v in labels.resource_violations)


def test_budget_exceeded_is_labeled_infeasible():
    domain = build_domain(budget=250.0)  # one flight (300) blows it
    problem, _ = _fresh_problem()
    tight = problem.model_copy(update={"domain": domain})
    labels = label_plan(tight, parse_plan(
        "Step 1: authenticate(flights-scope)\n"
        "Step 2: search_flights(trip1, flights-scope)\n"
        "Step 3: book_flight(trip1, flights-scope)",
        tight.domain,
    ))
    assert not labels.is_resource_feasible
    assert any("budget" in v for v in labels.resource_violations)
