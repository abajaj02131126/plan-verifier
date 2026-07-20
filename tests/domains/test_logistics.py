"""Generation determinism + gold-planner correctness for the logistics domain."""

from verifier.domains.logistics import generate_problem
from verifier.schema import goal_satisfied, initial_state, simulate

MIN_LEN, MAX_LEN = 3, 8


def test_generation_is_deterministic_under_fixed_seed():
    p1, plan1 = generate_problem(seed=42, index=0)
    p2, plan2 = generate_problem(seed=42, index=0)
    assert p1.model_dump() == p2.model_dump()
    assert plan1 == plan2


def test_different_indices_can_differ():
    problems = [generate_problem(seed=42, index=i)[0] for i in range(5)]
    dumps = [p.model_dump() for p in problems]
    assert len(set(str(d) for d in dumps)) > 1


def test_gold_plans_are_valid_and_resource_feasible():
    for i in range(20):
        problem, plan = generate_problem(seed=11, index=i)
        assert MIN_LEN <= len(plan) <= MAX_LEN

        state = initial_state(problem)
        final_state = simulate(problem.domain, state, plan)
        assert goal_satisfied(problem, final_state)

        resources = final_state.resource_dict()
        for dim in problem.domain.resource_dimensions:
            value = resources[dim.name]
            assert dim.floor <= value <= dim.cap


def test_custom_plan_length_range_is_respected():
    problem, plan = generate_problem(seed=3, index=0, min_plan_len=4, max_plan_len=6)
    assert 4 <= len(plan) <= 6
    final_state = simulate(problem.domain, initial_state(problem), plan)
    assert goal_satisfied(problem, final_state)