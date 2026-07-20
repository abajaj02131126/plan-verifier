"""A pure-Python BFS gold planner, plus shared helpers used by problem generators.

BFS explores states in order of increasing plan length, so the first goal
state found is reached by a shortest (optimal) plan. Action spaces here are
small (5-10 schemas, a handful of objects) so this stays laptop-fast; no
external planner dependency is needed.
"""

from __future__ import annotations

import random
from collections import deque
from itertools import product
from typing import Dict, List, Optional, Tuple

from verifier.schema.domain import Domain
from verifier.schema.problem import Problem, TypedObject
from verifier.schema.state import (
    GroundAction,
    State,
    applicable,
    apply_action,
    goal_satisfied,
    initial_state,
)


class PlannerTimeout(Exception):
    """Raised when BFS exceeds its expansion budget without finding a goal."""


def candidate_ground_actions(domain: Domain, objects: List[TypedObject]) -> List[GroundAction]:
    """All ground actions type-consistent with ``objects``, with distinct
    parameters bound to distinct objects (no schema in our domains needs an
    action to reuse the same object across two different parameters)."""
    objects_by_type: Dict[str, List[str]] = {}
    for obj in objects:
        objects_by_type.setdefault(obj.type, []).append(obj.name)

    candidates: List[GroundAction] = []
    for schema in domain.action_schemas:
        param_type_lists = [objects_by_type.get(p.type, []) for p in schema.parameters]
        if any(not lst for lst in param_type_lists):
            continue
        for combo in product(*param_type_lists):
            if len(set(combo)) != len(combo):
                continue
            candidates.append(GroundAction(schema_name=schema.name, args=combo))
    return candidates


def bfs_plan(
    problem: Problem,
    max_depth: int = 10,
    max_expansions: int = 200_000,
) -> Optional[List[GroundAction]]:
    """Shortest valid, resource-feasible plan from problem.init to problem.goal,
    or None if none exists within ``max_depth``."""
    domain = problem.domain
    start = initial_state(problem)
    if goal_satisfied(problem, start):
        return []

    candidates = candidate_ground_actions(domain, problem.objects)
    frontier: deque[Tuple[State, List[GroundAction]]] = deque([(start, [])])
    visited = {start}
    expansions = 0

    while frontier:
        state, path = frontier.popleft()
        if len(path) >= max_depth:
            continue
        for ga in candidates:
            expansions += 1
            if expansions > max_expansions:
                raise PlannerTimeout(f"exceeded {max_expansions} expansions")
            if not applicable(domain, state, ga):
                continue
            next_state = apply_action(domain, state, ga)
            if next_state in visited:
                continue
            next_path = path + [ga]
            if goal_satisfied(problem, next_state):
                return next_path
            visited.add(next_state)
            frontier.append((next_state, next_path))

    return None


def random_walk(
    domain: Domain,
    objects: List[TypedObject],
    start: State,
    rng: random.Random,
    steps: int,
) -> Tuple[State, List[GroundAction]]:
    """Apply up to ``steps`` random applicable actions from ``start``, stopping
    early if no action is applicable. Used by problem generators to reach a
    guaranteed-reachable state whose atoms become the goal."""
    candidates = candidate_ground_actions(domain, objects)
    state = start
    plan: List[GroundAction] = []
    for _ in range(steps):
        options = [a for a in candidates if applicable(domain, state, a)]
        if not options:
            break
        action = rng.choice(options)
        state = apply_action(domain, state, action)
        plan.append(action)
    return state, plan