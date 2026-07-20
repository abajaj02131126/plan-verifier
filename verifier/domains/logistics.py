"""Logistics-style domain: trucks move packages within a city, a plane moves
packages between cities' airports. Extended with "fuel" (consumed by
movement) and "budget" (consumed by every action) resource dimensions.

Kept intentionally small (2 cities x 2 locations, 1 truck per city, 1 plane,
2-3 packages) so BFS gold-planning and generation stay laptop-fast.
"""

from __future__ import annotations

import random
from typing import List, Tuple

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
    initial_state,
)
from verifier.schema.state import GroundAction
from verifier.domains.planner import PlannerTimeout, bfs_plan, random_walk

DOMAIN_NAME = "logistics"


def build_domain(fuel_cap: float = 39.0, budget_cap: float = 26.0) -> Domain:
    predicates = [
        PredicateDefinition(name="at-package", arg_types=["package", "location"]),
        PredicateDefinition(name="at-truck", arg_types=["truck", "location"]),
        PredicateDefinition(name="at-plane", arg_types=["plane", "location"]),
        PredicateDefinition(name="in-truck", arg_types=["package", "truck"]),
        PredicateDefinition(name="in-plane", arg_types=["package", "plane"]),
        PredicateDefinition(name="in-city", arg_types=["location", "city"]),
        PredicateDefinition(name="is-airport", arg_types=["location"]),
    ]

    action_schemas = [
        ActionSchema(
            name="load-truck",
            parameters=[
                Parameter(name="pkg", type="package"),
                Parameter(name="truck", type="truck"),
                Parameter(name="loc", type="location"),
            ],
            preconditions=[
                Literal.pos("at-package", "pkg", "loc"),
                Literal.pos("at-truck", "truck", "loc"),
            ],
            add_effects=[PredicateAtom(predicate="in-truck", args=["pkg", "truck"])],
            del_effects=[PredicateAtom(predicate="at-package", args=["pkg", "loc"])],
            resource_deltas={"budget": -1.0},
        ),
        ActionSchema(
            name="unload-truck",
            parameters=[
                Parameter(name="pkg", type="package"),
                Parameter(name="truck", type="truck"),
                Parameter(name="loc", type="location"),
            ],
            preconditions=[
                Literal.pos("in-truck", "pkg", "truck"),
                Literal.pos("at-truck", "truck", "loc"),
            ],
            add_effects=[PredicateAtom(predicate="at-package", args=["pkg", "loc"])],
            del_effects=[PredicateAtom(predicate="in-truck", args=["pkg", "truck"])],
            resource_deltas={"budget": -1.0},
        ),
        ActionSchema(
            name="load-plane",
            parameters=[
                Parameter(name="pkg", type="package"),
                Parameter(name="plane", type="plane"),
                Parameter(name="loc", type="location"),
            ],
            preconditions=[
                Literal.pos("at-package", "pkg", "loc"),
                Literal.pos("at-plane", "plane", "loc"),
                Literal.pos("is-airport", "loc"),
            ],
            add_effects=[PredicateAtom(predicate="in-plane", args=["pkg", "plane"])],
            del_effects=[PredicateAtom(predicate="at-package", args=["pkg", "loc"])],
            resource_deltas={"budget": -1.0},
        ),
        ActionSchema(
            name="unload-plane",
            parameters=[
                Parameter(name="pkg", type="package"),
                Parameter(name="plane", type="plane"),
                Parameter(name="loc", type="location"),
            ],
            preconditions=[
                Literal.pos("in-plane", "pkg", "plane"),
                Literal.pos("at-plane", "plane", "loc"),
            ],
            add_effects=[PredicateAtom(predicate="at-package", args=["pkg", "loc"])],
            del_effects=[PredicateAtom(predicate="in-plane", args=["pkg", "plane"])],
            resource_deltas={"budget": -1.0},
        ),
        ActionSchema(
            name="drive-truck",
            parameters=[
                Parameter(name="truck", type="truck"),
                Parameter(name="from_loc", type="location"),
                Parameter(name="to_loc", type="location"),
                Parameter(name="city", type="city"),
            ],
            preconditions=[
                Literal.pos("at-truck", "truck", "from_loc"),
                Literal.pos("in-city", "from_loc", "city"),
                Literal.pos("in-city", "to_loc", "city"),
            ],
            add_effects=[PredicateAtom(predicate="at-truck", args=["truck", "to_loc"])],
            del_effects=[PredicateAtom(predicate="at-truck", args=["truck", "from_loc"])],
            resource_deltas={"fuel": -1.0, "budget": -1.0},
        ),
        ActionSchema(
            name="fly-plane",
            parameters=[
                Parameter(name="plane", type="plane"),
                Parameter(name="from_loc", type="location"),
                Parameter(name="to_loc", type="location"),
            ],
            preconditions=[
                Literal.pos("at-plane", "plane", "from_loc"),
                Literal.pos("is-airport", "from_loc"),
                Literal.pos("is-airport", "to_loc"),
            ],
            add_effects=[PredicateAtom(predicate="at-plane", args=["plane", "to_loc"])],
            del_effects=[PredicateAtom(predicate="at-plane", args=["plane", "from_loc"])],
            resource_deltas={"fuel": -3.0, "budget": -2.0},
        ),
    ]

    return Domain(
        name=DOMAIN_NAME,
        types=["package", "truck", "plane", "location", "city"],
        predicates=predicates,
        action_schemas=action_schemas,
        resource_dimensions=[
            ResourceDimension(name="fuel", initial=fuel_cap, cap=fuel_cap),
            ResourceDimension(name="budget", initial=budget_cap, cap=budget_cap),
        ],
    )


_NUM_CITIES = 2
_LOCS_PER_CITY = 2


def generate_problem(
    seed: int,
    index: int = 0,
    min_plan_len: int = 3,
    max_plan_len: int = 8,
    min_packages: int = 2,
    max_packages: int = 3,
    max_attempts: int = 50,
) -> Tuple[Problem, List[GroundAction]]:
    """Deterministically (given seed, index) generate a solvable logistics
    Problem whose optimal (BFS-shortest) plan has length in
    [min_plan_len, max_plan_len], plus that gold plan."""
    fuel_cap = 3.0 * (max_plan_len + 5)
    budget_cap = 2.0 * (max_plan_len + 5)
    domain = build_domain(fuel_cap=fuel_cap, budget_cap=budget_cap)

    cities = [f"city{c}" for c in range(_NUM_CITIES)]
    locations: List[str] = []
    loc_city = {}
    airports = set()
    for c in cities:
        for l in range(_LOCS_PER_CITY):
            loc = f"{c}-loc{l}"
            locations.append(loc)
            loc_city[loc] = c
            if l == 0:
                airports.add(loc)
    trucks = [f"truck-{c}" for c in cities]
    planes = ["plane1"]

    for attempt in range(max_attempts):
        rng = random.Random(f"{DOMAIN_NAME}:{seed}:{index}:{attempt}")
        num_packages = rng.randint(min_packages, max_packages)
        packages = [f"pkg{j}" for j in range(1, num_packages + 1)]

        objects = (
            [TypedObject(name=c, type="city") for c in cities]
            + [TypedObject(name=l, type="location") for l in locations]
            + [TypedObject(name=t, type="truck") for t in trucks]
            + [TypedObject(name=p, type="plane") for p in planes]
            + [TypedObject(name=pkg, type="package") for pkg in packages]
        )

        init_atoms = [PredicateAtom(predicate="in-city", args=[l, loc_city[l]]) for l in locations]
        init_atoms += [PredicateAtom(predicate="is-airport", args=[l]) for l in sorted(airports)]
        for truck, city in zip(trucks, cities):
            home_locs = [l for l in locations if loc_city[l] == city]
            init_atoms.append(PredicateAtom(predicate="at-truck", args=[truck, rng.choice(home_locs)]))
        for plane in planes:
            init_atoms.append(PredicateAtom(predicate="at-plane", args=[plane, rng.choice(sorted(airports))]))
        for pkg in packages:
            init_atoms.append(PredicateAtom(predicate="at-package", args=[pkg, rng.choice(locations)]))

        stub = Problem(
            name=f"{DOMAIN_NAME}-{seed}-{index}-stub",
            domain=domain,
            objects=objects,
            init=init_atoms,
            goal=[],
        )
        start = initial_state(stub)

        walk_steps = rng.randint(max_plan_len, max_plan_len + 3)
        final_state, _walk_plan = random_walk(domain, objects, start, rng, walk_steps)

        # sorted(): frozenset iteration order depends on PYTHONHASHSEED, which
        # varies per process, so this must be sorted for cross-run determinism.
        goal_atoms = sorted(a for a in final_state.atoms if a[0] == "at-package")
        if not goal_atoms:
            continue
        goal = [Literal.pos(*a) for a in goal_atoms]

        problem = Problem(
            name=f"{DOMAIN_NAME}-{seed}-{index}",
            domain=domain,
            objects=objects,
            init=init_atoms,
            goal=goal,
        )

        try:
            gold_plan = bfs_plan(problem, max_depth=max_plan_len + 2)
        except PlannerTimeout:
            continue
        if gold_plan is None:
            continue
        if min_plan_len <= len(gold_plan) <= max_plan_len:
            return problem, gold_plan

    raise RuntimeError(
        f"could not generate a {DOMAIN_NAME} problem with optimal plan length in "
        f"[{min_plan_len}, {max_plan_len}] for seed={seed} index={index} "
        f"after {max_attempts} attempts"
    )