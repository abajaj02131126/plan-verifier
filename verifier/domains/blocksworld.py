"""Blocksworld-style domain extended with a "crane energy" resource capped per plan.

Standard 4-operator blocksworld (pick-up, put-down, stack, unstack) over a
single "crane" that must be empty to pick up and holds at most one block.
Every action consumes crane energy from a per-plan budget; the budget can
run out mid-plan, making some otherwise-legal plans resource-infeasible.
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

DOMAIN_NAME = "blocksworld"


def build_domain(energy_cap: float = 24.0) -> Domain:
    predicates = [
        PredicateDefinition(name="on", arg_types=["block", "block"]),
        PredicateDefinition(name="on-table", arg_types=["block"]),
        PredicateDefinition(name="clear", arg_types=["block"]),
        PredicateDefinition(name="holding", arg_types=["block"]),
        PredicateDefinition(name="crane-empty", arg_types=[]),
    ]

    action_schemas = [
        ActionSchema(
            name="pick-up",
            parameters=[Parameter(name="x", type="block")],
            preconditions=[
                Literal.pos("on-table", "x"),
                Literal.pos("clear", "x"),
                Literal.pos("crane-empty"),
            ],
            add_effects=[PredicateAtom(predicate="holding", args=["x"])],
            del_effects=[
                PredicateAtom(predicate="on-table", args=["x"]),
                PredicateAtom(predicate="clear", args=["x"]),
                PredicateAtom(predicate="crane-empty", args=[]),
            ],
            resource_deltas={"energy": -1.0},
        ),
        ActionSchema(
            name="put-down",
            parameters=[Parameter(name="x", type="block")],
            preconditions=[Literal.pos("holding", "x")],
            add_effects=[
                PredicateAtom(predicate="on-table", args=["x"]),
                PredicateAtom(predicate="clear", args=["x"]),
                PredicateAtom(predicate="crane-empty", args=[]),
            ],
            del_effects=[PredicateAtom(predicate="holding", args=["x"])],
            resource_deltas={"energy": -1.0},
        ),
        ActionSchema(
            name="stack",
            parameters=[Parameter(name="x", type="block"), Parameter(name="y", type="block")],
            preconditions=[Literal.pos("holding", "x"), Literal.pos("clear", "y")],
            add_effects=[
                PredicateAtom(predicate="on", args=["x", "y"]),
                PredicateAtom(predicate="clear", args=["x"]),
                PredicateAtom(predicate="crane-empty", args=[]),
            ],
            del_effects=[
                PredicateAtom(predicate="holding", args=["x"]),
                PredicateAtom(predicate="clear", args=["y"]),
            ],
            resource_deltas={"energy": -2.0},
        ),
        ActionSchema(
            name="unstack",
            parameters=[Parameter(name="x", type="block"), Parameter(name="y", type="block")],
            preconditions=[
                Literal.pos("on", "x", "y"),
                Literal.pos("clear", "x"),
                Literal.pos("crane-empty"),
            ],
            add_effects=[
                PredicateAtom(predicate="holding", args=["x"]),
                PredicateAtom(predicate="clear", args=["y"]),
            ],
            del_effects=[
                PredicateAtom(predicate="on", args=["x", "y"]),
                PredicateAtom(predicate="clear", args=["x"]),
                PredicateAtom(predicate="crane-empty", args=[]),
            ],
            resource_deltas={"energy": -2.0},
        ),
    ]

    return Domain(
        name=DOMAIN_NAME,
        types=["block"],
        predicates=predicates,
        action_schemas=action_schemas,
        resource_dimensions=[ResourceDimension(name="energy", initial=energy_cap, cap=energy_cap)],
    )


def _random_arrangement(blocks: List[str], rng: random.Random) -> List[PredicateAtom]:
    """A random valid blocksworld state: some blocks on the table, some stacked."""
    order = list(blocks)
    rng.shuffle(order)

    on_table: List[str] = []
    on_pairs: List[Tuple[str, str]] = []
    clear_tops: List[str] = []  # blocks with nothing currently on them

    for b in order:
        if clear_tops and rng.random() < 0.5:
            target = rng.choice(clear_tops)
            on_pairs.append((b, target))
            clear_tops.remove(target)
        else:
            on_table.append(b)
        clear_tops.append(b)

    atoms = [PredicateAtom(predicate="on-table", args=[b]) for b in on_table]
    atoms += [PredicateAtom(predicate="on", args=[x, y]) for x, y in on_pairs]
    atoms += [PredicateAtom(predicate="clear", args=[b]) for b in clear_tops]
    atoms.append(PredicateAtom(predicate="crane-empty", args=[]))
    return atoms


def generate_problem(
    seed: int,
    index: int = 0,
    min_plan_len: int = 3,
    max_plan_len: int = 8,
    min_blocks: int = 3,
    max_blocks: int = 5,
    max_attempts: int = 50,
) -> Tuple[Problem, List[GroundAction]]:
    """Deterministically (given seed, index) generate a solvable blocksworld
    Problem whose optimal (BFS-shortest) plan has length in
    [min_plan_len, max_plan_len], plus that gold plan."""
    energy_cap = 2.0 * (max_plan_len + 5)
    domain = build_domain(energy_cap=energy_cap)

    for attempt in range(max_attempts):
        rng = random.Random(f"{DOMAIN_NAME}:{seed}:{index}:{attempt}")
        num_blocks = rng.randint(min_blocks, max_blocks)
        blocks = [f"b{i}" for i in range(1, num_blocks + 1)]
        objects = [TypedObject(name=b, type="block") for b in blocks]

        init_atoms = _random_arrangement(blocks, rng)
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
        goal_atoms = sorted(a for a in final_state.atoms if a[0] in ("on", "on-table"))
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