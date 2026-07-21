"""Long-horizon problem generators (10/20/40/80-step bands).

BFS gold-planning is infeasible past ~10 steps (uninformed search over an
exponentially growing state space), so each domain gets a CONSTRUCTIVE gold
planner that scales the domain size (blocks / packages / trips) to the
target horizon and emits a plan that is valid by construction:

  - blocksworld: unstack everything to the table, then build the goal
    towers bottom-up (the classic 2-phase algorithm).
  - logistics: route each package sequentially (truck -> airport -> plane
    -> airport -> truck), tracking vehicle positions.
  - tools: authenticate each needed scope once, then run each trip's
    milestone prerequisite chain in dependency order.

IMPORTANT SEMANTIC DIFFERENCE from the short-horizon generators: these gold
plans are valid but NOT optimal (blocksworld in particular tears down
towers it could have reused). "Horizon H" is therefore a nominal band on
the REFERENCE plan length; LLM plans for the same problems may legitimately
be shorter. Resource-cap tightening against a wasteful reference plan is
correspondingly looser than at short horizons — recorded in
LIMITATIONS.md.

Every generated (problem, gold_plan) pair is validated at generation time
by strict simulation (verifier.schema.state.simulate raises on any
precondition/resource violation) plus a goal check, so a bug in a
constructive planner cannot silently ship an invalid reference plan.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from verifier.domains import blocksworld as bw
from verifier.domains import logistics as lg
from verifier.domains.tools import domain as tl
from verifier.schema import (
    Literal,
    PredicateAtom,
    Problem,
    ResourceDimension,
    TypedObject,
    initial_state,
)
from verifier.schema.state import GroundAction, goal_satisfied, simulate


def _band(target_len: int) -> Tuple[int, int]:
    """Accepted reference-plan lengths for a nominal horizon."""
    return math.ceil(0.85 * target_len), math.floor(1.2 * target_len)


def _validate(problem: Problem, plan: List[GroundAction]) -> None:
    """Strict-simulate the gold plan; raise if it is not flawless."""
    state = initial_state(problem)
    final = simulate(problem.domain, state, plan)  # raises on any violation
    if not goal_satisfied(problem, final):
        raise AssertionError(f"constructive gold plan misses goal for {problem.name}")


def _with_resources(problem: Problem, plan: List[GroundAction], slack: float = 2.0) -> Problem:
    """Size each resource dimension to slack x the gold plan's consumption
    (the Phase 2 tightening step later narrows this to 1.25x)."""
    schema_by_name = {a.name: a for a in problem.domain.action_schemas}
    spent: Dict[str, float] = {r.name: 0.0 for r in problem.domain.resource_dimensions}
    for ga in plan:
        for res, delta in schema_by_name[ga.schema_name].resource_deltas.items():
            if delta < 0:
                spent[res] += -delta
    dims = [
        ResourceDimension(
            name=r.name,
            initial=float(math.ceil(spent[r.name] * slack)) or r.initial,
            cap=float(math.ceil(spent[r.name] * slack)) or r.cap,
            floor=r.floor,
        )
        for r in problem.domain.resource_dimensions
    ]
    domain = problem.domain.model_copy(update={"resource_dimensions": dims})
    return Problem(
        name=problem.name,
        domain=domain,
        objects=problem.objects,
        init=problem.init,
        goal=problem.goal,
    )


# ---------------------------------------------------------------------------
# blocksworld
# ---------------------------------------------------------------------------

def _bw_constructive_plan(
    init_atoms: List[PredicateAtom], goal_on: List[Tuple[str, str]]
) -> List[GroundAction]:
    """Unstack everything to the table, then build goal towers bottom-up."""
    plan: List[GroundAction] = []
    on_map = {a.args[0]: a.args[1] for a in init_atoms if a.predicate == "on"}

    # phase 1: tear every tower down (top block first)
    while on_map:
        under = set(on_map.values())
        top = sorted(b for b in on_map if b not in under)[0]
        plan.append(GroundAction(schema_name="unstack", args=(top, on_map[top])))
        plan.append(GroundAction(schema_name="put-down", args=(top,)))
        del on_map[top]

    # phase 2: build goal towers bottom-up
    remaining = dict(goal_on)  # x -> y (x on y)
    placed: set = set()
    while remaining:
        progressed = False
        for x in sorted(remaining):
            y = remaining[x]
            # y must itself be fully placed (or never moves, i.e. not a key)
            if y not in remaining or y in placed:
                plan.append(GroundAction(schema_name="pick-up", args=(x,)))
                plan.append(GroundAction(schema_name="stack", args=(x, y)))
                placed.add(x)
                del remaining[x]
                progressed = True
                break
        if not progressed:  # cycle — caller retries with a new arrangement
            raise ValueError("cyclic goal arrangement")
    return plan


def generate_blocksworld(
    seed: int, index: int, target_len: int, max_attempts: int = 80
) -> Tuple[Problem, List[GroundAction]]:
    lo, hi = _band(target_len)
    for attempt in range(max_attempts):
        rng = random.Random(f"bw-h:{seed}:{index}:{target_len}:{attempt}")
        n_blocks = max(3, target_len // 2 + rng.randint(-2, 2))
        blocks = [f"b{i}" for i in range(1, n_blocks + 1)]
        objects = [TypedObject(name=b, type="block") for b in blocks]

        init_atoms = bw._random_arrangement(blocks, rng)
        goal_atoms = bw._random_arrangement(blocks, rng)
        goal_on = sorted(
            (a.args[0], a.args[1]) for a in goal_atoms if a.predicate == "on"
        )
        goal_table = sorted(a.args[0] for a in goal_atoms if a.predicate == "on-table")

        try:
            plan = _bw_constructive_plan(init_atoms, goal_on)
        except ValueError:
            continue
        if not lo <= len(plan) <= hi:
            continue

        goal = [Literal.pos("on", x, y) for x, y in goal_on] + [
            Literal.pos("on-table", b) for b in goal_table
        ]
        problem = Problem(
            name=f"blocksworld-h{target_len}-{seed}-{index}",
            domain=bw.build_domain(energy_cap=6.0 * len(plan)),
            objects=objects,
            init=init_atoms,
            goal=sorted(goal, key=str),
        )
        problem = _with_resources(problem, plan)
        _validate(problem, plan)
        return problem, plan
    raise RuntimeError(f"blocksworld horizon {target_len}: no problem after {max_attempts} attempts")


# ---------------------------------------------------------------------------
# logistics
# ---------------------------------------------------------------------------

def generate_logistics(
    seed: int, index: int, target_len: int, max_attempts: int = 80
) -> Tuple[Problem, List[GroundAction]]:
    lo, hi = _band(target_len)
    cities = ["city0", "city1"]
    locs = {c: [f"{c}-loc0", f"{c}-loc1"] for c in cities}
    airport = {c: f"{c}-loc0" for c in cities}
    loc_city = {l: c for c in cities for l in locs[c]}
    all_locs = [l for c in cities for l in locs[c]]
    trucks = {c: f"truck-{c}" for c in cities}

    for attempt in range(max_attempts):
        rng = random.Random(f"lg-h:{seed}:{index}:{target_len}:{attempt}")
        n_pkgs = max(1, round(target_len / 8) + rng.randint(-1, 1))
        pkgs = [f"pkg{j}" for j in range(1, n_pkgs + 1)]

        origin = {p: rng.choice(all_locs) for p in pkgs}
        dest = {p: rng.choice([l for l in all_locs if l != origin[p]]) for p in pkgs}

        truck_pos = {c: rng.choice(locs[c]) for c in cities}
        plane_pos = rng.choice([airport[c] for c in cities])
        init_truck_pos = dict(truck_pos)
        init_plane_pos = plane_pos

        plan: List[GroundAction] = []

        def drive(c: str, to: str) -> None:
            nonlocal truck_pos
            if truck_pos[c] != to:
                plan.append(
                    GroundAction(schema_name="drive-truck", args=(trucks[c], truck_pos[c], to, c))
                )
                truck_pos[c] = to

        def fly(to: str) -> None:
            nonlocal plane_pos
            if plane_pos != to:
                plan.append(GroundAction(schema_name="fly-plane", args=("plane1", plane_pos, to)))
                plane_pos = to

        for p in pkgs:
            o, d = origin[p], dest[p]
            co, cd = loc_city[o], loc_city[d]
            if co == cd:
                drive(co, o)
                plan.append(GroundAction(schema_name="load-truck", args=(p, trucks[co], o)))
                drive(co, d)
                plan.append(GroundAction(schema_name="unload-truck", args=(p, trucks[co], d)))
            else:
                # origin leg (skip the truck if the package starts at the airport)
                if o != airport[co]:
                    drive(co, o)
                    plan.append(GroundAction(schema_name="load-truck", args=(p, trucks[co], o)))
                    drive(co, airport[co])
                    plan.append(
                        GroundAction(schema_name="unload-truck", args=(p, trucks[co], airport[co]))
                    )
                fly(airport[co])
                plan.append(GroundAction(schema_name="load-plane", args=(p, "plane1", airport[co])))
                fly(airport[cd])
                plan.append(GroundAction(schema_name="unload-plane", args=(p, "plane1", airport[cd])))
                if d != airport[cd]:
                    drive(cd, airport[cd])
                    plan.append(GroundAction(schema_name="load-truck", args=(p, trucks[cd], airport[cd])))
                    drive(cd, d)
                    plan.append(GroundAction(schema_name="unload-truck", args=(p, trucks[cd], d)))
        if not lo <= len(plan) <= hi:
            continue

        objects = (
            [TypedObject(name=c, type="city") for c in cities]
            + [TypedObject(name=l, type="location") for l in all_locs]
            + [TypedObject(name=t, type="truck") for t in sorted(trucks.values())]
            + [TypedObject(name="plane1", type="plane")]
            + [TypedObject(name=p, type="package") for p in pkgs]
        )
        init_atoms = [PredicateAtom(predicate="in-city", args=[l, loc_city[l]]) for l in all_locs]
        init_atoms += [
            PredicateAtom(predicate="is-airport", args=[airport[c]]) for c in cities
        ]
        init_atoms += [
            PredicateAtom(predicate="at-truck", args=[trucks[c], init_truck_pos[c]])
            for c in cities
        ]
        init_atoms.append(PredicateAtom(predicate="at-plane", args=["plane1", init_plane_pos]))
        init_atoms += [
            PredicateAtom(predicate="at-package", args=[p, origin[p]]) for p in pkgs
        ]
        goal = sorted(
            (Literal.pos("at-package", p, dest[p]) for p in pkgs), key=str
        )
        problem = Problem(
            name=f"logistics-h{target_len}-{seed}-{index}",
            domain=lg.build_domain(fuel_cap=6.0 * len(plan), budget_cap=4.0 * len(plan)),
            objects=objects,
            init=init_atoms,
            goal=goal,
        )
        problem = _with_resources(problem, plan)
        _validate(problem, plan)
        return problem, plan
    raise RuntimeError(f"logistics horizon {target_len}: no problem after {max_attempts} attempts")


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

# per-milestone call chain (excluding auth): ordered prerequisite calls
_MILESTONE_CALLS = {
    "flight-booked": ["search_flights", "book_flight"],
    "hotel-booked": ["search_hotels", "book_hotel"],
    "payment-captured": ["search_flights", "book_flight", "charge_card"],
    "confirmation-sent": [
        "search_flights", "book_flight", "charge_card", "send_confirmation_email",
    ],
    "event-created": ["search_flights", "book_flight", "create_calendar_event"],
}
_CALL_SCOPE = {
    "search_flights": "flights-scope",
    "book_flight": "flights-scope",
    "search_hotels": "hotels-scope",
    "book_hotel": "hotels-scope",
    "charge_card": "payments-scope",
    "send_confirmation_email": "email-scope",
    "create_calendar_event": "calendar-scope",
}


def generate_tools(
    seed: int, index: int, target_len: int, max_attempts: int = 80
) -> Tuple[Problem, List[GroundAction]]:
    lo, hi = _band(target_len)
    for attempt in range(max_attempts):
        rng = random.Random(f"tl-h:{seed}:{index}:{target_len}:{attempt}")
        n_trips = max(1, round(target_len / 6) + rng.randint(-1, 1))
        trips = [f"trip{i}" for i in range(1, n_trips + 1)]

        goals: Dict[str, List[str]] = {}
        for trip in trips:
            n_goals = rng.choice([2, 3, 3])
            goals[trip] = sorted(rng.sample(tl._MILESTONES, n_goals))

        plan: List[GroundAction] = []
        authed: set = set()
        n_flights = n_hotels = 0
        for trip in trips:
            calls: List[str] = []
            for m in goals[trip]:
                for c in _MILESTONE_CALLS[m]:
                    if c not in calls:
                        calls.append(c)
            for c in calls:
                scope = _CALL_SCOPE[c]
                if scope not in authed:
                    plan.append(GroundAction(schema_name="authenticate", args=(scope,)))
                    authed.add(scope)
                plan.append(GroundAction(schema_name=c, args=(trip, scope)))
                if c == "book_flight":
                    n_flights += 1
                elif c == "book_hotel":
                    n_hotels += 1
        if not lo <= len(plan) <= hi:
            continue

        objects = [TypedObject(name=t, type="trip") for t in trips] + [
            TypedObject(name=sc, type="scope") for sc in tl.SCOPES
        ]
        init = [PredicateAtom(predicate=f"is-{sc}", args=[sc]) for sc in tl.SCOPES]
        goal = sorted(
            (
                Literal.pos(m, trip)
                for trip in trips
                for m in goals[trip]
            ),
            key=str,
        )
        problem = Problem(
            name=f"tools-h{target_len}-{seed}-{index}",
            domain=tl.build_domain(
                api_quota=2.0 * len(plan),
                budget=float(300 * n_flights + 200 * n_hotels + 500),
            ),
            objects=objects,
            init=init,
            goal=goal,
        )
        problem = _with_resources(problem, plan)
        _validate(problem, plan)
        return problem, plan
    raise RuntimeError(f"tools horizon {target_len}: no problem after {max_attempts} attempts")


HORIZON_GENERATORS = {
    "blocksworld": generate_blocksworld,
    "logistics": generate_logistics,
    "tools": generate_tools,
}
