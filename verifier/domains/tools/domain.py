"""Tool-use domain definition + problem generator.

Tools (8), all consuming 1 api-quota per call; bookings also spend budget:
  authenticate(scope)            -> authed(scope)
  search_flights(trip)           needs authed(flights-scope)          -> flight-options(trip)
  book_flight(trip)              needs flight-options + auth          -> flight-booked(trip),  300 budget
  search_hotels(trip)            needs authed(hotels-scope)           -> hotel-options(trip)
  book_hotel(trip)               needs hotel-options + auth           -> hotel-booked(trip),   200 budget
  charge_card(trip)              needs flight-booked + payments auth  -> payment-captured(trip)
  send_confirmation_email(trip)  needs payment-captured + email auth  -> confirmation-sent(trip)
  create_calendar_event(trip)    needs flight-booked + calendar auth  -> event-created(trip)

Goals are downstream milestones (confirmation-sent, event-created,
hotel-booked, ...) whose prerequisite chains force realistic call ordering.
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
from verifier.domains.planner import PlannerTimeout, bfs_plan

DOMAIN_NAME = "tools"

SCOPES = ["flights-scope", "hotels-scope", "payments-scope", "email-scope", "calendar-scope"]

# goal-eligible milestone predicates, cheapest-first is not needed; the BFS
# planner finds optimal orderings anyway
_MILESTONES = [
    "flight-booked",
    "hotel-booked",
    "payment-captured",
    "confirmation-sent",
    "event-created",
]


def build_domain(api_quota: float = 20.0, budget: float = 2000.0) -> Domain:
    predicates = [
        PredicateDefinition(name="authed", arg_types=["scope"]),
        PredicateDefinition(name="flight-options", arg_types=["trip"]),
        PredicateDefinition(name="flight-booked", arg_types=["trip"]),
        PredicateDefinition(name="hotel-options", arg_types=["trip"]),
        PredicateDefinition(name="hotel-booked", arg_types=["trip"]),
        PredicateDefinition(name="payment-captured", arg_types=["trip"]),
        PredicateDefinition(name="confirmation-sent", arg_types=["trip"]),
        PredicateDefinition(name="event-created", arg_types=["trip"]),
        # scope identity predicates so tool preconditions can name the scope
        # they need without a constant-argument mechanism in the DSL
        PredicateDefinition(name="is-flights-scope", arg_types=["scope"]),
        PredicateDefinition(name="is-hotels-scope", arg_types=["scope"]),
        PredicateDefinition(name="is-payments-scope", arg_types=["scope"]),
        PredicateDefinition(name="is-email-scope", arg_types=["scope"]),
        PredicateDefinition(name="is-calendar-scope", arg_types=["scope"]),
    ]

    def tool(name, params, pre, add, deltas):
        return ActionSchema(
            name=name,
            parameters=params,
            preconditions=pre,
            add_effects=add,
            del_effects=[],
            resource_deltas=deltas,
        )

    t = lambda: Parameter(name="trip", type="trip")
    s = lambda: Parameter(name="scope", type="scope")

    action_schemas = [
        tool(
            "authenticate",
            [s()],
            [],
            [PredicateAtom(predicate="authed", args=["scope"])],
            {"api-quota": -1.0},
        ),
        tool(
            "search_flights",
            [t(), s()],
            [Literal.pos("is-flights-scope", "scope"), Literal.pos("authed", "scope")],
            [PredicateAtom(predicate="flight-options", args=["trip"])],
            {"api-quota": -1.0},
        ),
        tool(
            "book_flight",
            [t(), s()],
            [
                Literal.pos("is-flights-scope", "scope"),
                Literal.pos("authed", "scope"),
                Literal.pos("flight-options", "trip"),
            ],
            [PredicateAtom(predicate="flight-booked", args=["trip"])],
            {"api-quota": -1.0, "budget": -300.0},
        ),
        tool(
            "search_hotels",
            [t(), s()],
            [Literal.pos("is-hotels-scope", "scope"), Literal.pos("authed", "scope")],
            [PredicateAtom(predicate="hotel-options", args=["trip"])],
            {"api-quota": -1.0},
        ),
        tool(
            "book_hotel",
            [t(), s()],
            [
                Literal.pos("is-hotels-scope", "scope"),
                Literal.pos("authed", "scope"),
                Literal.pos("hotel-options", "trip"),
            ],
            [PredicateAtom(predicate="hotel-booked", args=["trip"])],
            {"api-quota": -1.0, "budget": -200.0},
        ),
        tool(
            "charge_card",
            [t(), s()],
            [
                Literal.pos("is-payments-scope", "scope"),
                Literal.pos("authed", "scope"),
                Literal.pos("flight-booked", "trip"),
            ],
            [PredicateAtom(predicate="payment-captured", args=["trip"])],
            {"api-quota": -1.0},
        ),
        tool(
            "send_confirmation_email",
            [t(), s()],
            [
                Literal.pos("is-email-scope", "scope"),
                Literal.pos("authed", "scope"),
                Literal.pos("payment-captured", "trip"),
            ],
            [PredicateAtom(predicate="confirmation-sent", args=["trip"])],
            {"api-quota": -1.0},
        ),
        tool(
            "create_calendar_event",
            [t(), s()],
            [
                Literal.pos("is-calendar-scope", "scope"),
                Literal.pos("authed", "scope"),
                Literal.pos("flight-booked", "trip"),
            ],
            [PredicateAtom(predicate="event-created", args=["trip"])],
            {"api-quota": -1.0},
        ),
    ]

    return Domain(
        name=DOMAIN_NAME,
        types=["trip", "scope"],
        predicates=predicates,
        action_schemas=action_schemas,
        resource_dimensions=[
            ResourceDimension(name="api-quota", initial=api_quota, cap=api_quota),
            ResourceDimension(name="budget", initial=budget, cap=budget),
        ],
    )


# goal milestones and their full prerequisite chains (for sizing)
_CHAIN_LEN = {
    "flight-booked": 3,  # auth, search, book
    "hotel-booked": 3,
    "payment-captured": 5,  # + charge (auth payments)
    "confirmation-sent": 7,
    "event-created": 5,
}


def generate_problem(
    seed: int,
    index: int = 0,
    min_plan_len: int = 3,
    max_plan_len: int = 8,
    max_attempts: int = 50,
) -> Tuple[Problem, List[GroundAction]]:
    """Deterministically generate a solvable tool-use Problem with an optimal
    plan of length in [min_plan_len, max_plan_len], plus that gold plan.

    Instead of a random walk (the tool graph is a DAG of milestones), we pick
    1-2 trips, sample goal milestones, and sometimes pre-grant auth scopes or
    pre-complete early calls in the initial state (a session already in
    progress), which varies plan length and start states.
    """
    domain = build_domain()

    for attempt in range(max_attempts):
        rng = random.Random(f"{DOMAIN_NAME}:{seed}:{index}:{attempt}")
        n_trips = rng.choice([1, 1, 2])  # bias toward single-trip tasks
        trips = [f"trip{i}" for i in range(1, n_trips + 1)]

        objects = [TypedObject(name=t, type="trip") for t in trips] + [
            TypedObject(name=sc, type="scope") for sc in SCOPES
        ]

        init = [
            PredicateAtom(predicate=f"is-{sc}", args=[sc]) for sc in SCOPES
        ]
        # sometimes the session starts partially authenticated / searched
        for sc in SCOPES:
            if rng.random() < 0.3:
                init.append(PredicateAtom(predicate="authed", args=[sc]))
        for trip in trips:
            if rng.random() < 0.25:
                init.append(PredicateAtom(predicate="flight-options", args=[trip]))
            if rng.random() < 0.2:
                init.append(PredicateAtom(predicate="hotel-options", args=[trip]))

        goal = []
        for trip in trips:
            n_goals = rng.choice([1, 1, 2])
            for milestone in rng.sample(_MILESTONES, n_goals):
                goal.append(Literal.pos(milestone, trip))
        if not goal:
            continue

        problem = Problem(
            name=f"{DOMAIN_NAME}-{seed}-{index}",
            domain=domain,
            objects=objects,
            init=init,
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
        f"[{min_plan_len}, {max_plan_len}] for seed={seed} index={index}"
    )
