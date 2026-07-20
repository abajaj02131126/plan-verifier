"""Render a Problem as a natural-language planning prompt, under several
"prompt conditions" that make specific LLM failure modes more or less likely
(PROJECT_SPEC.md section 5: goal omission, precondition slip, resource
overrun, hallucinated action/effect).

The prompt asks for a strict output format —

    Step 1: action-name(arg1, arg2)

— constrained enough that the Phase 2 rule-based parser can reliably recover
structured actions without the full LLM extractor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List

from verifier.schema import Problem

# Per-domain prose descriptions of each action: what it does, its
# preconditions, and its resource cost. Written as prose, NOT a schema dump.
_ACTION_PROSE: Dict[str, str] = {
    "blocksworld": """\
You control a single crane that can hold at most one block at a time.
Available actions (each line is: name(args) — what it does; requirements; energy cost):
- pick-up(x) — pick up block x from the table; x must be on the table, clear (nothing on it), and the crane must be empty; costs 1 energy.
- put-down(x) — put the held block x down on the table; you must be holding x; costs 1 energy.
- stack(x, y) — stack the held block x onto block y; you must be holding x and y must be clear; costs 2 energy.
- unstack(x, y) — pick up block x from on top of block y; x must be on y, x must be clear, and the crane must be empty; costs 2 energy.""",
    "logistics": """\
Trucks drive between locations within their own city; the plane flies between airports.
Available actions (each line is: name(args) — what it does; requirements; costs):
- load-truck(pkg, truck, loc) — load package pkg into truck at loc; both must be at loc; costs 1 budget.
- unload-truck(pkg, truck, loc) — unload pkg from truck at loc; pkg must be in the truck and the truck at loc; costs 1 budget.
- load-plane(pkg, plane, loc) — load pkg into the plane at loc; both must be at loc and loc must be an airport; costs 1 budget.
- unload-plane(pkg, plane, loc) — unload pkg from the plane at loc; pkg must be in the plane and the plane at loc; costs 1 budget.
- drive-truck(truck, from, to, city) — drive truck from one location to another; both locations must be in the same city as given; costs 1 fuel and 1 budget.
- fly-plane(plane, from, to) — fly the plane between two airports; costs 3 fuel and 2 budget.""",
    "tools": """\
You are operating a travel-booking assistant that calls API tools. Every tool
call consumes 1 api-quota. A tool that needs an auth scope only works if you
have already called authenticate on that exact scope (unless the session
started with it granted).
Available tools (each line is: name(args) — what it does; requirements; costs):
- authenticate(scope) — acquire the given auth scope; no requirements; costs 1 api-quota.
- search_flights(trip, scope) — search flights for the trip using flights-scope; requires authed flights-scope; costs 1 api-quota.
- book_flight(trip, scope) — book a flight; requires authed flights-scope and prior flight search results for this trip; costs 1 api-quota and 300 budget.
- search_hotels(trip, scope) — search hotels; requires authed hotels-scope; costs 1 api-quota.
- book_hotel(trip, scope) — book a hotel; requires authed hotels-scope and prior hotel search results for this trip; costs 1 api-quota and 200 budget.
- charge_card(trip, scope) — capture payment; requires authed payments-scope and a booked flight for this trip; costs 1 api-quota.
- send_confirmation_email(trip, scope) — email the confirmation; requires authed email-scope and captured payment for this trip; costs 1 api-quota.
- create_calendar_event(trip, scope) — add the trip to the calendar; requires authed calendar-scope and a booked flight for this trip; costs 1 api-quota.""",
}

# Plausible-sounding but INVALID actions per domain, used by the distractor
# condition to tempt the model into hallucinating unsupported actions.
_DISTRACTOR_PROSE: Dict[str, str] = {
    "blocksworld": """\
- move(x, y) — move block x directly onto block y in one step; costs 1 energy.
- swap(x, y) — swap the positions of blocks x and y; costs 2 energy.""",
    "logistics": """\
- teleport-package(pkg, loc) — instantly move a package to any location; costs 2 budget.
- transfer(pkg, truck1, truck2) — move a package directly between two trucks; costs 1 budget.""",
    "tools": """\
- book_trip(trip) — search, book, pay, and confirm the whole trip in one call; costs 2 api-quota and 500 budget.
- auto_authenticate() — acquire every auth scope at once; costs 1 api-quota.""",
}


def _describe_init(problem: Problem) -> str:
    lines = []
    for atom in problem.init:
        lines.append(f"- {atom.predicate}({', '.join(atom.args)})" if atom.args else f"- {atom.predicate}")
    return "\n".join(lines)


def _describe_goal_literals(problem: Problem, skip_indices: frozenset[int] = frozenset()) -> str:
    lines = []
    for i, lit in enumerate(problem.goal):
        if i in skip_indices:
            continue
        atom = lit.atom
        neg = "NOT " if lit.negated else ""
        lines.append(f"- {neg}{atom.predicate}({', '.join(atom.args)})")
    return "\n".join(lines)


def _describe_resources(problem: Problem) -> str:
    lines = []
    for dim in problem.domain.resource_dimensions:
        lines.append(
            f"- {dim.name}: you start with {dim.initial:g} and may never go below "
            f"{dim.floor:g}. The total {dim.name} consumed by your plan must not "
            f"exceed {dim.initial:g}."
        )
    return "\n".join(lines)


_FORMAT_INSTRUCTIONS = """\
Respond with ONLY the plan, one action per line, in exactly this format:

Step 1: action-name(arg1, arg2)
Step 2: action-name(arg1)

Use only the action names and object names given above. Do not add commentary,
explanations, or blank lines between steps."""


def _base_prompt(
    problem: Problem,
    action_prose: str,
    goal_text: str,
    resource_text: str | None,
) -> str:
    objects_by_type: Dict[str, List[str]] = {}
    for o in problem.objects:
        objects_by_type.setdefault(o.type, []).append(o.name)
    objects_text = "\n".join(
        f"- {t}: {', '.join(names)}" for t, names in sorted(objects_by_type.items())
    )

    parts = [
        "You are a planning assistant. Produce a plan that achieves the goal below.",
        f"\nObjects:\n{objects_text}",
        f"\nActions:\n{action_prose}",
    ]
    if resource_text:
        parts.append(f"\nResource limits:\n{resource_text}")
    parts.append(f"\nInitial state (everything true at the start; anything not listed is false):\n{_describe_init(problem)}")
    parts.append(f"\nGoal (ALL of the following must hold at the end):\n{goal_text}")
    parts.append(f"\n{_FORMAT_INSTRUCTIONS}")
    return "\n".join(parts)


def render_baseline(problem: Problem, rng: random.Random) -> str:
    """Baseline condition: a complete, honest prompt — full goal, full action
    descriptions, resource limits stated. Failures that occur under this
    condition are 'naturally occurring' LLM planning errors, used to calibrate
    that the induced conditions below are not wildly out of distribution."""
    domain = problem.domain.name
    return _base_prompt(
        problem,
        _ACTION_PROSE[domain],
        _describe_goal_literals(problem),
        _describe_resources(problem),
    )


def render_goal_omission(problem: Problem, rng: random.Random) -> str:
    """Goal-omission-inducing condition: one goal conjunct is silently dropped
    from the prompt (chosen by the per-record rng), while ground-truth labels
    are still computed against the FULL goal. Elicits goal-incompleteness —
    the model plans faithfully for what it was shown, satisfying a strict
    subset of the real goal. Mirrors real deployments where the NL goal
    under-specifies the true requirements."""
    domain = problem.domain.name
    if len(problem.goal) > 1:
        skip = frozenset({rng.randrange(len(problem.goal))})
    else:
        skip = frozenset()
    return _base_prompt(
        problem,
        _ACTION_PROSE[domain],
        _describe_goal_literals(problem, skip_indices=skip),
        _describe_resources(problem),
    )


def render_resource_blind(problem: Problem, rng: random.Random) -> str:
    """Resource-blind condition: the resource caps are not mentioned at all
    (action costs are also stripped from the action prose). Elicits
    resource-infeasibility — the model has no reason to prefer cheap actions
    or short plans, so wasteful but logically-correct plans overrun the cap.
    Mirrors real deployments where budget/quota constraints live outside the
    prompt."""
    domain = problem.domain.name
    prose = _ACTION_PROSE[domain]
    # Strip the "; costs ..." suffix from each action line so cost information
    # is fully absent, not just the cap.
    stripped = "\n".join(
        line.split("; costs")[0].rstrip(".") + "." if "; costs" in line else line
        for line in prose.splitlines()
    )
    return _base_prompt(
        problem,
        stripped,
        _describe_goal_literals(problem),
        resource_text=None,
    )


def render_distractor(problem: Problem, rng: random.Random) -> str:
    """Distractor condition: plausible-sounding but invalid actions (e.g. a
    one-step 'move(x, y)' in blocksworld) are listed alongside the real ones.
    Elicits hallucinated-action use — the invalid actions are strictly more
    convenient, tempting the model to use operators the executor does not
    support. Mirrors real deployments where the model's prior (from seeing
    other planning domains) conflicts with this executor's actual API."""
    domain = problem.domain.name
    prose = _ACTION_PROSE[domain] + "\n" + _DISTRACTOR_PROSE[domain]
    return _base_prompt(
        problem,
        prose,
        _describe_goal_literals(problem),
        _describe_resources(problem),
    )


@dataclass(frozen=True)
class Condition:
    name: str
    render: Callable[[Problem, random.Random], str]


CONDITIONS: Dict[str, Condition] = {
    "baseline": Condition("baseline", render_baseline),
    "goal_omission": Condition("goal_omission", render_goal_omission),
    "resource_blind": Condition("resource_blind", render_resource_blind),
    "distractor": Condition("distractor", render_distractor),
}
