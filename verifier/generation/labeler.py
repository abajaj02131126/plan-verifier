"""Automatic ground-truth labeler for candidate plans. [SOUNDNESS-CRITICAL]

Given a Problem (whose domain model we fully know) and a parsed candidate
plan, simulate the plan and label it with:
  - is_consistent          (every step's action exists, args typecheck, and
                            preconditions hold in the state where it executes)
  - is_goal_complete       (+ which goal conjuncts are unmet at the end)
  - is_resource_feasible   (+ which resource / at which step violated)
  - overall_valid          (all three)

This labeler is the ground-truth oracle that both the symbolic checker
(Phase 4) and the learned layer (Phase 5) are evaluated against, so it is
deliberately an INDEPENDENT implementation of forward simulation — it does
not import or reuse verifier/schema/state.py's progression code. Phase 4's
verifier and this labeler must agree; because they are separate code paths,
agreement is real evidence of correctness rather than a tautology.

Simulation semantics (documented design decisions):
  - Lenient progression: when a step's preconditions fail, we record the
    violation but still apply the step's effects and resource deltas, then
    continue. This lets each label dimension be assessed quasi-independently
    (a single early slip doesn't mask a goal omission later in the plan).
    Ground truth for "would this plan actually work" is overall_valid, which
    requires all three checks to pass, so leniency never upgrades a bad plan.
  - Unparseable / unknown-action steps make the plan inconsistent (an
    executor could not run them) and are otherwise no-ops in simulation.
  - Resource feasibility is checked at every prefix: after each step, every
    dimension must lie in [floor, cap]. Violations record (resource, step
    index, value).
  - Closed-world: only atoms in init (as transformed by effects) are true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from pydantic import BaseModel, Field

from verifier.generation.parser import ParseResult
from verifier.schema import Problem

_EPS = 1e-9


class PlanLabels(BaseModel):
    is_consistent: bool
    consistency_violations: List[str] = Field(default_factory=list)
    is_goal_complete: bool
    unmet_goals: List[str] = Field(default_factory=list)
    is_resource_feasible: bool
    resource_violations: List[str] = Field(default_factory=list)
    overall_valid: bool


@dataclass
class _SimState:
    atoms: Set[Tuple[str, ...]] = field(default_factory=set)
    resources: Dict[str, float] = field(default_factory=dict)


def _format_atom(pred: str, args: Tuple[str, ...]) -> str:
    return f"{pred}({', '.join(args)})" if args else pred


def label_plan(problem: Problem, parse: ParseResult) -> PlanLabels:
    domain = problem.domain
    obj_types = {o.name: o.type for o in problem.objects}
    schemas = {a.name: a for a in domain.action_schemas}
    dims = {d.name: d for d in domain.resource_dimensions}

    state = _SimState(
        atoms={(a.predicate, *a.args) for a in problem.init},
        resources={d.name: d.initial for d in domain.resource_dimensions},
    )

    consistency_violations: List[str] = []
    resource_violations: List[str] = []

    for idx, step in enumerate(parse.steps, start=1):
        if step.action is None:
            consistency_violations.append(f"step {idx}: {step.error}")
            continue

        ga = step.action
        schema = schemas.get(ga.schema_name)
        if schema is None:
            # parser normally catches this, but the labeler must not assume a
            # well-behaved parser — defend anyway.
            consistency_violations.append(f"step {idx}: unknown action '{ga.schema_name}'")
            continue

        if len(ga.args) != len(schema.parameters):
            consistency_violations.append(
                f"step {idx}: {ga.schema_name} arity mismatch "
                f"({len(ga.args)} args, expected {len(schema.parameters)})"
            )
            continue

        binding: Dict[str, str] = {}
        type_ok = True
        for param, arg in zip(schema.parameters, ga.args):
            binding[param.name] = arg
            actual_type = obj_types.get(arg)
            if actual_type is None:
                consistency_violations.append(f"step {idx}: unknown object '{arg}' in {ga}")
                type_ok = False
            elif actual_type != param.type:
                consistency_violations.append(
                    f"step {idx}: argument '{arg}' of {ga.schema_name} has type "
                    f"'{actual_type}', expected '{param.type}'"
                )
                type_ok = False
        if not type_ok:
            # cannot meaningfully simulate a mistyped action; skip its effects
            continue

        for lit in schema.preconditions:
            ground = (lit.atom.predicate, *(binding.get(a, a) for a in lit.atom.args))
            holds = ground in state.atoms
            if holds == lit.negated:  # positive-and-absent, or negative-and-present
                want = "false" if lit.negated else "true"
                consistency_violations.append(
                    f"step {idx}: precondition {_format_atom(ground[0], ground[1:])} "
                    f"of {ga} must be {want} but is not"
                )

        # Lenient progression: apply effects regardless (see module docstring).
        for atom in schema.del_effects:
            state.atoms.discard((atom.predicate, *(binding.get(a, a) for a in atom.args)))
        for atom in schema.add_effects:
            state.atoms.add((atom.predicate, *(binding.get(a, a) for a in atom.args)))

        for res_name, delta in schema.resource_deltas.items():
            dim = dims.get(res_name)
            if dim is None:
                consistency_violations.append(
                    f"step {idx}: action references undeclared resource '{res_name}'"
                )
                continue
            state.resources[res_name] = state.resources.get(res_name, 0.0) + delta
            value = state.resources[res_name]
            if value < dim.floor - _EPS:
                resource_violations.append(
                    f"step {idx}: resource '{res_name}' dropped to {value:g} "
                    f"(floor {dim.floor:g})"
                )
            elif dim.cap is not None and value > dim.cap + _EPS:
                resource_violations.append(
                    f"step {idx}: resource '{res_name}' rose to {value:g} (cap {dim.cap:g})"
                )

    unmet_goals: List[str] = []
    for lit in problem.goal:
        ground = (lit.atom.predicate, *lit.atom.args)
        holds = ground in state.atoms
        if holds == lit.negated:
            want = "false" if lit.negated else "true"
            unmet_goals.append(f"{_format_atom(ground[0], ground[1:])} should be {want}")

    is_consistent = not consistency_violations
    is_goal_complete = not unmet_goals
    is_resource_feasible = not resource_violations
    return PlanLabels(
        is_consistent=is_consistent,
        consistency_violations=consistency_violations,
        is_goal_complete=is_goal_complete,
        unmet_goals=unmet_goals,
        is_resource_feasible=is_resource_feasible,
        resource_violations=resource_violations,
        overall_valid=is_consistent and is_goal_complete and is_resource_feasible,
    )
