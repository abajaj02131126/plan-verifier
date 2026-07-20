"""Deterministic symbolic verifier [SOUNDNESS-CRITICAL] — the hard gate
(spec sections 3, 4.1).

Operates on a sequence of extracted structured actions (from the Phase 2
rule-based parser or the Phase 3 LLM extractor) plus a Problem:

  consistency_check        — forward simulation; every step's preconditions
                             must hold in the state where it executes, args
                             must typecheck against the schema.
  goal_completeness_check  — the final simulated state must entail every goal
                             conjunct; unmet conjuncts are reported.
  resource_feasibility_check — running-sum per resource dimension, checked at
                             every prefix against floor/cap.
  verify                   — bundles all three into a Verdict with an overall
                             bool and a human-readable explanation.

Progression semantics INTENTIONALLY match the Phase 2 oracle labeler
(lenient: violations are recorded, effects still applied, simulation
continues) so their overall_valid verdicts are comparable one-to-one. The two
are separate implementations — the labeler is self-contained, this verifier
builds on verifier/schema/state.py's grounding helpers — so cross-checking
them (scripts/verify_plans.py --crosscheck) is real evidence of soundness,
not a tautology.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from verifier.schema import Domain, Problem
from verifier.schema.state import GroundAction

_EPS = 1e-9

GroundAtomT = Tuple[str, ...]


class ResourceViolationDetail(BaseModel):
    resource: str
    step: int  # 1-indexed step at which the violation occurred
    value: float
    limit: float
    kind: str  # "floor" or "cap"

    def render(self) -> str:
        rel = "below floor" if self.kind == "floor" else "above cap"
        return (
            f"step {self.step}: resource '{self.resource}' reached {self.value:g}, "
            f"{rel} {self.limit:g}"
        )


class Verdict(BaseModel):
    """Bundled symbolic verdict for one plan against one problem."""

    is_consistent: bool
    consistency_violations: List[str] = Field(default_factory=list)
    is_goal_complete: bool
    unmet_goals: List[str] = Field(default_factory=list)
    is_resource_feasible: bool
    resource_violations: List[ResourceViolationDetail] = Field(default_factory=list)
    overall_valid: bool
    explanation: str = ""


def _fmt(pred: str, args: Tuple[str, ...]) -> str:
    return f"{pred}({', '.join(args)})" if args else pred


def _simulate(
    actions: List[GroundAction],
    problem: Problem,
) -> Tuple[Set[GroundAtomT], List[str], List[ResourceViolationDetail]]:
    """Lenient forward simulation. Returns (final atoms, consistency
    violations, resource violations)."""
    domain: Domain = problem.domain
    obj_types = {o.name: o.type for o in problem.objects}
    schemas = {a.name: a for a in domain.action_schemas}
    dims = {d.name: d for d in domain.resource_dimensions}

    atoms: Set[GroundAtomT] = {(a.predicate, *a.args) for a in problem.init}
    resources: Dict[str, float] = {d.name: d.initial for d in domain.resource_dimensions}
    consistency: List[str] = []
    resource_viols: List[ResourceViolationDetail] = []

    for idx, ga in enumerate(actions, start=1):
        schema = schemas.get(ga.schema_name)
        if schema is None:
            consistency.append(f"step {idx}: unknown action '{ga.schema_name}'")
            continue
        if len(ga.args) != len(schema.parameters):
            consistency.append(
                f"step {idx}: '{ga.schema_name}' arity mismatch "
                f"({len(ga.args)} args, expected {len(schema.parameters)})"
            )
            continue

        binding: Dict[str, str] = {}
        typed_ok = True
        for param, arg in zip(schema.parameters, ga.args):
            binding[param.name] = arg
            arg_type = obj_types.get(arg)
            if arg_type is None:
                consistency.append(f"step {idx}: unknown object '{arg}' in {ga}")
                typed_ok = False
            elif arg_type != param.type:
                consistency.append(
                    f"step {idx}: '{arg}' has type '{arg_type}' but "
                    f"'{ga.schema_name}' parameter '{param.name}' needs '{param.type}'"
                )
                typed_ok = False
        if not typed_ok:
            continue

        for lit in schema.preconditions:
            ground = (lit.atom.predicate, *(binding.get(a, a) for a in lit.atom.args))
            holds = ground in atoms
            if holds == lit.negated:
                want = "false" if lit.negated else "true"
                consistency.append(
                    f"step {idx}: precondition {_fmt(ground[0], ground[1:])} of "
                    f"{ga} must be {want} in the state where it executes, but is not"
                )

        for atom in schema.del_effects:
            atoms.discard((atom.predicate, *(binding.get(a, a) for a in atom.args)))
        for atom in schema.add_effects:
            atoms.add((atom.predicate, *(binding.get(a, a) for a in atom.args)))

        for res_name, delta in schema.resource_deltas.items():
            dim = dims.get(res_name)
            if dim is None:
                consistency.append(f"step {idx}: undeclared resource '{res_name}'")
                continue
            resources[res_name] = resources.get(res_name, 0.0) + delta
            value = resources[res_name]
            if value < dim.floor - _EPS:
                resource_viols.append(
                    ResourceViolationDetail(
                        resource=res_name, step=idx, value=value, limit=dim.floor, kind="floor"
                    )
                )
            elif dim.cap is not None and value > dim.cap + _EPS:
                resource_viols.append(
                    ResourceViolationDetail(
                        resource=res_name, step=idx, value=value, limit=dim.cap, kind="cap"
                    )
                )

    return atoms, consistency, resource_viols


def consistency_check(actions: List[GroundAction], problem: Problem) -> Tuple[bool, List[str]]:
    _, violations, _ = _simulate(actions, problem)
    return (not violations, violations)


def goal_completeness_check(final_atoms: Set[GroundAtomT], problem: Problem) -> Tuple[bool, List[str]]:
    unmet: List[str] = []
    for lit in problem.goal:
        ground = (lit.atom.predicate, *lit.atom.args)
        holds = ground in final_atoms
        if holds == lit.negated:
            want = "false" if lit.negated else "true"
            unmet.append(f"goal conjunct {_fmt(ground[0], ground[1:])} should be {want} but is not")
    return (not unmet, unmet)


def resource_feasibility_check(
    actions: List[GroundAction], problem: Problem
) -> Tuple[bool, List[ResourceViolationDetail]]:
    _, _, violations = _simulate(actions, problem)
    return (not violations, violations)


def verify(
    actions: List[GroundAction],
    problem: Problem,
    input_errors: Optional[List[str]] = None,
) -> Verdict:
    """Full symbolic verdict. ``input_errors`` carries upstream parse or
    extraction failures (steps that never became structured actions); an
    executor could not run those steps, so they count against consistency."""
    final_atoms, consistency, resource_viols = _simulate(actions, problem)
    if input_errors:
        consistency = [f"unstructured step: {e}" for e in input_errors] + consistency
    goal_ok, unmet = goal_completeness_check(final_atoms, problem)

    is_consistent = not consistency
    is_resource_feasible = not resource_viols
    overall = is_consistent and goal_ok and is_resource_feasible

    if overall:
        explanation = (
            f"VALID: all {len(actions)} steps executable in sequence, every goal "
            f"conjunct satisfied, all resource dimensions within limits."
        )
    else:
        parts = []
        if consistency:
            parts.append(f"{len(consistency)} consistency violation(s): " + "; ".join(consistency[:3]))
        if unmet:
            parts.append(f"{len(unmet)} unmet goal conjunct(s): " + "; ".join(unmet[:3]))
        if resource_viols:
            parts.append(
                f"{len(resource_viols)} resource violation(s): "
                + "; ".join(v.render() for v in resource_viols[:3])
            )
        explanation = "INVALID: " + " | ".join(parts)

    return Verdict(
        is_consistent=is_consistent,
        consistency_violations=consistency,
        is_goal_complete=goal_ok,
        unmet_goals=unmet,
        is_resource_feasible=is_resource_feasible,
        resource_violations=resource_viols,
        overall_valid=overall,
        explanation=explanation,
    )
