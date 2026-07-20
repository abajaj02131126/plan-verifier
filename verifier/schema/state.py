"""Grounding, progression (forward simulation), and resource-feasibility checks.

Not a pydantic model — this is the executable semantics of the DSL defined in
predicates.py/actions.py/domain.py/problem.py. ``State`` is a plain frozen
dataclass (hashable, used as a search-node key by the gold planner).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

from verifier.schema.domain import Domain
from verifier.schema.predicates import Literal, PredicateAtom
from verifier.schema.problem import Problem

GroundAtom = Tuple[str, ...]  # (predicate, arg1, arg2, ...)


class PreconditionViolation(Exception):
    """Raised when apply_action is called on a state that doesn't satisfy preconditions."""


class ResourceViolation(Exception):
    """Raised when applying an action would push a resource below its floor or above its cap."""


@dataclass(frozen=True)
class GroundAction:
    """An instantiated action: a schema name plus object-name arguments."""

    schema_name: str
    args: Tuple[str, ...]

    def __repr__(self) -> str:
        return f"{self.schema_name}({', '.join(self.args)})"


@dataclass(frozen=True)
class State:
    """A search state: the set of true ground atoms plus current resource levels."""

    atoms: FrozenSet[GroundAtom]
    resources: Tuple[Tuple[str, float], ...]  # sorted (name, value) pairs, hashable

    def resource_dict(self) -> Dict[str, float]:
        return dict(self.resources)


def atom_to_ground(atom: PredicateAtom, binding: Dict[str, str]) -> GroundAtom:
    """Substitute parameter names in ``atom`` with the objects in ``binding``."""
    return (atom.predicate, *(binding.get(a, a) for a in atom.args))


def initial_state(problem: Problem) -> State:
    atoms = frozenset((a.predicate, *a.args) for a in problem.init)
    resources = tuple(sorted((r.name, r.initial) for r in problem.domain.resource_dimensions))
    return State(atoms=atoms, resources=resources)


def literal_holds(lit: Literal, binding: Dict[str, str], atoms: FrozenSet[GroundAtom]) -> bool:
    ground = atom_to_ground(lit.atom, binding)
    return (ground in atoms) != lit.negated


def applicable(domain: Domain, state: State, ground: GroundAction) -> bool:
    """True iff ``ground`` can be legally applied in ``state`` (preconditions hold and
    every resource delta stays within [floor, cap])."""
    schema = domain.action_by_name(ground.schema_name)
    binding = dict(zip(schema.param_names(), ground.args))
    if not all(literal_holds(lit, binding, state.atoms) for lit in schema.preconditions):
        return False

    resources = state.resource_dict()
    for res_name, delta in schema.resource_deltas.items():
        dim = domain.resource_by_name(res_name)
        new_val = resources[res_name] + delta
        if new_val < dim.floor - 1e-9:
            return False
        if dim.cap is not None and new_val > dim.cap + 1e-9:
            return False
    return True


def apply_action(domain: Domain, state: State, ground: GroundAction) -> State:
    """Apply ``ground`` to ``state``, raising if preconditions or resource bounds are violated."""
    schema = domain.action_by_name(ground.schema_name)
    binding = dict(zip(schema.param_names(), ground.args))

    for lit in schema.preconditions:
        if not literal_holds(lit, binding, state.atoms):
            raise PreconditionViolation(f"{ground}: precondition {lit} not satisfied")

    resources = state.resource_dict()
    for res_name, delta in schema.resource_deltas.items():
        dim = domain.resource_by_name(res_name)
        new_val = resources[res_name] + delta
        if new_val < dim.floor - 1e-9 or (dim.cap is not None and new_val > dim.cap + 1e-9):
            raise ResourceViolation(
                f"{ground}: resource '{res_name}' would go to {new_val} "
                f"(floor={dim.floor}, cap={dim.cap})"
            )
        resources[res_name] = new_val

    new_atoms = set(state.atoms)
    for atom in schema.del_effects:
        new_atoms.discard(atom_to_ground(atom, binding))
    for atom in schema.add_effects:
        new_atoms.add(atom_to_ground(atom, binding))

    return State(atoms=frozenset(new_atoms), resources=tuple(sorted(resources.items())))


def goal_satisfied(problem: Problem, state: State) -> bool:
    return all(literal_holds(lit, {}, state.atoms) for lit in problem.goal)


def simulate(domain: Domain, state: State, plan: List[GroundAction]) -> State:
    """Apply a full plan in sequence, raising on the first invalid step."""
    for ga in plan:
        state = apply_action(domain, state, ga)
    return state