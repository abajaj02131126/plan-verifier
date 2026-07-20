"""Problem: a Domain instance + typed objects + initial state + goal condition."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, model_validator

from verifier.schema.domain import Domain
from verifier.schema.predicates import Literal, PredicateAtom, PredicateDefinition


class TypedObject(BaseModel):
    name: str
    type: str


class Problem(BaseModel):
    """A concrete planning problem: objects, initial state, and a goal condition.

    ``init`` is the set of ground atoms true at the start (closed-world
    assumption: anything not listed is false). ``goal`` is a conjunction of
    (possibly negated) ground literals.
    """

    name: str
    domain: Domain
    objects: List[TypedObject] = Field(default_factory=list)
    init: List[PredicateAtom] = Field(default_factory=list)
    goal: List[Literal] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "Problem":
        obj_types: Dict[str, str] = {}
        for o in self.objects:
            if o.name in obj_types:
                raise ValueError(f"duplicate object '{o.name}'")
            obj_types[o.name] = o.type
            if self.domain.types and o.type not in self.domain.types:
                raise ValueError(f"object '{o.name}' has undeclared type '{o.type}'")

        pred_index = {p.name: p for p in self.domain.predicates}
        for atom in self.init:
            self._check_ground_atom(atom, pred_index, obj_types, "init")
        for lit in self.goal:
            self._check_ground_atom(lit.atom, pred_index, obj_types, "goal")
        return self

    @staticmethod
    def _check_ground_atom(
        atom: PredicateAtom,
        pred_index: Dict[str, PredicateDefinition],
        obj_types: Dict[str, str],
        where: str,
    ) -> None:
        pdef = pred_index.get(atom.predicate)
        if pdef is None:
            raise ValueError(f"{where}: undeclared predicate '{atom.predicate}'")
        if len(atom.args) != len(pdef.arg_types):
            raise ValueError(f"{where}: predicate '{atom.predicate}' arity mismatch")
        for arg, expected_type in zip(atom.args, pdef.arg_types):
            if arg not in obj_types:
                raise ValueError(f"{where}: unknown object '{arg}'")
            if obj_types[arg] != expected_type:
                raise ValueError(
                    f"{where}: object '{arg}' has type '{obj_types[arg]}', expected '{expected_type}'"
                )

    def objects_by_type(self, type_name: str) -> List[str]:
        return [o.name for o in self.objects if o.type == type_name]