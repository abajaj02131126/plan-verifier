"""Domain: a set of action schemas + predicate + resource dimension declarations."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, model_validator

from verifier.schema.actions import ActionSchema
from verifier.schema.predicates import PredicateAtom, PredicateDefinition
from verifier.schema.resources import ResourceDimension


class Domain(BaseModel):
    """A planning domain: types, predicates, action schemas, and resource dimensions."""

    name: str
    types: List[str] = Field(default_factory=list)
    predicates: List[PredicateDefinition] = Field(default_factory=list)
    action_schemas: List[ActionSchema] = Field(default_factory=list)
    resource_dimensions: List[ResourceDimension] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Domain":
        pred_index: Dict[str, PredicateDefinition] = {}
        for p in self.predicates:
            if p.name in pred_index:
                raise ValueError(f"duplicate predicate definition '{p.name}'")
            pred_index[p.name] = p

        resource_names = set()
        for r in self.resource_dimensions:
            if r.name in resource_names:
                raise ValueError(f"duplicate resource dimension '{r.name}'")
            resource_names.add(r.name)

        action_names = set()
        for a in self.action_schemas:
            if a.name in action_names:
                raise ValueError(f"duplicate action schema '{a.name}'")
            action_names.add(a.name)

            param_types = {p.name: p.type for p in a.parameters}
            if len(param_types) != len(a.parameters):
                raise ValueError(f"action '{a.name}' has duplicate parameter names")
            for p in a.parameters:
                if self.types and p.type not in self.types:
                    raise ValueError(
                        f"action '{a.name}' parameter '{p.name}' has undeclared type '{p.type}'"
                    )

            for lit in a.preconditions:
                self._check_atom(a.name, lit.atom, pred_index, param_types)
            for atom in a.add_effects:
                self._check_atom(a.name, atom, pred_index, param_types)
            for atom in a.del_effects:
                self._check_atom(a.name, atom, pred_index, param_types)
            for res_name in a.resource_deltas:
                if res_name not in resource_names:
                    raise ValueError(f"action '{a.name}' references undeclared resource '{res_name}'")

        return self

    @staticmethod
    def _check_atom(
        action_name: str,
        atom: PredicateAtom,
        pred_index: Dict[str, PredicateDefinition],
        param_types: Dict[str, str],
    ) -> None:
        pdef = pred_index.get(atom.predicate)
        if pdef is None:
            raise ValueError(f"action '{action_name}' references undeclared predicate '{atom.predicate}'")
        if len(atom.args) != len(pdef.arg_types):
            raise ValueError(
                f"action '{action_name}' predicate '{atom.predicate}' expects "
                f"{len(pdef.arg_types)} args, got {len(atom.args)}"
            )
        for arg, expected_type in zip(atom.args, pdef.arg_types):
            if arg not in param_types:
                raise ValueError(
                    f"action '{action_name}' predicate '{atom.predicate}' references "
                    f"unknown parameter '{arg}'"
                )
            actual_type = param_types[arg]
            if actual_type != expected_type:
                raise ValueError(
                    f"action '{action_name}' predicate '{atom.predicate}' expects arg '{arg}' "
                    f"to have type '{expected_type}', but parameter '{arg}' has type '{actual_type}'"
                )

    def action_by_name(self, name: str) -> ActionSchema:
        for a in self.action_schemas:
            if a.name == name:
                return a
        raise KeyError(f"no action schema named '{name}' in domain '{self.name}'")

    def resource_by_name(self, name: str) -> ResourceDimension:
        for r in self.resource_dimensions:
            if r.name == name:
                return r
        raise KeyError(f"no resource dimension named '{name}' in domain '{self.name}'")