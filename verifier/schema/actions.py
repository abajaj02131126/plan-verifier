"""Action schemas: typed parameters, preconditions, effects, resource deltas."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from verifier.schema.predicates import Literal, PredicateAtom


class Parameter(BaseModel):
    name: str
    type: str


class ActionSchema(BaseModel):
    """A lifted action: parameters plus preconditions/effects over those parameters."""

    name: str
    parameters: List[Parameter] = Field(default_factory=list)
    preconditions: List[Literal] = Field(default_factory=list)
    add_effects: List[PredicateAtom] = Field(default_factory=list)
    del_effects: List[PredicateAtom] = Field(default_factory=list)
    resource_deltas: Dict[str, float] = Field(default_factory=dict)

    def param_names(self) -> List[str]:
        return [p.name for p in self.parameters]