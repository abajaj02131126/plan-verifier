"""Predicate declarations and literals shared by action schemas and problems."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class PredicateDefinition(BaseModel):
    """Declares a predicate's name and the types of its positional arguments."""

    name: str
    arg_types: List[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("predicate name must be non-empty")
        return v


class PredicateAtom(BaseModel):
    """A predicate applied to arguments.

    Arguments are parameter names when the atom appears inside an
    ``ActionSchema`` (lifted), or object names when it appears in a
    ``Problem``'s init/goal (grounded). Both are plain strings; which
    one applies depends on context.
    """

    predicate: str
    args: List[str] = Field(default_factory=list)

    def arity(self) -> int:
        return len(self.args)


class Literal(BaseModel):
    """A predicate atom, optionally negated."""

    atom: PredicateAtom
    negated: bool = False

    @classmethod
    def pos(cls, predicate: str, *args: str) -> "Literal":
        return cls(atom=PredicateAtom(predicate=predicate, args=list(args)))

    @classmethod
    def neg(cls, predicate: str, *args: str) -> "Literal":
        return cls(atom=PredicateAtom(predicate=predicate, args=list(args)), negated=True)