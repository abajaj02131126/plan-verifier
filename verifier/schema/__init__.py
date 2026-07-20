"""Typed DSL for PDDL-style domains/problems with resource dimensions."""

from verifier.schema.actions import ActionSchema, Parameter
from verifier.schema.domain import Domain
from verifier.schema.predicates import Literal, PredicateAtom, PredicateDefinition
from verifier.schema.problem import Problem, TypedObject
from verifier.schema.resources import ResourceDimension
from verifier.schema.state import (
    GroundAction,
    GroundAtom,
    PreconditionViolation,
    ResourceViolation,
    State,
    applicable,
    apply_action,
    atom_to_ground,
    goal_satisfied,
    initial_state,
    literal_holds,
    simulate,
)

__all__ = [
    "ActionSchema",
    "Parameter",
    "Domain",
    "Literal",
    "PredicateAtom",
    "PredicateDefinition",
    "Problem",
    "TypedObject",
    "ResourceDimension",
    "GroundAction",
    "GroundAtom",
    "PreconditionViolation",
    "ResourceViolation",
    "State",
    "applicable",
    "apply_action",
    "atom_to_ground",
    "goal_satisfied",
    "initial_state",
    "literal_holds",
    "simulate",
]