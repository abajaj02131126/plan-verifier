"""Validation tests for the typed DSL in verifier/schema/."""

import pytest
from pydantic import ValidationError

from verifier.schema import (
    ActionSchema,
    Domain,
    GroundAction,
    Literal,
    Parameter,
    PredicateAtom,
    PredicateDefinition,
    Problem,
    ResourceDimension,
    TypedObject,
    apply_action,
    goal_satisfied,
    initial_state,
)


def _mini_domain(**resource_kwargs) -> Domain:
    return Domain(
        name="mini",
        types=["block"],
        predicates=[PredicateDefinition(name="clear", arg_types=["block"])],
        action_schemas=[
            ActionSchema(
                name="tap",
                parameters=[Parameter(name="x", type="block")],
                preconditions=[Literal.pos("clear", "x")],
                add_effects=[],
                del_effects=[],
                resource_deltas={"energy": -1.0},
            )
        ],
        resource_dimensions=[ResourceDimension(name="energy", initial=5, cap=5, **resource_kwargs)],
    )


def test_valid_domain_builds():
    domain = _mini_domain()
    assert domain.action_by_name("tap").name == "tap"
    assert domain.resource_by_name("energy").initial == 5


def test_domain_rejects_undeclared_predicate():
    with pytest.raises(ValidationError):
        Domain(
            name="bad",
            types=["block"],
            predicates=[],
            action_schemas=[
                ActionSchema(
                    name="tap",
                    parameters=[Parameter(name="x", type="block")],
                    preconditions=[Literal.pos("clear", "x")],
                )
            ],
            resource_dimensions=[],
        )


def test_domain_rejects_undeclared_resource():
    with pytest.raises(ValidationError):
        Domain(
            name="bad",
            types=["block"],
            predicates=[PredicateDefinition(name="clear", arg_types=["block"])],
            action_schemas=[
                ActionSchema(
                    name="tap",
                    parameters=[Parameter(name="x", type="block")],
                    preconditions=[Literal.pos("clear", "x")],
                    resource_deltas={"ghost": -1.0},
                )
            ],
            resource_dimensions=[],
        )


def test_domain_rejects_predicate_arity_mismatch():
    with pytest.raises(ValidationError):
        Domain(
            name="bad",
            types=["block"],
            predicates=[PredicateDefinition(name="on", arg_types=["block", "block"])],
            action_schemas=[
                ActionSchema(
                    name="tap",
                    parameters=[Parameter(name="x", type="block")],
                    preconditions=[Literal.pos("on", "x")],
                )
            ],
            resource_dimensions=[],
        )


def test_domain_rejects_parameter_type_mismatch():
    with pytest.raises(ValidationError):
        Domain(
            name="bad",
            types=["block", "location"],
            predicates=[PredicateDefinition(name="clear", arg_types=["block"])],
            action_schemas=[
                ActionSchema(
                    name="tap",
                    parameters=[Parameter(name="x", type="location")],
                    preconditions=[Literal.pos("clear", "x")],
                )
            ],
            resource_dimensions=[],
        )


def test_resource_dimension_rejects_initial_above_cap():
    with pytest.raises(ValidationError):
        ResourceDimension(name="energy", initial=10, cap=5)


def test_problem_rejects_object_with_undeclared_type():
    domain = _mini_domain()
    with pytest.raises(ValidationError):
        Problem(
            name="p",
            domain=domain,
            objects=[TypedObject(name="b1", type="widget")],
            init=[],
            goal=[],
        )


def test_problem_rejects_goal_with_unknown_object():
    domain = _mini_domain()
    with pytest.raises(ValidationError):
        Problem(
            name="p",
            domain=domain,
            objects=[TypedObject(name="b1", type="block")],
            init=[],
            goal=[Literal.pos("clear", "b2")],
        )


def test_problem_valid_round_trips_and_simulates():
    domain = _mini_domain()
    problem = Problem(
        name="p",
        domain=domain,
        objects=[TypedObject(name="b1", type="block")],
        init=[PredicateAtom(predicate="clear", args=["b1"])],
        goal=[Literal.pos("clear", "b1")],
    )
    state = initial_state(problem)
    assert goal_satisfied(problem, state)

    next_state = apply_action(domain, state, GroundAction(schema_name="tap", args=("b1",)))
    assert next_state.resource_dict()["energy"] == 4.0

    # model_dump/model_validate round-trip (as used by the JSONL CLI)
    restored = Problem.model_validate(problem.model_dump())
    assert restored == problem