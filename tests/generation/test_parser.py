"""Rule-based plan parser: format tolerance and validation against the domain."""

from verifier.domains.blocksworld import build_domain
from verifier.generation.parser import parse_plan

DOMAIN = build_domain()


def test_parses_canonical_format():
    text = "Step 1: pick-up(b1)\nStep 2: stack(b1, b2)"
    result = parse_plan(text, DOMAIN)
    assert result.fully_parsed
    assert [a.schema_name for a in result.actions] == ["pick-up", "stack"]
    assert result.actions[1].args == ("b1", "b2")


def test_parses_tolerant_variants():
    text = """\
1. pick-up(b1)
2) stack(b1, b2)
unstack(b2, b3)
STEP 4: Put-Down(b2).
"""
    result = parse_plan(text, DOMAIN)
    assert result.fully_parsed
    assert [a.schema_name for a in result.actions] == ["pick-up", "stack", "unstack", "put-down"]


def test_skips_prose_but_flags_broken_steps():
    text = """\
Here is my plan:
Step 1: pick-up(b1)
Step 2: this is not an action
Step 3: stack(b1, b2)
"""
    result = parse_plan(text, DOMAIN)
    assert len(result.steps) == 3  # prose line skipped, broken step kept
    assert not result.fully_parsed
    assert "unparseable" in result.errors[0]
    assert len(result.actions) == 2


def test_unknown_action_is_error_not_crash():
    result = parse_plan("Step 1: teleport(b1, b2)", DOMAIN)
    assert not result.fully_parsed
    assert "unknown action 'teleport'" in result.errors[0]


def test_arity_mismatch_is_error():
    result = parse_plan("Step 1: stack(b1)", DOMAIN)
    assert not result.fully_parsed
    assert "expects 2 args" in result.errors[0]


def test_empty_and_no_step_text():
    assert parse_plan("", DOMAIN).steps == []
    assert parse_plan("I cannot solve this problem.", DOMAIN).steps == []
