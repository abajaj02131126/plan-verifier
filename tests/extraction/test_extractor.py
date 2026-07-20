"""Extractor tests. Pure-logic tests always run; tests that hit the Anthropic
API are skipped when no key is available (CI-offline safe) and use the same
fixture plans as the Phase 2 parser tests for the equivalence check.
"""

import pytest

from verifier.domains.blocksworld import build_domain
from verifier.extraction.extractor import _validate
from verifier.extraction.plan_extractor import split_plan_lines
from verifier.generation.parser import parse_plan

DOMAIN = build_domain()


def _have_key() -> bool:
    try:
        from verifier.llm import get_client

        get_client()
        return True
    except RuntimeError:
        return False


needs_api = pytest.mark.skipif(not _have_key(), reason="no ANTHROPIC_API_KEY available")


# ---------- pure logic (no API) ----------


def test_validate_accepts_known_action():
    assert _validate(DOMAIN, "stack", ["b1", "b2"]) is None


def test_validate_rejects_unknown_action():
    err = _validate(DOMAIN, "teleport", ["b1"])
    assert err is not None and "unknown action" in err


def test_validate_rejects_bad_arity():
    err = _validate(DOMAIN, "stack", ["b1"])
    assert err is not None and "expects 2 args" in err


def test_split_plan_lines():
    raw = """\
Here's the plan:

Step 1: pick-up(b1)
2) stack(b1, b2)
some commentary without a call
put-down(b3)
"""
    lines = split_plan_lines(raw)
    assert lines == ["Step 1: pick-up(b1)", "2) stack(b1, b2)", "put-down(b3)"]


# ---------- live API ----------


@needs_api
def test_extractor_matches_rule_based_parser_on_clean_input():
    from verifier.extraction import extract_step
    from verifier.llm import get_client

    client = get_client()
    fixture = "Step 1: pick-up(b1)\nStep 2: stack(b1, b3)"
    parsed = parse_plan(fixture, DOMAIN)

    for step, expected in zip(fixture.splitlines(), parsed.actions):
        e = extract_step(client, DOMAIN, step, temperature=0.0)
        assert e.valid
        assert e.action_type == expected.schema_name
        assert tuple(e.args) == expected.args


@needs_api
def test_extractor_degrades_gracefully_on_garbage():
    from verifier.extraction import extract_step
    from verifier.llm import get_client

    client = get_client()
    e = extract_step(client, DOMAIN, "Step 9: recalibrate the flux capacitor please")
    # must not crash; must signal low trust one way or the other
    assert e.extractor_confidence <= 0.5 or not e.valid


@needs_api
def test_self_consistency_agreement_on_unambiguous_step():
    from verifier.extraction import extract_step_self_consistent
    from verifier.llm import get_client

    client = get_client()
    sc = extract_step_self_consistent(client, DOMAIN, "Step 1: unstack(b3, b2)", k=3)
    assert sc.k == 3
    assert sc.extraction.action_type == "unstack"
    assert sc.agreement_exact >= 2 / 3  # allow one dissenting resample
