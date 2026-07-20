"""Baseline plumbing tests: verdict parsing from hand-written (including
malformed) LLM responses, learned-only feature exclusion, symbolic-only
wiring. No live-LLM behavior is tested (non-deterministic)."""

from verifier.baselines import parse_judge_verdict, symbolic_only
from verifier.baselines.learned_only import LEARNED_ONLY_FEATURES, _SYMBOLIC_FEATURES


def test_judge_parses_valid():
    assert parse_judge_verdict("VERDICT: VALID") is True
    assert parse_judge_verdict("blah blah\nVERDICT: INVALID") is False


def test_judge_parses_case_and_spacing_variants():
    assert parse_judge_verdict("verdict:   valid") is True
    assert parse_judge_verdict("Verdict : INVALID") is False


def test_judge_uses_last_verdict_when_cot_mentions_both():
    text = "If step 3 failed the answer would be VERDICT: INVALID, but it works.\nVERDICT: VALID"
    assert parse_judge_verdict(text) is True


def test_judge_malformed_returns_none():
    assert parse_judge_verdict("The plan looks great to me!") is None
    assert parse_judge_verdict("") is None


def test_judge_none_means_fail_open():
    from verifier.baselines.llm_judge import JudgeResult

    r = JudgeResult(system="llm_judge_zeroshot", predicted_valid=None, raw_response="???")
    assert r.reject is False  # unparseable judge output lets the plan through


def test_symbolic_only_maps_verdict_directly():
    assert symbolic_only(True).predicted_valid is True
    assert symbolic_only(False).predicted_valid is False
    assert symbolic_only(False).reject is True


def test_learned_only_excludes_every_symbolic_feature():
    assert not _SYMBOLIC_FEATURES & set(LEARNED_ONLY_FEATURES)
    assert len(LEARNED_ONLY_FEATURES) > 5  # still has real features to learn from
