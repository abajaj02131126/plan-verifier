"""Fusion rule tests — including the dedicated soundness-invariant test:
hard symbolic failure rejects regardless of trust score."""

import pytest

from verifier.fusion import decide, prf, sweep_threshold


@pytest.mark.parametrize("trust", [0.0, 0.5, 0.999, 1.0])
def test_symbolic_failure_always_rejects_regardless_of_trust(trust):
    d = decide(symbolic_valid=False, trust_score=trust, threshold=0.5)
    assert not d.accept
    assert d.reason == "symbolic_reject"


def test_symbolic_pass_high_trust_accepts():
    d = decide(symbolic_valid=True, trust_score=0.9, threshold=0.5)
    assert d.accept and d.reason == "accept"


def test_symbolic_pass_low_trust_flags():
    d = decide(symbolic_valid=True, trust_score=0.3, threshold=0.5)
    assert not d.accept and d.reason == "low_trust"


def test_threshold_zero_is_symbolic_only():
    # at threshold 0.0 every symbolic pass is accepted -> pure symbolic baseline
    for trust in (0.0, 0.5, 1.0):
        assert decide(True, trust, 0.0).accept
        assert not decide(False, trust, 0.0).accept


def test_prf_math():
    # preds: reject, reject, accept, accept ; truth: flawed, ok, flawed, ok
    m = prf([True, True, False, False], [True, False, True, False])
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_sweep_includes_symbolic_only_row():
    rows = sweep_threshold(
        symbolic_valids=[True, False, True],
        trust_scores=[0.9, 0.9, 0.1],
        flawed_true=[False, True, True],
    )
    row0 = next(r for r in rows if r["threshold"] == 0.0)
    # symbolic-only: rejects only record 2 -> P=1, R=0.5
    assert row0["precision"] == 1.0 and row0["recall"] == 0.5
    row_mid = next(r for r in rows if r["threshold"] == 0.5)
    # trust gating additionally rejects record 3 (low trust, actually flawed)
    assert row_mid["recall"] == 1.0
