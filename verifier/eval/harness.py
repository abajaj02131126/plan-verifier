"""Unified evaluation harness (Phase 8): metrics for every system over every
domain, flaw-type breakdowns, calibration data, and the downstream
execute-or-reject experiment.

Convention throughout: the positive class for P/R/F1 is "plan is flawed"
(ground truth: NOT labels.overall_valid); predicting reject = predicting
flawed. Systems are compared on the problem-level TEST split only (the trust
and learned-only models are trained on train/val), so no result row ever
includes a record whose problem was seen in training.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from verifier.fusion import decide, prf
from verifier.learned import expected_calibration_error, make_label

FLAW_TYPES = ["inconsistency", "goal_incompleteness", "resource_infeasibility"]


def flaw_types_of(labels: dict) -> List[str]:
    flaws = []
    if not labels["is_consistent"]:
        flaws.append("inconsistency")
    if not labels["is_goal_complete"]:
        flaws.append("goal_incompleteness")
    if not labels["is_resource_feasible"]:
        flaws.append("resource_infeasibility")
    return flaws


def evaluate_system(
    records: Sequence[dict],
    reject_fn: Callable[[dict], Optional[bool]],
    system: str,
) -> Dict:
    """Overall + per-flaw-type P/R/F1 for one system.

    reject_fn(record) -> True (reject/flag), False (accept), or None
    (system produced no verdict; counted as accept, i.e. fail-open).
    Per-flaw-type recall is computed over records exhibiting that flaw type
    (can a system catch THIS kind of flaw); precision is only meaningful
    overall, so per-type rows report recall and support.
    """
    rejects = [bool(reject_fn(r)) for r in records]
    flawed = [not r["labels"]["overall_valid"] for r in records]
    overall = prf(rejects, flawed)

    per_type = {}
    for ft in FLAW_TYPES:
        idx = [i for i, r in enumerate(records) if ft in flaw_types_of(r["labels"])]
        caught = sum(1 for i in idx if rejects[i])
        per_type[ft] = {
            "support": len(idx),
            "recall": caught / len(idx) if idx else None,
        }

    return {"system": system, "n": len(records), "overall": overall, "per_flaw_type": per_type}


def hybrid_reject_fn(trust_model, threshold: float) -> Callable[[dict], bool]:
    def fn(record: dict) -> bool:
        trust = float(trust_model.predict_proba([record])[0])
        return not decide(record["verdict"]["overall_valid"], trust, threshold).accept

    return fn


def pick_threshold(val_records: Sequence[dict], trust_model, thresholds=None) -> float:
    """Operating point: threshold maximizing flawed-detection F1 on the
    validation split (ties -> lowest threshold, i.e. least intervention)."""
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(21)]
    flawed = [not r["labels"]["overall_valid"] for r in val_records]
    probs = trust_model.predict_proba(list(val_records))
    best_th, best_f1 = 0.0, -1.0
    for th in thresholds:
        rejects = [
            not decide(r["verdict"]["overall_valid"], float(p), th).accept
            for r, p in zip(val_records, probs)
        ]
        f1 = prf(rejects, flawed)["f1"]
        if f1 > best_f1 + 1e-12:
            best_th, best_f1 = th, f1
    return best_th


def trust_calibration(test_records: Sequence[dict], trust_model) -> Dict:
    probs = trust_model.predict_proba(list(test_records))
    labels = np.array([make_label(r) for r in test_records])
    ece, bins = expected_calibration_error(np.asarray(probs), labels)
    return {"ece": ece, "bins": bins, "n": len(test_records)}


def clean_but_flawed_analysis(
    records: Sequence[dict], trust_model, threshold: float
) -> Dict:
    """The paper's load-bearing number (spec section 10): among plans the
    symbolic layer PASSES (on the production extraction path) but that are
    actually flawed per the oracle, how many does the trust gate flag?"""
    passed = [r for r in records if r["verdict"]["overall_valid"]]
    clean_but_flawed = [r for r in passed if not r["labels"]["overall_valid"]]
    if not clean_but_flawed:
        return {"n_symbolic_pass": len(passed), "n_clean_but_flawed": 0, "caught_by_trust": 0}
    probs = trust_model.predict_proba(clean_but_flawed)
    caught = int(sum(p < threshold for p in probs))
    return {
        "n_symbolic_pass": len(passed),
        "n_clean_but_flawed": len(clean_but_flawed),
        "caught_by_trust": caught,
        "catch_rate": caught / len(clean_but_flawed),
    }


def measure_latency(records: Sequence[dict], fn: Callable[[dict], object], n: int = 20) -> float:
    """Mean seconds/record for a local (non-API) decision function."""
    sample = list(records)[:n]
    t0 = time.perf_counter()
    for r in sample:
        fn(r)
    return (time.perf_counter() - t0) / max(len(sample), 1)
