"""Fusion decision rule (spec section 4.3).

  1. Hard symbolic failure  -> ALWAYS reject. This is the soundness claim of
     the whole system: no trust score, however high, can overrule a provably
     broken plan.
  2. Symbolic pass + trust >= threshold -> accept.
  3. Symbolic pass + trust <  threshold -> flag/reject (low-confidence parse).

The threshold is configurable; sweep_threshold() reports the precision/
recall/F1 curve for picking an operating point (and the paper's PR figure).
Convention: the positive class for P/R/F1 is "plan is flawed" (the verifier's
job is catching flaws), so a rejection is a positive prediction.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from pydantic import BaseModel


class FusionDecision(BaseModel):
    accept: bool
    reason: str  # "symbolic_reject" | "low_trust" | "accept"
    trust_score: float
    symbolic_valid: bool


def decide(symbolic_valid: bool, trust_score: float, threshold: float) -> FusionDecision:
    if not symbolic_valid:
        return FusionDecision(
            accept=False, reason="symbolic_reject", trust_score=trust_score, symbolic_valid=False
        )
    if trust_score < threshold:
        return FusionDecision(
            accept=False, reason="low_trust", trust_score=trust_score, symbolic_valid=True
        )
    return FusionDecision(accept=True, reason="accept", trust_score=trust_score, symbolic_valid=True)


def prf(reject_pred: Sequence[bool], flawed_true: Sequence[bool]) -> Dict[str, float]:
    """Precision/recall/F1 for flaw detection: predicting reject == predicting
    'this plan is flawed'; ground truth flawed comes from the oracle labels."""
    tp = sum(1 for p, t in zip(reject_pred, flawed_true) if p and t)
    fp = sum(1 for p, t in zip(reject_pred, flawed_true) if p and not t)
    fn = sum(1 for p, t in zip(reject_pred, flawed_true) if not p and t)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def sweep_threshold(
    symbolic_valids: Sequence[bool],
    trust_scores: Sequence[float],
    flawed_true: Sequence[bool],
    thresholds: Sequence[float] | None = None,
) -> List[Dict]:
    """P/R/F1 of the fused decision at each threshold. threshold=0.0 is
    exactly the symbolic-only baseline (trust score ignored)."""
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(21)]
    rows = []
    for th in thresholds:
        rejects = [
            not decide(sv, ts, th).accept for sv, ts in zip(symbolic_valids, trust_scores)
        ]
        rows.append({"threshold": th, **prf(rejects, flawed_true)})
    return rows
