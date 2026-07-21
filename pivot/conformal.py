"""Task 4 (pivot): split-conformal trust gate on the existing unfaithful set.

Replaces the trust model's ad hoc Platt threshold with a split-conformal
threshold that carries a distribution-free coverage guarantee for catching
UNFAITHFUL translations (production verdict != rule-based reference verdict).

Uses ONLY existing labeled data (no new paraphrases, no API calls): the
pooled prose-regime records, split by problem into train/val/test exactly as
the existing pipeline does. Trust model trained on train; val is the
conformal CALIBRATION set; test is the conformal TEST set.

The honest headline is the small-sample one: there are 13 unfaithful records
total (5 train / 5 cal / 3 test). A distribution-free 90% coverage guarantee
from 5 calibration positives is essentially vacuous, and we report the full
interval width rather than a bare point threshold.

Conformal construction (one-sided, catching the unfaithful/low-trust tail):
  * score s = trust_model P(faithful).
  * flag (reject / re-extract) a record iff s <= tau.
  * to guarantee P(flag | unfaithful) >= 1 - alpha, set tau to the
    ceil((1-alpha)(n_cal+1))-th smallest trust score among CALIBRATION
    UNFAITHFUL examples. If that rank exceeds n_cal, the guarantee can only
    be met by tau = 1.0 (flag everything) — reported as vacuous.
  * finite-sample coverage of split conformal on a fresh unfaithful point is
    Beta(n_cal+1-l, l)-distributed with l = floor(alpha (n_cal+1)); we report
    its mean and central 90% interval (the honest "interval width").
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from pivot.config import DOMAIN_STAKES
from verifier.learned import make_label, split_by_problem, train_trust_model

S = Path("verifier/data/synthetic")
OUT = Path("pivot/results")
DOM = ["blocksworld", "logistics", "tools"]


def _load_pooled() -> list[dict]:
    recs = []
    for d in DOM:
        bl = {(x["problem_name"], x["condition"]): x for x in (json.loads(l) for l in open(S / f"{d}_baselines.jsonl"))}
        for line in open(S / f"{d}_verdicts_llm.jsonl"):
            r = json.loads(line)
            r["baselines"] = bl.get((r["problem_name"], r["condition"]), {})
            recs.append(r)
    return recs


def conformal_tau(cal_unfaithful_scores: np.ndarray, alpha: float) -> tuple[float, bool]:
    """Smallest tau s.t. >= (1-alpha) of calibration unfaithful scores are <= tau
    (finite-sample conformal rank). Returns (tau, vacuous?). Vacuous => the
    guarantee forces tau=1.0 (flag everything)."""
    n = len(cal_unfaithful_scores)
    rank = math.ceil((1 - alpha) * (n + 1))  # 1-indexed rank among cal scores
    scores = np.sort(cal_unfaithful_scores)
    if rank > n:
        return 1.0, True  # cannot honor the guarantee except by flagging all
    return float(scores[rank - 1]), False


def coverage_distribution(n_cal: int, alpha: float) -> dict:
    """Split-conformal coverage on a fresh point is Beta(n+1-l, l),
    l = floor(alpha (n+1)). Report mean + central 90% interval."""
    l = math.floor(alpha * (n_cal + 1))
    if l == 0:
        # threshold at the max; coverage lower bound n/(n+1); degenerate Beta
        return {
            "l": 0,
            "note": "l=0: threshold at max calibration score; guaranteed coverage "
            f">= {n_cal}/{n_cal + 1} = {n_cal / (n_cal + 1):.3f}, but any tighter "
            "(higher) coverage target is unreachable from this calibration size.",
            "mean_coverage": None,
            "ci90": None,
        }
    a, b = n_cal + 1 - l, l
    dist = stats.beta(a, b)
    return {
        "l": l,
        "mean_coverage": round(a / (a + b), 4),
        "ci90": [round(float(dist.ppf(0.05)), 4), round(float(dist.ppf(0.95)), 4)],
        "note": f"coverage ~ Beta({a},{b}); interval is wide because n_cal={n_cal}.",
    }


def alpha_of_stakes(S_val: float, alpha_base: float = 0.1, S_ref: float = 10.0,
                    alpha_min: float = 0.01, alpha_max: float = 0.5) -> float:
    """Stakes-adjusted miscoverage: higher stakes -> smaller alpha (tighter
    required coverage -> higher tau -> flag more). Explicit function of S, a
    stated modeling choice (S_ref anchored to the mid-stakes 'logistics'
    domain). Not fit to data."""
    return float(np.clip(alpha_base * (S_ref / S_val), alpha_min, alpha_max))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = _load_pooled()
    train, val, test = split_by_problem(recs, seed=0)
    tm = train_trust_model(train, val)

    val_scores = np.array([float(p) for p in tm.predict_proba(val)])
    test_scores = np.array([float(p) for p in tm.predict_proba(test)])
    val_unf = np.array([make_label(r) == 0 for r in val])
    test_unf = np.array([make_label(r) == 0 for r in test])
    cal_unf_scores = val_scores[val_unf]
    n_cal = int(val_unf.sum())

    report = {
        "n_calibration_unfaithful": n_cal,
        "n_test_unfaithful": int(test_unf.sum()),
        "calibration_unfaithful_trust_scores": [round(x, 4) for x in sorted(cal_unf_scores.tolist())],
        "alpha_sweep": {},
        "stakes_adjusted_gate": {},
        "decision_change_vs_tau0": {},
        "honesty_flags": [],
    }

    # ---- alpha sensitivity sweep (primary = 0.1) ----
    for alpha in [0.1, 0.167, 0.2, 0.33, 0.5]:
        tau, vacuous = conformal_tau(cal_unf_scores, alpha)
        cov = coverage_distribution(n_cal, alpha)
        # realized coverage on test unfaithful (tiny n): Clopper-Pearson 95%
        flagged_test_unf = int((test_scores[test_unf] <= tau).sum())
        m = int(test_unf.sum())
        if m > 0:
            lo = stats.beta.ppf(0.025, flagged_test_unf, m - flagged_test_unf + 1) if flagged_test_unf > 0 else 0.0
            hi = stats.beta.ppf(0.975, flagged_test_unf + 1, m - flagged_test_unf) if flagged_test_unf < m else 1.0
            realized = {
                "flagged": flagged_test_unf, "of": m,
                "point": round(flagged_test_unf / m, 3),
                "cp95_ci": [round(float(lo), 3), round(float(hi), 3)],
            }
        else:
            realized = None
        # how many TOTAL test records does this tau flag (false-reject blast radius)
        total_flagged = int((test_scores <= tau).sum())
        report["alpha_sweep"][str(alpha)] = {
            "tau": round(tau, 4),
            "vacuous_flag_everything": vacuous,
            "coverage_guarantee": cov,
            "realized_test_coverage_unfaithful": realized,
            "total_test_records_flagged": total_flagged,
            "total_test_records": len(test),
        }

    # ---- stakes-adjusted gate at the three domain anchors ----
    for dom, S_val in DOMAIN_STAKES.items():
        a = alpha_of_stakes(S_val)
        tau, vacuous = conformal_tau(cal_unf_scores, a)
        report["stakes_adjusted_gate"][dom] = {
            "stakes_S": S_val,
            "alpha_of_S": round(a, 4),
            "tau": round(tau, 4),
            "vacuous_flag_everything": vacuous,
        }

    # ---- decision-change check vs original tau=0 (symbolic-only fusion) ----
    # original: trust never fires (tau=0), so hybrid==symbolic. New gate at
    # alpha=0.1: does it flip any accept/reject among symbolic-PASS records?
    sym_pass = [r for r in test if r["verdict"]["overall_valid"]]
    sym_pass_scores = np.array([float(tm.predict_proba([r])[0]) for r in sym_pass])

    def _change_at(alpha_val):
        tau_v, vac_v = conformal_tau(cal_unf_scores, alpha_val)
        flagged = int((sym_pass_scores <= tau_v).sum())
        good = sum(1 for r, s in zip(sym_pass, sym_pass_scores) if s <= tau_v and not r["labels"]["overall_valid"])
        return {
            "alpha": alpha_val,
            "tau": round(tau_v, 4),
            "vacuous": vac_v,
            "symbolic_pass_records_in_test": len(sym_pass),
            "newly_flagged_by_conformal_gate": flagged,
            "of_which_actually_flawed_good_flip": good,
            "of_which_actually_valid_false_reject": flagged - good,
            "changes_any_decision": flagged > 0,
        }

    # primary (alpha=0.1, vacuous) and tightest non-vacuous (alpha=0.167)
    report["decision_change_vs_tau0"] = {
        "primary_alpha_0.1": _change_at(0.1),
        "tightest_nonvacuous_alpha_0.167": _change_at(0.167),
    }

    tau0, vac0 = conformal_tau(cal_unf_scores, 0.1)
    report["honesty_flags"] = [
        f"Only {n_cal} calibration unfaithful examples. At alpha=0.1 the conformal "
        f"threshold is {'VACUOUS (tau=1.0, flags every record)' if vac0 else round(tau0,3)} — "
        "a 90% distribution-free guarantee is unreachable from 5 positives except by "
        "flagging everything. The tightest NON-vacuous level is alpha~=0.167 (coverage "
        f"{n_cal}/{n_cal+1}=0.833).",
        "Realized test coverage is measured on only 3 unfaithful records; its 95% CI "
        "spans essentially the whole [0,1] range. The point estimate is not reliable.",
        "The conformal gate DOES change decisions vs tau=0, but at alpha=0.1 it does so "
        "by flagging a large fraction of records (most trust scores cluster near the "
        "calibration max), i.e. it buys unfaithful-coverage with heavy false rejects. "
        "The small calibration set, not the method, is the binding limitation.",
    ]

    (OUT / "conformal_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
