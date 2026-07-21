"""Task 5 (pivot, now PRIMARY): guardrail-collapse taxonomy.

Reorganizes the existing Section 5.5 judge-budget-sweep finding
(64/512/2048 output tokens) into a taxonomy of SILENT failure modes, using
ONLY data already collected (no new API calls). Then a worked example plugs
the real observed recall numbers into Task 3's expected_cost formula to show
the gap between a deployer's APPARENT cost (truncation undetected) and the
TRUE cost.

Why this is the primary argument after the Sonnet-5 result: Sonnet-5 matches
the symbolic checker on raw accuracy (recall 1.0 at H=20/40), so "judges
degrade with horizon" is NOT the checker's selling point. What IS
tier-independent is that the checker cannot enter these failure modes at all
— it has no token budget to truncate, is a total deterministic function so
it cannot emit an unparseable verdict, fails CLOSED by construction, and
costs zero at decision time. Those properties hold no matter how strong the
judge model is.

Observed data source (Section 5.5, PROGRESS.md — these ARE the recorded dev
runs, 360 short-horizon records):
  budget 64   : 360/360 unparseable (100%), fail-open -> P=R=F1=0.0
  budget 512  : 146/360 unparseable (~41%)
  budget 2048 : 0/360 unparseable, judge recovers to perfect at short horizon
We do NOT invent failure categories beyond what these runs evidence (e.g. no
prompt-injection row — that is a separate, unobserved, scoped-out experiment).
"""

from __future__ import annotations

import json
from pathlib import Path

from pivot.config import DOMAIN_STAKES, LAMBDA_DEFAULT, REPLAN_COST_DEFAULT

OUT = Path("pivot/results")

# ---- observed Section 5.5 budget-sweep data (recorded dev runs) ----
BUDGET_SWEEP = [
    {"budget": 64, "n": 360, "unparseable": 360, "unparse_rate": 1.00, "precision": 0.0, "recall": 0.0, "f1": 0.0},
    {"budget": 512, "n": 360, "unparseable": 146, "unparse_rate": 0.41, "precision": None, "recall": None, "f1": None,
     "note": "partial collapse; per-record P/R not separately tabulated in the dev log beyond the unparseable rate"},
    {"budget": 2048, "n": 360, "unparseable": 0, "unparse_rate": 0.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
     "note": "adequate budget; judge recovers to perfect at short horizon (3-8 steps)"},
]

# ---- taxonomy of observed silent-failure modes ----
TAXONOMY = [
    {
        "category": "truncation-before-verdict",
        "trigger": "output budget too small for the model's actual verbosity "
                   "(haiku simulates the plan step-by-step regardless of the "
                   "'answer in one line' instruction); observed at 64 and 512 tokens",
        "observed_effect_on_PR": "unparseable-verdict rate 100% at 64 tokens, ~41% at 512; "
                                 "no exception is raised — the call 'succeeds' with a truncated body",
        "deployment_default": "n/a (mechanism); becomes catastrophic or benign depending on the "
                              "unparseable-handling policy below",
        "checker_immune": "yes — the symbolic checker has no generative budget to exhaust",
    },
    {
        "category": "fail-open-on-unparseable-output",
        "trigger": "any unparseable verdict combined with the naive deployment default of "
                   "treating 'no parseable verdict' as ACCEPT",
        "observed_effect_on_PR": "at 64 tokens this turns 100% truncation into P=R=F1=0.0: every "
                                 "flawed plan is accepted and executed, while the guardrail appears "
                                 "to run normally (no errors, fast 'approvals')",
        "deployment_default": "FAIL-OPEN (observed default). A FAIL-CLOSED default would instead "
                              "reject all 360 (false-reject storm) — same bug, opposite failure; "
                              "either way the judge's verdict is meaningless, just silently",
        "checker_immune": "yes — the checker is a total deterministic function; it always emits a "
                          "parseable verdict and fails CLOSED (an unrunnable step is a violation)",
    },
]


def expected_cost(recall, fr, call_cost_usd, S, r, lam):
    return (1 - recall) * S + fr * r + call_cost_usd * lam


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- worked example: apparent vs true expected cost, per Task 3 formula ----
    # Context = short horizon (where 5.5 was measured). Recall values:
    #   truncated (64-tok) judge : recall 0.0  (fail-open collapse, observed)
    #   working  (2048-tok) judge: recall 1.0  (perfect at short horizon, observed)
    #   symbolic checker         : recall 1.0  (sound; verified elsewhere)
    # A deployer who does not notice the 64-token truncation ASSUMES they have
    # the working judge (recall ~1.0 -> apparent missed-flaw cost ~0). The TRUE
    # missed-flaw cost is (1-0)*S = S per flawed plan. call_cost term is dropped
    # here (dominated by the stakes gap and unavailable at H=5); noted, not hidden.
    worked = {}
    for dom, S in DOMAIN_STAKES.items():
        apparent = expected_cost(recall=1.0, fr=0.0, call_cost_usd=0.0, S=S, r=REPLAN_COST_DEFAULT, lam=LAMBDA_DEFAULT)
        true_cost = expected_cost(recall=0.0, fr=0.0, call_cost_usd=0.0, S=S, r=REPLAN_COST_DEFAULT, lam=LAMBDA_DEFAULT)
        worked[dom] = {
            "stakes_S": S,
            "apparent_expected_missed_flaw_cost_if_truncation_undetected": round(apparent, 3),
            "true_expected_missed_flaw_cost_per_flawed_plan": round(true_cost, 3),
            "hidden_cost_gap": round(true_cost - apparent, 3),
        }

    report = {
        "budget_sweep_observed": BUDGET_SWEEP,
        "taxonomy": TAXONOMY,
        "worked_example": {
            "context": "short-horizon (3-8 step) records, where the 64/512/2048 budget sweep was measured",
            "formula": "expected_cost = (1-recall)*S + fr*r + call_cost*lambda (Task 3)",
            "truncated_judge_recall": 0.0,
            "working_judge_recall": 1.0,
            "symbolic_recall": 1.0,
            "by_domain": worked,
            "reading": (
                "At 64 tokens the guardrail silently has recall 0.0 while returning fast, "
                "error-free 'approvals'. A deployer who assumes it works like the 2048-token "
                "judge estimates ~0 missed-flaw cost; the true cost is the full stakes S per "
                "flawed plan (1, 10, and 50 at the blocksworld/logistics/tools anchors). The "
                "hidden gap grows linearly with stakes — exactly where a guardrail matters most."
            ),
        },
        "primary_argument": (
            "Because Sonnet-5 matches the checker on accuracy (Task 1/1b), the checker's "
            "durable, tier-INDEPENDENT value is that it structurally cannot enter these "
            "failure modes: no token budget (immune to truncation), total deterministic "
            "verdict (cannot produce unparseable output), fails CLOSED by construction, and "
            "zero marginal decision-time cost. These hold regardless of which judge tier a "
            "deployer would otherwise trust."
        ),
        "honesty_flags": [
            "The 512-token row's per-record P/R was not separately tabulated in the original "
            "dev log beyond the 41% unparseable rate; reported as a gap, not back-filled.",
            "The worked example drops the call_cost*lambda term (H=5 judge token cost was never "
            "logged; and it is dominated by the stakes gap anyway). Stated, not hidden.",
            "Adversarial prompt-injection immunity is claimed for the checker on structural "
            "grounds ONLY; it was NOT tested here (scoped out) and is not a row in the observed "
            "taxonomy. Listed as a conceptual advantage / follow-up, not an empirical result.",
        ],
    }

    (OUT / "guardrail_taxonomy.json").write_text(json.dumps(report, indent=2))

    # ---- also emit a readable markdown table for the paper ----
    md = ["# Guardrail-collapse taxonomy (Task 5)", "",
          "Observed judge-budget sweep (Section 5.5, 360 short-horizon records):", "",
          "| budget (tok) | unparseable | P | R | F1 |", "|---|---|---|---|---|"]
    for b in BUDGET_SWEEP:
        p = "-" if b["precision"] is None else b["precision"]
        r_ = "-" if b["recall"] is None else b["recall"]
        f_ = "-" if b["f1"] is None else b["f1"]
        md.append(f"| {b['budget']} | {b['unparseable']}/{b['n']} ({b['unparse_rate']:.0%}) | {p} | {r_} | {f_} |")
    md += ["", "## Silent-failure modes", "",
           "| category | trigger | observed effect on P/R | deployment default | checker immune? |",
           "|---|---|---|---|---|"]
    for t in TAXONOMY:
        md.append(f"| {t['category']} | {t['trigger']} | {t['observed_effect_on_PR']} | "
                  f"{t['deployment_default']} | {t['checker_immune']} |")
    md += ["", "## Worked example: apparent vs true expected cost", "",
           "| domain | stakes S | apparent (truncation undetected) | true (recall 0) | hidden gap |",
           "|---|---|---|---|---|"]
    for dom, w in worked.items():
        md.append(f"| {dom} | {w['stakes_S']:g} | "
                  f"{w['apparent_expected_missed_flaw_cost_if_truncation_undetected']:g} | "
                  f"{w['true_expected_missed_flaw_cost_per_flawed_plan']:g} | {w['hidden_cost_gap']:g} |")
    md += ["", report["primary_argument"], ""]
    (OUT / "guardrail_taxonomy.md").write_text("\n".join(md))

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
