"""Task 3 (pivot, reparameterized): per-judge-tier expected-cost model.

The Sonnet-5 null-degradation result (Task 1/1b) reparameterizes this from
"judge vs symbolic" into a per-tier cost comparison with a real billed-cost
term:

    expected_cost(tier, H, S, r, lam) =
        (1 - recall(tier,H)) * S          # missed-flaw cost (false accepts)
      + false_reject_rate(tier,H) * r     # wasted-replan cost (false rejects)
      + call_cost_usd(tier,H) * lam       # real per-record API $ cost

All recall / false-reject / token numbers are LOADED or COMPUTED from
existing result artifacts (verifier/data/horizon/*, pivot/results/sonnet_*),
never regenerated or estimated. Symbolic recall=1.0 is VERIFIED against the
loaded rule-parsed verdicts, not assumed. S, r, lam come from pivot.config
(stated assumptions, not tuned).

Deliverables:
  1. pivot/results/decision_table.json  — the traceable per-(tier,horizon)
     inputs (recall, fr, mean tokens, $ call cost, record basis).
  2. Pareto frontier over (S, lam) at each horizon: which tier is
     expected-cost-optimal in each region (pivot/results/figures/).
  3. Crossover-vs-S for the cheap-judge (Haiku) tier (original deliverable).
  4. Honest findings JSON incl. the "does stakes-pricing move the boundary
     earlier than raw-recall divergence?" check and the H=5 token-cost gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pivot.config import (
    DOMAIN_STAKES,
    LAMBDA_DEFAULT,
    LAMBDA_SWEEP,
    OUTPUT_PRICE_PER_TOKEN_USD,
    REPLAN_COST_DEFAULT,
    STAKES_SWEEP,
)

HDATA = Path("verifier/data/horizon")
OUT = Path("pivot/results")
DOM = ["blocksworld", "logistics", "tools"]
HORIZONS = [10, 20, 40, 80]


# ---------------------------------------------------------------------------
# Load / compute per-(tier, horizon) numbers from existing artifacts
# ---------------------------------------------------------------------------
def _load(pat: str) -> list[dict]:
    recs = []
    for d in DOM:
        p = HDATA / pat.format(d=d)
        if p.exists():
            recs += [json.loads(l) for l in p.open()]
    return recs


def _recall_fr(records, reject_of) -> tuple:
    flawed = [r for r in records if not r["labels"]["overall_valid"]]
    valid = [r for r in records if r["labels"]["overall_valid"]]
    tp = sum(1 for r in flawed if reject_of(r))
    fp = sum(1 for r in valid if reject_of(r))
    recall = tp / len(flawed) if flawed else None
    fr = fp / len(valid) if valid else None  # None when no valid plans (H=80)
    return recall, fr, len(flawed), len(valid)


def build_table() -> dict:
    table = {}  # (tier, horizon) -> dict
    for h in HORIZONS:
        bl = _load(f"{{d}}_h{h}_baselines.jsonl")
        vr = _load(f"{{d}}_h{h}_verdicts.jsonl")  # rule-parsed symbolic, full coverage

        # symbolic (extract+check strategy): use the LLM-extraction verdict where
        # available for the strategy's true false-reject behavior; rule-parsed
        # gives full coverage and is the sound reference. At H>=10 extraction
        # fidelity was 100%, so the two coincide; we use rule-parsed for full n.
        zs_r, zs_fr, nf, nv = _recall_fr(bl, lambda r: r["judge_zeroshot_valid"] is False)
        cot_r, cot_fr, _, _ = _recall_fr(bl, lambda r: r["judge_cot_valid"] is False)
        sym_r, sym_fr, _, _ = _recall_fr(vr, lambda r: not r["verdict"]["overall_valid"])

        zt = [r["judge_zeroshot_output_tokens"] for r in bl if r.get("judge_zeroshot_output_tokens")]
        ct = [r["judge_cot_output_tokens"] for r in bl if r.get("judge_cot_output_tokens")]

        table[("symbolic", h)] = _row("symbolic", sym_r, sym_fr, 0.0, nf, nv, f"rule-parsed verdicts, n={len(vr)}")
        table[("haiku_zeroshot", h)] = _row(
            "claude-haiku-4-5", zs_r, zs_fr, float(np.mean(zt)), nf, nv, f"baselines, n={len(bl)}"
        )
        table[("haiku_cot", h)] = _row(
            "claude-haiku-4-5", cot_r, cot_fr, float(np.mean(ct)), nf, nv, f"baselines, n={len(bl)}"
        )

    _build_h5(table)  # Task 6: short-horizon cell, token cost recovered separately

    # Sonnet-5 tier: only the two measured points (H=20 full-120, H=40 36-subset).
    s20 = json.load(open(OUT / "sonnet_judge_h20.json"))
    s40 = json.load(open(OUT / "sonnet_judge_h40.json"))
    z20 = s20["sonnet5_new"]["llm_judge_zeroshot"]
    # fr from precision: precision 1.0 => 0 false rejects (verified via P=1.0)
    table[("sonnet_zeroshot", 20)] = _row(
        "claude-sonnet-5", z20["recall"], 0.0 if z20["precision"] == 1.0 else None,
        z20["mean_output_tokens"], None, None, f"sonnet_judge_h20.json, n={z20['n']}, P={z20['precision']}"
    )
    z40 = s40["sonnet5_zeroshot"]
    table[("sonnet_zeroshot", 40)] = _row(
        "claude-sonnet-5", z40["recall"], 0.0 if z40["precision"] == 1.0 else None,
        z40["mean_output_tokens"], None, None, f"sonnet_judge_h40.json, n={z40['n']}, P={z40['precision']}"
    )
    return table


def _build_h5(table: dict) -> None:
    """Task 6: add the H=5 short-horizon cell. Data from the synthetic test
    split (72 records); judge token cost RECOVERED via pivot.h5_judge_tokens
    (re-run of only the missing Haiku zs/CoT combos), not estimated. Unlike
    H>=10, symbolic here has a nonzero false-reject rate (LLM-extraction noise
    on 3 hyphenated-arg tool plans), while both judges have recall 1.0 fr 0."""
    from verifier.learned import split_by_problem

    Sy = Path("verifier/data/synthetic")
    recs = []
    for d in DOM:
        bl = {(x["problem_name"], x["condition"]): x for x in (json.loads(l) for l in open(Sy / f"{d}_baselines.jsonl"))}
        for line in open(Sy / f"{d}_verdicts_llm.jsonl"):
            r = json.loads(line)
            r["baselines"] = bl.get((r["problem_name"], r["condition"]), {})
            recs.append(r)
    _, _, test = split_by_problem(recs, seed=0)

    sym_r, sym_fr, nf, nv = _recall_fr(test, lambda r: not r["verdict"]["overall_valid"])
    zs_r, zs_fr, _, _ = _recall_fr(test, lambda r: r["baselines"].get("judge_zeroshot_valid") is False)
    cot_r, cot_fr, _, _ = _recall_fr(test, lambda r: r["baselines"].get("judge_cot_valid") is False)

    tok = json.load(open(OUT / "h5_judge_tokens.json"))["combos"]
    table[("symbolic", 5)] = _row("symbolic", sym_r, sym_fr, 0.0, nf, nv, f"synthetic test split, n={len(test)}")
    table[("haiku_zeroshot", 5)] = _row(
        "claude-haiku-4-5", zs_r, zs_fr, tok["haiku_zeroshot"]["mean_output_tokens"], nf, nv,
        f"synthetic test split n={len(test)}; tokens recovered (h5_judge_tokens.json)"
    )
    table[("haiku_cot", 5)] = _row(
        "claude-haiku-4-5", cot_r, cot_fr, tok["haiku_cot"]["mean_output_tokens"], nf, nv,
        f"synthetic test split n={len(test)}; tokens recovered (h5_judge_tokens.json)"
    )


def _row(model, recall, fr, mean_tokens, nf, nv, basis) -> dict:
    price = OUTPUT_PRICE_PER_TOKEN_USD.get(model, 0.0)
    call_cost = mean_tokens * price if mean_tokens is not None else 0.0
    return {
        "model": model,
        "recall": None if recall is None else round(recall, 4),
        "false_reject_rate": None if fr is None else round(fr, 4),
        "mean_output_tokens": None if mean_tokens is None else round(mean_tokens, 1),
        "call_cost_usd": round(call_cost, 8),
        "n_flawed": nf,
        "n_valid": nv,
        "basis": basis,
    }


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def expected_cost(row: dict, S: float, r: float, lam: float) -> float:
    """(1-recall)*S + fr*r + call_cost*lam. Requires recall and fr present."""
    if row["recall"] is None or row["false_reject_rate"] is None:
        return np.nan
    return (1 - row["recall"]) * S + row["false_reject_rate"] * r + row["call_cost_usd"] * lam


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    table = build_table()

    # ---- verify symbolic recall == 1.0 at every horizon (do not assume) ----
    sym_recalls = {h: table[("symbolic", h)]["recall"] for h in HORIZONS}
    symbolic_perfect = all(v == 1.0 for v in sym_recalls.values())
    assert symbolic_perfect, f"symbolic recall not 1.0 everywhere: {sym_recalls}"

    # serialize the traceable input table
    ser = {f"{tier}@H{h}": row for (tier, h), row in sorted(table.items(), key=lambda x: (x[0][1], x[0][0]))}
    (OUT / "decision_table.json").write_text(json.dumps(ser, indent=2))

    findings = {
        "symbolic_recall_by_horizon": sym_recalls,
        "symbolic_recall_verified_1.0": symbolic_perfect,
        "dominance": {},
        "pareto_by_horizon": {},
        "h5_gap": {},
        "stakes_pricing_check": {},
        "honesty_flags": [],
    }

    # ---- dominance analysis at each horizon (does symbolic weakly dominate?) ----
    for h in HORIZONS:
        sym = table[("symbolic", h)]
        tiers = [("haiku_zeroshot", h), ("haiku_cot", h)]
        if ("sonnet_zeroshot", h) in table:
            tiers.append(("sonnet_zeroshot", h))
        notes = []
        for t in tiers:
            row = table[t]
            if row["recall"] is None or row["false_reject_rate"] is None:
                notes.append(f"{t[0]}: fr undefined (no valid plans at H={h}); recall={row['recall']}")
                continue
            # symbolic weakly dominates row iff recall>=, fr<=, cost<= (one strict)
            better_recall = sym["recall"] >= row["recall"]
            better_fr = sym["false_reject_rate"] <= row["false_reject_rate"]
            better_cost = sym["call_cost_usd"] <= row["call_cost_usd"]
            strict = (sym["recall"] > row["recall"]) or (sym["false_reject_rate"] < row["false_reject_rate"]) or (sym["call_cost_usd"] < row["call_cost_usd"])
            dominates = better_recall and better_fr and better_cost and strict
            ties_accuracy = sym["recall"] == row["recall"] and sym["false_reject_rate"] == row["false_reject_rate"]
            notes.append(
                f"{t[0]}: recall {row['recall']} vs sym {sym['recall']}, fr {row['false_reject_rate']} vs 0, "
                f"$cost {row['call_cost_usd']:.2e} vs 0 -> symbolic {'WEAKLY DOMINATES' if dominates else 'does NOT dominate'}"
                + ("; ties on accuracy, symbolic wins only via call cost (lam>0)" if ties_accuracy else "")
            )
        findings["dominance"][f"H{h}"] = notes

    # ---- Pareto frontier over (S, lam) at H=20 and H=40 (Sonnet available) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    for h in [5, 20, 40]:
        tiers = [("symbolic", h), ("haiku_zeroshot", h), ("haiku_cot", h), ("sonnet_zeroshot", h)]
        tiers = [t for t in tiers if t in table and table[t]["recall"] is not None and table[t]["false_reject_rate"] is not None]
        names = [t[0] for t in tiers]
        Sg, Lg = np.meshgrid(STAKES_SWEEP, LAMBDA_SWEEP)
        winner = np.zeros_like(Sg, dtype=int)
        for i in range(Sg.shape[0]):
            for j in range(Sg.shape[1]):
                costs = [expected_cost(table[t], Sg[i, j], REPLAN_COST_DEFAULT, Lg[i, j]) for t in tiers]
                winner[i, j] = int(np.argmin(costs))
        region_counts = {names[k]: int((winner == k).sum()) for k in range(len(names))}
        findings["pareto_by_horizon"][f"H{h}"] = {
            "tiers": names,
            "grid_cells_won": region_counts,
            "note": "cells where each tier is expected-cost-optimal over the (S in [1,200], lam in [0.01,100]) grid",
        }
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = ListedColormap(["#2a6", "#e8a", "#c55", "#58c"][: len(names)])
        pc = ax.pcolormesh(Sg, Lg, winner, cmap=cmap, shading="auto", vmin=0, vmax=len(names) - 1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("stakes S (missed-flaw cost)")
        ax.set_ylabel(r"$\lambda$ (dollar-cost weight)")
        ax.set_title(f"Expected-cost-optimal strategy at H={h}")
        for dom, s in DOMAIN_STAKES.items():
            ax.axvline(s, color="k", ls=":", lw=0.8, alpha=0.6)
            ax.text(s, LAMBDA_SWEEP[-1], dom, rotation=90, va="top", ha="right", fontsize=7)
        cbar = fig.colorbar(pc, ticks=range(len(names)))
        cbar.ax.set_yticklabels(names, fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"pareto_H{h}.png", dpi=150)
        fig.savefig(OUT / "figures" / f"pareto_H{h}.svg")
        plt.close(fig)

    # ---- expected cost vs horizon at each domain stakes anchor (readable view
    #      of the dominance; lam and r at defaults) ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    tier_style = {
        "symbolic": ("#2a6", "-", "o"),
        "haiku_zeroshot": ("#c55", "--", "s"),
        "haiku_cot": ("#e59", "--", "^"),
        "sonnet_zeroshot": ("#58c", ":", "D"),
    }
    for ax, (dom, S) in zip(axes, DOMAIN_STAKES.items()):
        for tier in ["symbolic", "haiku_zeroshot", "haiku_cot", "sonnet_zeroshot"]:
            xs, ys = [], []
            for h in HORIZONS:
                if (tier, h) in table:
                    c = expected_cost(table[(tier, h)], S, REPLAN_COST_DEFAULT, LAMBDA_DEFAULT)
                    if not np.isnan(c):
                        xs.append(h)
                        ys.append(c)
            if xs:
                col, ls, mk = tier_style[tier]
                ax.plot(xs, ys, ls, color=col, marker=mk, ms=4, label=tier)
        ax.set_xscale("log")
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([str(h) for h in HORIZONS])
        ax.set_title(f"{dom}  (S={S:g}, r={REPLAN_COST_DEFAULT:g}, $\\lambda$={LAMBDA_DEFAULT:g})", fontsize=9)
        ax.set_xlabel("horizon")
        ax.set_ylabel("expected cost")
        ax.legend(fontsize=7)
    fig.suptitle("Expected cost by strategy vs horizon (lower is better) — symbolic is the flat zero line", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "expected_cost_vs_horizon.png", dpi=150)
    fig.savefig(OUT / "figures" / "expected_cost_vs_horizon.svg")
    plt.close(fig)

    # ---- Task 6: H=5 token-cost gap is now CLOSED (tokens recovered by
    #      re-running only the missing Haiku zs/CoT combos; see
    #      pivot/h5_judge_tokens.py, 72/72 verdict-match). The H=5 cell is now
    #      in the table and the Pareto grid above. ----
    sym5, zs5 = table[("symbolic", 5)], table[("haiku_zeroshot", 5)]
    h5_pareto = findings["pareto_by_horizon"].get("H5", {}).get("grid_cells_won", {})
    total_cells = len(STAKES_SWEEP) * len(LAMBDA_SWEEP)
    findings["h5_gap_RESOLVED"] = {
        "status": "CLOSED via re-run (no estimate); tokens 72/72 verdict-matched",
        "symbolic_H5": {"recall": sym5["recall"], "false_reject_rate": sym5["false_reject_rate"], "call_cost": 0.0},
        "haiku_zeroshot_H5": {"recall": zs5["recall"], "false_reject_rate": zs5["false_reject_rate"], "call_cost_usd": zs5["call_cost_usd"]},
        "grid_cells_won_H5": h5_pareto,
        "changes_the_finding": True,
        "note": (
            f"The finding CHANGES at H=5. Unlike H>=10 (symbolic 3600/3600), at H=5 the "
            f"symbolic strategy wins only {h5_pareto.get('symbolic', '?')}/{total_cells} cells; the "
            f"cheap Haiku zero-shot judge wins {h5_pareto.get('haiku_zeroshot', '?')}/{total_cells}. "
            "Reason: at H=5 LLM-extraction noise gives the symbolic strategy a nonzero "
            "false-reject rate (0.068, the 3 hyphenated-arg tool plans), while the judge "
            "has recall 1.0 AND fr 0; the judge's dollar cost is tiny, so it is "
            "expected-cost-optimal except in the high-lambda strip where call cost "
            "dominates. So horizon genuinely matters: at short horizon a cheap accurate "
            "judge can beat the extract+check strategy; at H>=10 the checker dominates."
        ),
    }

    # ---- "does pricing stakes move the boundary earlier than raw-recall
    #      divergence?" honest check ----
    # Raw recall diverges (judge < symbolic) starting at H=10 already (judges
    # <1.0 from H=10 up). Since symbolic recall=1.0 and fr=0 at every H>=10,
    # symbolic dominates on ACCURACY ALONE at H>=10, before stakes enter. So
    # stakes-pricing cannot move a crossover "earlier" at H>=10: there is no
    # judge-favorable region there for stakes to shrink. Stakes only matter in
    # the H<10 regime where strong judges edge symbolic on accuracy.
    findings["stakes_pricing_check"] = {
        "raw_recall_divergence_horizon": 10,
        "symbolic_dominant_on_accuracy_alone_at": "H>=10 (recall 1.0, fr 0)",
        "conclusion": (
            "The 'stakes-pricing moves the boundary earlier' claim does NOT "
            "hold in the direction originally hypothesized: at every horizon "
            ">=10 the symbolic strategy already dominates every judge tier on "
            "recall and false-rejects BEFORE stakes are priced in, so stakes "
            "cannot move a crossover earlier (no judge-optimal region exists "
            "there). Pricing stakes only sharpens WHY to prefer symbolic (it "
            "adds the missed-flaw term on top of an already-winning position). "
            "The reframed, data-supported claim: a strong judge (Sonnet-5) "
            "matches symbolic on accuracy at H=20/40, but symbolic is still "
            "expected-cost-optimal because its call cost is zero -- soundness "
            "at zero marginal cost, not accuracy superiority, is the win."
        ),
    }

    findings["honesty_flags"] = [
        "Sonnet-5 matches symbolic on recall (1.0) at H=20 and H=40; the symbolic "
        "advantage in this cost model is ZERO CALL COST + guaranteed soundness, "
        "NOT higher accuracy than the strong judge. Do not overclaim accuracy.",
        "At lambda=0 (dollar cost ignored) Sonnet-5 TIES symbolic at H=20/40; "
        "symbolic is not strictly optimal there. Reported, not hidden.",
        "Task 6 UPDATE: the H=5 gap is closed (tokens recovered by re-run, not "
        "estimated) and it CHANGES the finding: at H=5 the cheap Haiku zero-shot "
        "judge is expected-cost-optimal in the large majority of the grid "
        f"({findings['pareto_by_horizon'].get('H5', {}).get('grid_cells_won', {}).get('haiku_zeroshot', '?')}/3600), "
        "because the extract+check strategy has a nonzero false-reject rate there. "
        "Symbolic dominance is a H>=10 property, not universal across all horizons.",
    ]

    (OUT / "decision_findings.json").write_text(json.dumps(findings, indent=2))
    print(json.dumps(findings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
