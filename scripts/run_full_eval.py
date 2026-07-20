"""CLI: full evaluation grid (Phase 8) — every system over every domain,
ablations, calibration, latency/cost proxies, the downstream
execute-or-reject experiment, figures, and qualitative examples.

Reproducible: everything in results/ is regenerated from the JSONL data by
this one script.

Usage:
    python -m scripts.run_full_eval --out results/ [--skip-downstream]

Inputs (produced by earlier pipeline stages):
    verifier/data/synthetic/{domain}_verdicts_llm.jsonl  (production path)
    verifier/data/synthetic/{domain}_baselines.jsonl     (LLM baselines)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from verifier.baselines import train_learned_only
from verifier.eval import (
    clean_but_flawed_analysis,
    evaluate_system,
    hybrid_reject_fn,
    measure_latency,
    pick_threshold,
    run_downstream,
    trust_calibration,
)
from verifier.fusion import decide, sweep_threshold
from verifier.learned import make_label, split_by_problem, train_trust_model
from verifier.llm import DEFAULT_MODEL

DOMAINS = ["blocksworld", "logistics", "tools"]
DATA = Path("verifier/data/synthetic")


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def _join_baselines(records: list[dict], baselines: list[dict]) -> None:
    """Attach baseline outputs to verdict records in place (keyed by
    problem_name + condition; one plan per key by construction)."""
    index = {(b["problem_name"], b["condition"]): b for b in baselines}
    for r in records:
        b = index.get((r["problem_name"], r["condition"]))
        r["baselines"] = b or {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_full_eval", description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-downstream", action="store_true", help="skip the API-calling downstream experiment")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    out: Path = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # ---------- load + join ----------
    all_records: dict[str, list[dict]] = {}
    for domain in DOMAINS:
        records = _load(DATA / f"{domain}_verdicts_llm.jsonl")
        baselines_path = DATA / f"{domain}_baselines.jsonl"
        _join_baselines(records, _load(baselines_path) if baselines_path.exists() else [])
        all_records[domain] = records

    pooled = [r for records in all_records.values() for r in records]
    train, val, test = split_by_problem(pooled, seed=args.seed)
    print(f"pooled: {len(pooled)} records -> {len(train)} train / {len(val)} val / {len(test)} test")

    # ---------- train models ----------
    trust_model = train_trust_model(train, val)
    learned_only_model = train_learned_only(train)
    threshold = pick_threshold(val, trust_model)
    print(f"trust threshold picked on val: {threshold}")

    labels_all = np.array([make_label(r) for r in pooled])
    print(f"trust label balance (pooled): {labels_all.mean():.1%} faithful")

    # ---------- system reject functions ----------
    hybrid_fn = hybrid_reject_fn(trust_model, threshold)
    symbolic_fn = lambda r: not r["verdict"]["overall_valid"]
    learned_only_fn = lambda r: float(learned_only_model.predict_valid_proba([r])[0]) < 0.5
    judge_zs_fn = lambda r: r["baselines"].get("judge_zeroshot_valid") is False
    judge_cot_fn = lambda r: r["baselines"].get("judge_cot_valid") is False
    self_repair_fn = lambda r: bool(r["baselines"].get("self_repair_changed_plan"))

    systems = [
        ("hybrid", hybrid_fn),
        ("symbolic_only", symbolic_fn),
        ("learned_only", learned_only_fn),
        ("llm_judge_zeroshot", judge_zs_fn),
        ("llm_judge_cot", judge_cot_fn),
        ("self_repair_detect", self_repair_fn),
    ]

    # ---------- headline metrics ----------
    metrics_rows = []
    for domain in DOMAINS + ["ALL"]:
        domain_test = test if domain == "ALL" else [r for r in test if r["domain"] == domain]
        for name, fn in systems:
            m = evaluate_system(domain_test, fn, name)
            row = {
                "domain": domain,
                "system": name,
                "n": m["n"],
                "precision": round(m["overall"]["precision"], 3),
                "recall": round(m["overall"]["recall"], 3),
                "f1": round(m["overall"]["f1"], 3),
            }
            for ft, d in m["per_flaw_type"].items():
                row[f"recall_{ft}"] = None if d["recall"] is None else round(d["recall"], 3)
                row[f"support_{ft}"] = d["support"]
            metrics_rows.append(row)

    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    # ---------- self-repair repair-quality (separate lens) ----------
    repair_rows = []
    for domain in DOMAINS:
        domain_test = [r for r in test if r["domain"] == domain and r["baselines"]]
        flawed = [r for r in domain_test if not r["labels"]["overall_valid"]]
        fixed = sum(1 for r in flawed if r["baselines"].get("self_repair_repaired_valid"))
        ok = [r for r in domain_test if r["labels"]["overall_valid"]]
        broke = sum(1 for r in ok if not r["baselines"].get("self_repair_repaired_valid", True))
        repair_rows.append(
            {
                "domain": domain,
                "flawed_n": len(flawed),
                "flawed_fixed_by_repair": fixed,
                "valid_n": len(ok),
                "valid_broken_by_repair": broke,
            }
        )
    (out / "self_repair_quality.json").write_text(json.dumps(repair_rows, indent=2))

    # ---------- calibration ----------
    calib = trust_calibration(test, trust_model)
    (out / "calibration.json").write_text(json.dumps(calib, indent=2))

    # ---------- clean-but-flawed (the load-bearing analysis) ----------
    cbf = clean_but_flawed_analysis(test, trust_model, threshold)
    (out / "clean_but_flawed.json").write_text(json.dumps(cbf, indent=2))

    # ---------- threshold sweep for the PR curve ----------
    probs = trust_model.predict_proba(test)
    sweep = sweep_threshold(
        [r["verdict"]["overall_valid"] for r in test],
        list(map(float, probs)),
        [not r["labels"]["overall_valid"] for r in test],
    )
    (out / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))

    # ---------- latency / cost proxies ----------
    lat_symbolic = measure_latency(test, symbolic_fn)
    lat_hybrid_local = measure_latency(test, hybrid_fn)
    cost_rows = {
        "symbolic_only": {"api_calls_per_plan": 0, "local_seconds_per_plan": lat_symbolic},
        "hybrid": {
            "api_calls_per_plan": "k*steps extraction (~12-24, cacheable/batchable)",
            "local_seconds_per_plan": lat_hybrid_local,
        },
        "llm_judge_zeroshot": {
            "api_calls_per_plan": 1,
            "note": "up to ~2048 output tokens (haiku simulates the plan even "
            "without a CoT instruction; tighter budgets truncate the verdict "
            "line — see LIMITATIONS.md)",
        },
        "llm_judge_cot": {"api_calls_per_plan": 1, "note": "up to ~2048 output tokens"},
        "self_repair": {"api_calls_per_plan": 1, "note": "critique + rewritten plan"},
    }
    (out / "latency_cost.json").write_text(json.dumps(cost_rows, indent=2))

    # ---------- downstream experiment ----------
    downstream = {}
    if not args.skip_downstream:
        from verifier.llm import get_client

        client = get_client()
        explanation_fn = lambda r: r["verdict"]["explanation"]
        downstream["no_verifier"] = run_downstream(test, None, None)
        downstream["hybrid_gated"] = run_downstream(
            test, hybrid_fn, explanation_fn, client=client, model=args.model
        )
        downstream["llm_judge_cot_gated"] = run_downstream(
            test, judge_cot_fn, lambda r: "the plan was judged invalid", client=client, model=args.model
        )
        (out / "downstream.json").write_text(json.dumps(downstream, indent=2))

    # ---------- figures ----------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PR curve from the threshold sweep
    fig, ax = plt.subplots(figsize=(5, 4))
    rs = [row["recall"] for row in sweep]
    ps = [row["precision"] for row in sweep]
    ax.plot(rs, ps, marker="o", ms=3, label="hybrid (trust threshold sweep)")
    sym_row = next(r for r in sweep if r["threshold"] == 0.0)
    ax.scatter([sym_row["recall"]], [sym_row["precision"]], color="red", zorder=5, label="symbolic-only (th=0)")
    ax.set_xlabel("recall (flawed plans)")
    ax.set_ylabel("precision")
    ax.set_title("Flaw detection PR curve (test split, pooled domains)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "pr_curve.png", dpi=150)
    fig.savefig(out / "figures" / "pr_curve.svg")

    # reliability diagram
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [b["conf"] for b in calib["bins"] if b["count"]]
    ys = [b["acc"] for b in calib["bins"] if b["count"]]
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.bar(xs, ys, width=0.08, alpha=0.7, label=f"ECE={calib['ece']:.3f}")
    ax.set_xlabel("predicted trust")
    ax.set_ylabel("empirical faithfulness")
    ax.set_title("Trust score reliability (test split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "reliability.png", dpi=150)
    fig.savefig(out / "figures" / "reliability.svg")

    # ablation bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    ablate = [r for r in metrics_rows if r["domain"] == "ALL"]
    names = [r["system"] for r in ablate]
    f1s = [r["f1"] for r in ablate]
    ax.bar(names, f1s, color=["#2a6" if n == "hybrid" else "#579" for n in names])
    ax.set_ylabel("F1 (flaw detection)")
    ax.set_title("Ablation: all systems, pooled test split")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out / "figures" / "ablation.png", dpi=150)
    fig.savefig(out / "figures" / "ablation.svg")

    # ---------- qualitative examples (error analysis) ----------
    lines = [
        "# Qualitative examples (auto-generated by scripts/run_full_eval.py)",
        "",
        "Positive class = flawed plan; 'caught' = system rejected it.",
        "",
    ]

    def _describe(r, note):
        plan = r.get("raw_llm_plan_original", r.get("raw_llm_plan", ""))[:400]
        return [
            f"### {r['problem_name']} [{r['condition']}] — {note}",
            "",
            f"- oracle: valid={r['labels']['overall_valid']}"
            f" (consistent={r['labels']['is_consistent']},"
            f" goal={r['labels']['is_goal_complete']},"
            f" resources={r['labels']['is_resource_feasible']})",
            f"- symbolic verdict (LLM-extraction path): {r['verdict']['explanation'][:300]}",
            f"- plan: `{plan.strip()[:350]}`",
            "",
        ]

    probs_by_id = {id(r): float(p) for r, p in zip(test, probs)}
    caught_by_hybrid_missed_by_judge = [
        r
        for r in test
        if not r["labels"]["overall_valid"]
        and hybrid_fn(r)
        and r["baselines"].get("judge_cot_valid") is not False
    ][:5]
    lines.append("## (a) Flawed plans the hybrid caught but the CoT LLM-judge accepted\n")
    for r in caught_by_hybrid_missed_by_judge:
        lines += _describe(r, "hybrid caught, judge missed")

    trust_caught_symbolic_missed = [
        r
        for r in test
        if not r["labels"]["overall_valid"]
        and r["verdict"]["overall_valid"]  # symbolic (production path) passed it
        and probs_by_id[id(r)] < threshold  # trust gate flagged it
    ][:5]
    lines.append("## (b) Symbolically-clean-but-flawed plans flagged by the trust layer\n")
    for r in trust_caught_symbolic_missed:
        lines += _describe(r, f"trust={probs_by_id[id(r)]:.2f} < th={threshold}")

    hybrid_wrong = [
        r
        for r in test
        if (not r["labels"]["overall_valid"]) != hybrid_fn(r)
    ][:5]
    lines.append("## (c) Hybrid mistakes (false accepts / false rejects)\n")
    for r in hybrid_wrong:
        kind = "false accept" if not r["labels"]["overall_valid"] else "false reject"
        lines += _describe(r, f"{kind}, trust={probs_by_id[id(r)]:.2f}")

    (out / "qualitative_examples.md").write_text("\n".join(lines))

    # ---------- summary ----------
    summary = {
        "n_pooled": len(pooled),
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "trust_threshold": threshold,
        "trust_label_balance_faithful": float(labels_all.mean()),
        "calibration_ece": calib["ece"],
        "clean_but_flawed": cbf,
        "downstream": downstream,
        "metrics_csv": str(out / "metrics.csv"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
