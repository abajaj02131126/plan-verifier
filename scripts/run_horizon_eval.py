"""CLI: horizon experiment evaluation — does the short-horizon picture
(judges perfect, symbolic+trust merely sound) change at 10/20/40/80-step
plans?

Inputs: verifier/data/horizon/{domain}_h{H}_*.jsonl (see
scripts/horizon_pipeline.sh) plus the short-horizon dev data in
verifier/data/synthetic/ as the reference point (plotted at its measured
mean gold length, ~5).

Design notes (documented ambiguity calls):
- The trust model and learned-only baseline are trained ONLY on the
  short-horizon records (same problem-level split as Phase 8) and applied
  zero-shot at every horizon: no long-horizon record is ever in training,
  so full cells can be evaluated without leakage, and the trust model's
  horizon generalization is itself measured.
- At the short-horizon reference point, systems are evaluated on the Phase 8
  72-record test split (the trust model saw the rest in training); at
  h>=10, on every available record (extraction-path systems: all 40/cell at
  h10/h20, the 12-record stratified subsets at h40/h80 — tiered design).

Usage:
    python -m scripts.run_horizon_eval --out results/horizon/ [--skip-downstream]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from verifier.baselines import train_learned_only
from verifier.eval import evaluate_system, hybrid_reject_fn, pick_threshold, run_downstream
from verifier.generation.parser import parse_plan
from verifier.learned import make_label, split_by_problem, train_trust_model
from verifier.llm import DEFAULT_MODEL
from verifier.schema import Problem

DOMAINS = ["blocksworld", "logistics", "tools"]
HORIZONS = [10, 20, 40, 80]
HDATA = Path("verifier/data/horizon")
SDATA = Path("verifier/data/synthetic")


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def _join_baselines(records: list[dict], baselines: list[dict]) -> None:
    index = {(b["problem_name"], b["condition"]): b for b in baselines}
    for r in records:
        r["baselines"] = index.get((r["problem_name"], r["condition"])) or {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_horizon_eval", description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/horizon"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-downstream", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    out: Path = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # ---------- train trust + learned-only on SHORT-HORIZON data only ----------
    short_records = []
    for d in DOMAINS:
        recs = _load(SDATA / f"{d}_verdicts_llm.jsonl")
        _join_baselines(recs, _load(SDATA / f"{d}_baselines.jsonl"))
        short_records.extend(recs)
    s_train, s_val, s_test = split_by_problem(short_records, seed=args.seed)
    trust_model = train_trust_model(s_train, s_val)
    learned_only_model = train_learned_only(s_train)
    threshold = pick_threshold(s_val, trust_model)
    print(f"trust model trained on {len(s_train)} short-horizon records; threshold={threshold}")

    # ---------- load horizon cells ----------
    # per horizon: extraction-path records (with verdict+extraction meta) and
    # judge/self-repair records (baselines joined onto oracle labels)
    ext_by_h: dict[int, list[dict]] = {}
    base_by_h: dict[int, list[dict]] = {}
    for h in HORIZONS:
        ext_by_h[h] = []
        base_by_h[h] = []
        for d in DOMAINS:
            base = HDATA / f"{d}_h{h}"
            ext = _load(base.with_name(base.name + "_verdicts_llm.jsonl"))
            bl = _load(base.with_name(base.name + "_baselines.jsonl"))
            _join_baselines(ext, bl)
            ext_by_h[h].extend(ext)
            # judge/self-repair evaluation uses ALL records (labels live on
            # the baseline rows already)
            base_by_h[h].extend(bl)

    # short-horizon reference: Phase 8 test split for every system
    ext_by_h[5] = s_test
    base_by_h[5] = [r["baselines"] | {"labels": r["labels"], "domain": r["domain"]} for r in s_test]
    all_h = [5] + HORIZONS

    # ---------- reject fns ----------
    hybrid_fn = hybrid_reject_fn(trust_model, threshold)
    symbolic_fn = lambda r: not r["verdict"]["overall_valid"]
    learned_only_fn = lambda r: float(learned_only_model.predict_valid_proba([r])[0]) < 0.5
    judge_zs_fn = lambda r: r.get("judge_zeroshot_valid") is False
    judge_cot_fn = lambda r: r.get("judge_cot_valid") is False
    self_repair_fn = lambda r: bool(r.get("self_repair_changed_plan"))

    ext_systems = [
        ("hybrid", hybrid_fn),
        ("symbolic_extraction", symbolic_fn),
        ("learned_only", learned_only_fn),
    ]
    base_systems = [
        ("llm_judge_zeroshot", judge_zs_fn),
        ("llm_judge_cot", judge_cot_fn),
        ("self_repair_detect", self_repair_fn),
    ]

    # ---------- metrics per horizon ----------
    rows = []
    for h in all_h:
        for scope in ["ALL"] + DOMAINS:
            for name, fn, records in [
                *[(n_, f_, ext_by_h[h]) for n_, f_ in ext_systems],
                *[
                    (
                        n_,
                        f_,
                        base_by_h[h]
                        if h != 5
                        else [r["baselines"] | {"labels": r["labels"], "domain": r["domain"]} for r in s_test],
                    )
                    for n_, f_ in base_systems
                ],
            ]:
                sel = records if scope == "ALL" else [r for r in records if r["domain"] == scope]
                if not sel:
                    continue
                m = evaluate_system(sel, fn, name)
                row = {
                    "horizon": h,
                    "domain": scope,
                    "system": name,
                    "n": m["n"],
                    "precision": round(m["overall"]["precision"], 3),
                    "recall": round(m["overall"]["recall"], 3),
                    "f1": round(m["overall"]["f1"], 3),
                }
                for ft, dd in m["per_flaw_type"].items():
                    row[f"recall_{ft}"] = None if dd["recall"] is None else round(dd["recall"], 3)
                    row[f"support_{ft}"] = dd["support"]
                rows.append(row)

    with (out / "horizon_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- per-horizon diagnostics ----------
    diag = {}
    for h in all_h:
        base_records = base_by_h[h]
        ext_records = ext_by_h[h]
        flawed = [r for r in base_records if not r["labels"]["overall_valid"]]
        zs_tok = [r.get("judge_zeroshot_output_tokens") for r in base_records]
        cot_tok = [r.get("judge_cot_output_tokens") for r in base_records]
        zs_tok = [t for t in zs_tok if t]
        cot_tok = [t for t in cot_tok if t]
        faithful = [make_label(r) for r in ext_records]
        # LLM plan length (rule-parsed constrained original)
        plan_lens = []
        if h != 5:
            for d in DOMAINS:
                for rec in _load(HDATA / f"{d}_h{h}_plans.jsonl"):
                    problem = Problem.model_validate(rec["problem"])
                    plan_lens.append(len(parse_plan(rec["raw_llm_plan"], problem.domain).steps))
        diag[str(h)] = {
            "n_judge_records": len(base_records),
            "n_extraction_records": len(ext_records),
            "flaw_rate": round(len(flawed) / max(len(base_records), 1), 3),
            "judge_zs_unparseable": sum(1 for r in base_records if r.get("judge_zeroshot_valid") is None),
            "judge_cot_unparseable": sum(1 for r in base_records if r.get("judge_cot_valid") is None),
            "judge_zs_mean_output_tokens": round(float(np.mean(zs_tok)), 1) if zs_tok else None,
            "judge_cot_mean_output_tokens": round(float(np.mean(cot_tok)), 1) if cot_tok else None,
            "extraction_faithful_rate": round(float(np.mean(faithful)), 4) if faithful else None,
            "mean_llm_plan_len": round(float(np.mean(plan_lens)), 1) if plan_lens else None,
        }
    (out / "horizon_diagnostics.json").write_text(json.dumps(diag, indent=2))

    # ---------- downstream at h10 / h20 ----------
    downstream = {}
    if not args.skip_downstream:
        from verifier.llm import get_client

        client = get_client()
        explanation_fn = lambda r: r["verdict"]["explanation"]
        judge_fn_ext = lambda r: r["baselines"].get("judge_cot_valid") is False
        for h in [10, 20]:
            recs = ext_by_h[h]
            downstream[str(h)] = {
                "no_verifier": run_downstream(recs, None, None),
                "hybrid_gated": run_downstream(
                    recs, hybrid_fn, explanation_fn, client=client, model=args.model
                ),
                "llm_judge_cot_gated": run_downstream(
                    recs, judge_fn_ext, lambda r: "the plan was judged invalid",
                    client=client, model=args.model,
                ),
            }
        (out / "downstream_horizon.json").write_text(json.dumps(downstream, indent=2))

    # ---------- figures ----------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def curve(metric: str, systems: list[str], fname: str, title: str):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in systems:
            xs, ys = [], []
            for h in all_h:
                r = next(
                    (r for r in rows if r["horizon"] == h and r["domain"] == "ALL" and r["system"] == name),
                    None,
                )
                if r is not None:
                    xs.append(h)
                    ys.append(r[metric])
            ax.plot(xs, ys, marker="o", label=name)
        ax.set_xscale("log")
        ax.set_xticks(all_h)
        ax.set_xticklabels([str(h) for h in all_h])
        ax.set_xlabel("plan horizon (steps)")
        ax.set_ylabel(metric)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{fname}.png", dpi=150)
        fig.savefig(out / "figures" / f"{fname}.svg")

    sys_all = ["hybrid", "symbolic_extraction", "llm_judge_zeroshot", "llm_judge_cot", "self_repair_detect"]
    curve("recall", sys_all, "recall_vs_horizon", "Flaw detection recall vs plan horizon (pooled)")
    curve("f1", sys_all, "f1_vs_horizon", "Flaw detection F1 vs plan horizon (pooled)")
    curve("precision", sys_all, "precision_vs_horizon", "Flaw detection precision vs plan horizon (pooled)")

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in [
        ("judge_zs_mean_output_tokens", "judge zero-shot"),
        ("judge_cot_mean_output_tokens", "judge CoT"),
    ]:
        xs = [h for h in all_h if diag[str(h)][key] is not None]
        ys = [diag[str(h)][key] for h in xs]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xticks(all_h)
    ax.set_xticklabels([str(h) for h in all_h])
    ax.set_xlabel("plan horizon (steps)")
    ax.set_ylabel("mean judge output tokens per plan")
    ax.set_title("LLM-judge cost vs plan horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "judge_tokens_vs_horizon.png", dpi=150)
    fig.savefig(out / "figures" / "judge_tokens_vs_horizon.svg")

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [h for h in all_h if diag[str(h)]["extraction_faithful_rate"] is not None]
    ys = [diag[str(h)]["extraction_faithful_rate"] for h in xs]
    ax.plot(xs, ys, marker="o", color="#a33")
    ax.set_xscale("log")
    ax.set_xticks(all_h)
    ax.set_xticklabels([str(h) for h in all_h])
    ax.set_xlabel("plan horizon (steps)")
    ax.set_ylabel("verdict faithful to rule-based reference")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("NL->schema extraction fidelity vs plan horizon")
    fig.tight_layout()
    fig.savefig(out / "figures" / "extraction_fidelity_vs_horizon.png", dpi=150)
    fig.savefig(out / "figures" / "extraction_fidelity_vs_horizon.svg")

    # ---------- summary ----------
    def _all_row(h, name):
        return next(
            (r for r in rows if r["horizon"] == h and r["domain"] == "ALL" and r["system"] == name),
            None,
        )

    summary = {
        "horizons": all_h,
        "trust_threshold": threshold,
        "diagnostics": diag,
        "pooled_f1_by_horizon": {
            name: {str(h): (_all_row(h, name) or {}).get("f1") for h in all_h} for name in sys_all
        },
        "pooled_recall_by_horizon": {
            name: {str(h): (_all_row(h, name) or {}).get("recall") for h in all_h} for name in sys_all
        },
        "downstream": downstream,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
