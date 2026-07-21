"""Task 1b (pivot amendment): second Sonnet-5 data point at H=40.

After Sonnet-5 showed NO horizon degradation at H=20 (recall 1.0 vs Haiku
0.914), the amendment asks for a second point before the decision model:
Sonnet-5 ZERO-SHOT at H=40, on the existing 12-record-per-domain stratified
subset already used for the Haiku extraction path at that horizon (36
records total). Same prompts, same 8192-token budget, no changes.

For an apples-to-apples comparison we also compute HAIKU's zero-shot P/R/F1
on the exact same 36 records, read from the existing baselines files (no new
Haiku calls). The Table-2 Haiku H=40 number is over all 120 records; the
same-subset number is the fair head-to-head and is reported as primary, with
the full-120 number alongside for context.

Reported plainly whatever it shows. Not averaged with H=20; both points
stand. Only H=40, only zero-shot, only Sonnet-5 (no extra horizons/models).

Output: pivot/results/sonnet_judge_h40.json
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verifier.baselines.llm_judge import judge_plan
from verifier.eval import evaluate_system
from verifier.llm import get_client
from verifier.schema import Problem

SECOND_MODEL = "claude-sonnet-5"
HORIZON = 40
DOMAINS = ["blocksworld", "logistics", "tools"]
HDATA = Path("verifier/data/horizon")
OUT = Path("pivot/results")


def _budget() -> int:
    import ast

    for node in ast.walk(ast.parse(Path("verifier/baselines/llm_judge.py").read_text())):
        if isinstance(node, ast.keyword) and node.arg == "max_tokens":
            if isinstance(node.value, ast.Constant):
                return node.value.value
    raise RuntimeError("max_tokens literal not found")


def _load_subset() -> list[dict]:
    recs = []
    for d in DOMAINS:
        with (HDATA / f"{d}_h{HORIZON}_plans_nl_sub.jsonl").open() as f:
            recs.extend(json.loads(line) for line in f)
    return recs


def _prf(records: list[dict], pred_valid_map: dict) -> dict:
    recs = []
    for r in records:
        rr = dict(r)
        rr["_pred_valid"] = pred_valid_map[(r["problem_name"], r["condition"])]
        recs.append(rr)
    m = evaluate_system(recs, lambda rr: rr["_pred_valid"] is False, "judge")
    n_unparse = sum(1 for v in pred_valid_map.values() if v is None)
    return {
        "n": m["n"],
        "precision": round(m["overall"]["precision"], 3),
        "recall": round(m["overall"]["recall"], 3),
        "f1": round(m["overall"]["f1"], 3),
        "parse_failures": n_unparse,
    }


def _haiku_same_subset(subset_keys: set) -> dict:
    """Haiku zero-shot verdicts on the exact 36 subset records, from existing
    baselines files. No new API calls."""
    pred = {}
    toks = []
    for d in DOMAINS:
        for line in (HDATA / f"{d}_h{HORIZON}_baselines.jsonl").open():
            r = json.loads(line)
            key = (r["problem_name"], r["condition"])
            if key in subset_keys:
                pred[key] = r["judge_zeroshot_valid"]
                if r.get("judge_zeroshot_output_tokens"):
                    toks.append(r["judge_zeroshot_output_tokens"])
    return pred, toks


def main() -> int:
    budget = _budget()
    print(f"budget = {budget}; Sonnet-5 zero-shot only, H=40 subset")
    OUT.mkdir(parents=True, exist_ok=True)

    records = _load_subset()
    subset_keys = {(r["problem_name"], r["condition"]) for r in records}
    client = get_client()

    def judge(rec):
        problem = Problem.model_validate(rec["problem"])
        res = judge_plan(client, problem, rec["raw_llm_plan"], cot=False, model=SECOND_MODEL)
        return (rec["problem_name"], rec["condition"]), res

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(judge, records))

    sonnet_pred = {k: r.predicted_valid for k, r in results}
    sonnet_tok = [r.output_tokens for _, r in results if r.output_tokens]
    haiku_pred, haiku_tok = _haiku_same_subset(subset_keys)

    sonnet_metrics = _prf(records, sonnet_pred)
    sonnet_metrics["mean_output_tokens"] = round(sum(sonnet_tok) / len(sonnet_tok), 1)
    haiku_metrics = _prf(records, haiku_pred)
    haiku_metrics["mean_output_tokens"] = round(sum(haiku_tok) / len(haiku_tok), 1) if haiku_tok else None

    # full-120 Haiku H=40 zero-shot from Table 2, for context
    import csv

    haiku_full = None
    with open("results/horizon/horizon_metrics.csv") as f:
        for r in csv.DictReader(f):
            if r["horizon"] == "40" and r["domain"] == "ALL" and r["system"] == "llm_judge_zeroshot":
                haiku_full = {"n": int(r["n"]), "precision": float(r["precision"]),
                              "recall": float(r["recall"]), "f1": float(r["f1"])}

    flags = []
    if sonnet_metrics["recall"] >= 0.98:
        flags.append(
            f"Sonnet-5 zero-shot recall at H=40 is {sonnet_metrics['recall']} (near-perfect) "
            f"on the same 36 records where Haiku scores {haiku_metrics['recall']}. Combined "
            f"with the H=20 point (Sonnet 1.0), Sonnet-5 shows NO horizon degradation at "
            f"either tested point — the degradation is Haiku-specific (model-strength), not "
            f"a horizon-structural property. First-class finding; do not soften."
        )

    out = {
        "horizon": HORIZON,
        "budget_used": budget,
        "comparison_basis": "same 36-record stratified subset (12/domain)",
        "sonnet5_zeroshot": sonnet_metrics,
        "haiku_zeroshot_same_subset": haiku_metrics,
        "haiku_zeroshot_full_120_table2_context": haiku_full,
        "recall_delta_sonnet_minus_haiku_same_subset": round(
            sonnet_metrics["recall"] - haiku_metrics["recall"], 3
        ),
        "honesty_flags": flags,
    }
    (OUT / "sonnet_judge_h40.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    for fl in flags:
        print("\nFLAG: " + fl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
