"""Task 7 (pivot amendment 2): Sonnet-5 at H=80 — the point beyond H=40.

Tests whether Sonnet-5's accuracy tie with the symbolic checker (recall 1.0,
fr 0 at H=20 and H=40) holds even further out, or finally breaks at H=80.

Sonnet-5 ZERO-SHOT only (no CoT — H=40 established the pattern; CoT@H=80 is
out of this budget), same verified 8192 budget, identical prompts, on the
EXISTING 12-record-per-domain H=80 stratified subset (36 records; the same
one used for Haiku at H=80). No new subset, no other model, no other horizon.

The H=80 subset is 100% flawed (0 valid plans), so false-reject rate is
UNDEFINED at this horizon (there is nothing valid to falsely reject) — we
report recall and parse-failure rate, and mark fr as undefined, for both
Sonnet-5 and (same-subset, from existing data) Haiku.

Output: pivot/results/sonnet_judge_h80.json
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verifier.baselines.llm_judge import judge_plan
from verifier.llm import get_client
from verifier.schema import Problem

SECOND_MODEL = "claude-sonnet-5"
HORIZON = 80
DOM = ["blocksworld", "logistics", "tools"]
HDATA = Path("verifier/data/horizon")
OUT = Path("pivot/results")


def _budget() -> int:
    import ast

    for node in ast.walk(ast.parse(Path("verifier/baselines/llm_judge.py").read_text())):
        if isinstance(node, ast.keyword) and node.arg == "max_tokens" and isinstance(node.value, ast.Constant):
            return node.value.value
    raise RuntimeError("max_tokens literal not found")


def _recall_only(records, pred_map) -> dict:
    flawed = [r for r in records if not r["labels"]["overall_valid"]]
    valid = [r for r in records if r["labels"]["overall_valid"]]
    tp = sum(1 for r in flawed if pred_map[(r["problem_name"], r["condition"])] is False)
    n_unparse = sum(1 for v in pred_map.values() if v is None)
    return {
        "n": len(records),
        "n_flawed": len(flawed),
        "n_valid": len(valid),
        "recall": round(tp / len(flawed), 3) if flawed else None,
        "false_reject_rate": None,  # undefined: 0 valid plans at H=80
        "false_reject_rate_note": "undefined — subset has 0 valid plans",
        "parse_failures": n_unparse,
        "parse_failure_rate": round(n_unparse / len(records), 3),
    }


def main() -> int:
    budget = _budget()
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for d in DOM:
        records += [json.loads(l) for l in (HDATA / f"{d}_h{HORIZON}_plans_nl_sub.jsonl").open()]
    client = get_client()

    def judge(rec):
        problem = Problem.model_validate(rec["problem"])
        res = judge_plan(client, problem, rec["raw_llm_plan"], cot=False, model=SECOND_MODEL)
        return (rec["problem_name"], rec["condition"]), res

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(judge, records))
    sonnet_pred = {k: r.predicted_valid for k, r in results}
    sonnet_tok = [r.output_tokens for _, r in results if r.output_tokens]
    sonnet = _recall_only(records, sonnet_pred)
    sonnet["mean_output_tokens"] = round(sum(sonnet_tok) / len(sonnet_tok), 1)

    # Haiku same-subset from existing baselines (no new calls)
    subset_keys = {(r["problem_name"], r["condition"]) for r in records}
    haiku_pred = {}
    for d in DOM:
        for line in (HDATA / f"{d}_h{HORIZON}_baselines.jsonl").open():
            r = json.loads(line)
            k = (r["problem_name"], r["condition"])
            if k in subset_keys:
                haiku_pred[k] = r["judge_zeroshot_valid"]
    haiku = _recall_only(records, haiku_pred)

    # degradation check across the three Sonnet points
    s20 = json.load(open(OUT / "sonnet_judge_h20.json"))["sonnet5_new"]["llm_judge_zeroshot"]["recall"]
    s40 = json.load(open(OUT / "sonnet_judge_h40.json"))["sonnet5_zeroshot"]["recall"]
    trajectory = {"H20": s20, "H40": s40, "H80": sonnet["recall"]}
    degraded = sonnet["recall"] is not None and sonnet["recall"] < 0.98

    flags = []
    if degraded:
        flags.append(
            f"NEW FINDING: Sonnet-5 zero-shot recall DROPS to {sonnet['recall']} at H=80 "
            f"(was 1.0 at H=20 and H=40). The accuracy tie with the symbolic checker "
            f"finally breaks at 80 steps — report as a genuine degradation point, not as "
            f"consistent with H=20/40."
        )
    else:
        flags.append(
            f"Sonnet-5 zero-shot recall stays {sonnet['recall']} at H=80 (1.0 at H=20/40 too): "
            f"the accuracy tie with the checker holds across every horizon tested (20-80). "
            f"No degradation."
        )

    out = {
        "horizon": HORIZON,
        "budget_used": budget,
        "comparison_basis": "existing 36-record H=80 stratified subset (12/domain), 100% flawed",
        "sonnet5_zeroshot": sonnet,
        "haiku_zeroshot_same_subset": haiku,
        "sonnet_recall_trajectory": trajectory,
        "degradation_at_H80": degraded,
        "honesty_flags": flags,
    }
    (OUT / "sonnet_judge_h80.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    for f in flags:
        print("\nFLAG: " + f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
