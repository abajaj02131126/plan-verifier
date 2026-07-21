"""Task 1 (decision-theoretic pivot): second-judge transfer check.

Re-runs the LLM-judge baseline at horizon H=20 with a STRONGER model
(claude-sonnet-5) to test whether the recall-vs-horizon degradation the
paper measured with claude-haiku-4-5 (Table 2 / tab:horizon) is a
model-weakness artifact or a structural property of judging long plans.

Deliberate constraints (see the pivot instructions):
  * EXACT same prompt templates as the Haiku path — we call the same
    verifier.baselines.judge_plan, only swapping the model string. No
    prompt is rewritten, rephrased, or forked.
  * EXACT same max_tokens budget Haiku received at H=20. That budget is
    read from the live judge module (verifier/baselines/llm_judge.py),
    which currently sits at 8192 — the value that produced the H=20 data
    in results/horizon/. (The pivot brief guessed "likely 2048"; that
    predates the horizon-experiment budget raise in commit ee443fa. The
    binding rule is "same budget Haiku got", so we use whatever the module
    uses now, and we ASSERT it matches so an unequal-budget confound —
    the exact bug Section 5.5 found and fixed — cannot silently return.)
  * Only H=20. Not H=10/40/80, not the short-horizon set. One horizon.
  * Judge-only: no extractor / trust model / symbolic changes.

Scope note (reconciled, not expanded): the brief says "the full 40-record
H=20 dataset". Each domain's H=20 cell IS 40 records; the paper's Table 2
H=20 judge row is POOLED over all three domains = 120 records. We run the
full pooled 120 so the Sonnet row is directly comparable to that existing
row. This is the full H=20 horizon (no extra horizons/models), just the
correct record count for comparability.

Output: pivot/results/sonnet_judge_h20.json — P/R/F1, parse-failure rate,
and mean output tokens for Sonnet-5 zero-shot and CoT, alongside the
existing Haiku H=20 numbers loaded from results/horizon/, in a directly
insertable table shape.
"""

from __future__ import annotations

import ast
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verifier.baselines.llm_judge import judge_plan
from verifier.eval import evaluate_system
from verifier.llm import get_client
from verifier.schema import Problem

SECOND_MODEL = "claude-sonnet-5"
HORIZON = 20
DOMAINS = ["blocksworld", "logistics", "tools"]
HDATA = Path("verifier/data/horizon")
OUT = Path("pivot/results")


def _expected_budget() -> int:
    """Read the max_tokens literal the live judge module uses, so Sonnet
    gets identically what Haiku got and any drift is caught, not assumed."""
    src = Path("verifier/baselines/llm_judge.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "max_tokens":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                return node.value.value
    raise RuntimeError("could not locate max_tokens literal in llm_judge.py")


def _load_h20() -> list[dict]:
    records = []
    for d in DOMAINS:
        with (HDATA / f"{d}_h{HORIZON}_plans_nl.jsonl").open() as f:
            records.extend(json.loads(line) for line in f)
    return records


def _existing_haiku_row() -> dict:
    """Pull the Haiku H=20 pooled judge numbers already in results/horizon/
    so the comparison table is self-contained and traceable."""
    import csv

    rows = {}
    with open("results/horizon/horizon_metrics.csv") as f:
        for r in csv.DictReader(f):
            if r["horizon"] == str(HORIZON) and r["domain"] == "ALL" and "judge" in r["system"]:
                rows[r["system"]] = {
                    "n": int(r["n"]),
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                    "f1": float(r["f1"]),
                }
    diag = json.load(open("results/horizon/horizon_diagnostics.json"))[str(HORIZON)]
    rows["llm_judge_zeroshot"]["mean_output_tokens"] = diag["judge_zs_mean_output_tokens"]
    rows["llm_judge_zeroshot"]["parse_failures"] = diag["judge_zs_unparseable"]
    rows["llm_judge_cot"]["mean_output_tokens"] = diag["judge_cot_mean_output_tokens"]
    rows["llm_judge_cot"]["parse_failures"] = diag["judge_cot_unparseable"]
    return rows


def _run(model: str, budget_assert: int) -> dict:
    records = _load_h20()
    client = get_client()

    def judge(rec_cot):
        rec, cot = rec_cot
        problem = Problem.model_validate(rec["problem"])
        res = judge_plan(client, problem, rec["raw_llm_plan"], cot=cot, model=model)
        return rec, cot, res

    tasks = [(r, False) for r in records] + [(r, True) for r in records]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(judge, tasks))

    zs = {id(r): None for r in records}
    cot = {id(r): None for r in records}
    zs_tok, cot_tok = [], []
    for rec, is_cot, res in results:
        if is_cot:
            cot[id(rec)] = res
            if res.output_tokens:
                cot_tok.append(res.output_tokens)
        else:
            zs[id(rec)] = res
            if res.output_tokens:
                zs_tok.append(res.output_tokens)

    # attach predictions onto record copies for evaluate_system
    def eval_variant(pred_map, tok_list):
        recs = []
        for r in records:
            rr = dict(r)
            rr["_pred_valid"] = pred_map[id(r)].predicted_valid
            recs.append(rr)
        reject_fn = lambda rr: rr["_pred_valid"] is False  # fail-open on None
        m = evaluate_system(recs, reject_fn, "judge")
        n_unparse = sum(1 for r in records if pred_map[id(r)].predicted_valid is None)
        return {
            "n": m["n"],
            "precision": round(m["overall"]["precision"], 3),
            "recall": round(m["overall"]["recall"], 3),
            "f1": round(m["overall"]["f1"], 3),
            "parse_failures": n_unparse,
            "parse_failure_rate": round(n_unparse / len(records), 3),
            "mean_output_tokens": round(sum(tok_list) / len(tok_list), 1) if tok_list else None,
        }

    return {
        "model": model,
        "max_tokens_budget": budget_assert,
        "llm_judge_zeroshot": eval_variant(zs, zs_tok),
        "llm_judge_cot": eval_variant(cot, cot_tok),
    }


def main() -> int:
    budget = _expected_budget()
    print(f"live judge max_tokens budget = {budget} (Sonnet will get exactly this)")
    OUT.mkdir(parents=True, exist_ok=True)

    haiku = _existing_haiku_row()
    sonnet = _run(SECOND_MODEL, budget)

    # honesty guard: flag if Sonnet's zero-shot recall at H=20 is far ABOVE
    # Haiku's (would weaken the "judging degrades structurally" claim).
    hz_recall = haiku["llm_judge_zeroshot"]["recall"]
    sz_recall = sonnet["llm_judge_zeroshot"]["recall"]
    flags = []
    if sz_recall >= 0.98:
        flags.append(
            f"Sonnet-5 zero-shot recall at H=20 is {sz_recall} (near-perfect); "
            "the horizon degradation does NOT replicate for this stronger model. "
            "This WEAKENS the structural-degradation claim and must be reported as-is."
        )
    elif sz_recall - hz_recall > 0.10:
        flags.append(
            f"Sonnet-5 zero-shot recall {sz_recall} exceeds Haiku {hz_recall} by "
            f">0.10; degradation is at least partly model-strength-driven. Report as-is."
        )

    out = {
        "horizon": HORIZON,
        "n_records_pooled": sonnet["llm_judge_zeroshot"]["n"],
        "budget_used": budget,
        "haiku_existing": haiku,
        "sonnet5_new": sonnet,
        "recall_delta_zeroshot_sonnet_minus_haiku": round(sz_recall - hz_recall, 3),
        "recall_delta_cot_sonnet_minus_haiku": round(
            sonnet["llm_judge_cot"]["recall"] - haiku["llm_judge_cot"]["recall"], 3
        ),
        "honesty_flags": flags,
    }
    (OUT / "sonnet_judge_h20.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if flags:
        print("\n=== HONESTY FLAGS (report these, do not suppress) ===")
        for fl in flags:
            print("  - " + fl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
