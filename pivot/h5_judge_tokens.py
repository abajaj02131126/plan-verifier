"""Task 6 (pivot amendment 2): recover the H=5 judge output-token cost.

The H=5 (short-horizon) judge runs predate output-token logging and saved
only parsed boolean verdicts — no raw text anywhere (checked: baselines
files, task logs). So the free recovery path is unavailable; we re-run ONLY
the missing combos (Haiku zero-shot + CoT at H=5) on the 72-record test
split, same prompts, same verified 8192 budget, to measure tokens. We do NOT
re-run any other horizon, and we verify the re-run verdicts match the saved
ones (temperature 0) so nothing else is perturbed.

Output: pivot/results/h5_judge_tokens.json (mean output tokens per combo,
verdict-match check). Feeds the H=5 row of the Task 3 decision model.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verifier.baselines.llm_judge import judge_plan
from verifier.learned import split_by_problem
from verifier.llm import get_client
from verifier.schema import Problem

S = Path("verifier/data/synthetic")
OUT = Path("pivot/results")
MODEL = "claude-haiku-4-5"


def _load_pooled_with_baselines() -> list[dict]:
    recs = []
    for d in ["blocksworld", "logistics", "tools"]:
        bl = {(x["problem_name"], x["condition"]): x for x in (json.loads(l) for l in open(S / f"{d}_baselines.jsonl"))}
        for line in open(S / f"{d}_verdicts_llm.jsonl"):
            r = json.loads(line)
            r["baselines"] = bl.get((r["problem_name"], r["condition"]), {})
            recs.append(r)
    return recs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = _load_pooled_with_baselines()
    _, _, test = split_by_problem(recs, seed=0)  # 72 records — the H=5 eval split

    # the prose plan text the judges saw lives in the *_plans_nl records
    nl = {}
    for d in ["blocksworld", "logistics", "tools"]:
        for line in open(S / f"{d}_plans_nl.jsonl"):
            r = json.loads(line)
            nl[(r["problem_name"], r["condition"])] = r
    client = get_client()

    def run(rec_cot):
        rec, cot = rec_cot
        src = nl[(rec["problem_name"], rec["condition"])]
        problem = Problem.model_validate(src["problem"])
        res = judge_plan(client, problem, src["raw_llm_plan"], cot=cot, model=MODEL)
        return rec, cot, res

    tasks = [(r, False) for r in test] + [(r, True) for r in test]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, tasks))

    out = {"n_records": len(test), "model": MODEL, "budget": 8192, "combos": {}}
    for cot, name in [(False, "haiku_zeroshot"), (True, "haiku_cot")]:
        rows = [(rec, res) for rec, c, res in results if c == cot]
        toks = [res.output_tokens for _, res in rows if res.output_tokens]
        saved_key = "judge_cot_valid" if cot else "judge_zeroshot_valid"
        matches = sum(1 for rec, res in rows if res.predicted_valid == rec["baselines"].get(saved_key))
        out["combos"][name] = {
            "mean_output_tokens": round(sum(toks) / len(toks), 1),
            "n_tokens_measured": len(toks),
            "verdict_match_vs_saved": f"{matches}/{len(rows)}",
        }
    (OUT / "h5_judge_tokens.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
