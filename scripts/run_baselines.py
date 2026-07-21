"""CLI: run the LLM-call baselines (LLM-judge zero-shot, LLM-judge CoT,
self-repair) over paraphrased plan records, writing one JSONL with all
baseline outputs per record.

The judge and repair baselines see the SAME prose plan text
(raw_llm_plan of the *_plans_nl.jsonl files) that the hybrid system's
extractor consumes, for a like-for-like comparison.

Usage:
    python -m scripts.run_baselines \\
        --plans verifier/data/synthetic/blocksworld_plans_nl.jsonl \\
        --out verifier/data/synthetic/blocksworld_baselines.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verifier.baselines import judge_plan, self_repair
from verifier.generation.parser import parse_plan
from verifier.llm import DEFAULT_MODEL, get_client
from verifier.schema import Problem


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run_baselines", description=__doc__)
    p.add_argument("--plans", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args(argv)

    records = [json.loads(line) for line in args.plans.open()]
    client = get_client()

    def work(rec):
        problem = Problem.model_validate(rec["problem"])
        plan_text = rec["raw_llm_plan"]
        zs = judge_plan(client, problem, plan_text, cot=False, model=args.model)
        cot = judge_plan(client, problem, plan_text, cot=True, model=args.model)
        rep = self_repair(client, problem, plan_text, model=args.model)

        # self-repair's implicit flaw judgment: did it change the action
        # sequence? (compare rule-parses of original constrained text vs the
        # repaired output, both in the constrained format)
        original_actions = parse_plan(rec.get("raw_llm_plan_original", plan_text), problem.domain).actions
        repaired_actions = parse_plan(rep.repaired_plan, problem.domain).actions
        changed = original_actions != repaired_actions

        return {
            "problem_name": rec["problem_name"],
            "domain": rec["domain"],
            "condition": rec["condition"],
            "labels": rec["labels"],
            "judge_zeroshot_valid": zs.predicted_valid,
            "judge_cot_valid": cot.predicted_valid,
            "judge_zeroshot_output_tokens": zs.output_tokens,
            "judge_cot_output_tokens": cot.output_tokens,
            "self_repair_changed_plan": changed,
            "self_repair_repaired_valid": rep.repaired_labels.overall_valid,
            "self_repair_repaired_plan": rep.repaired_plan,
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        out_records = list(pool.map(work, records))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(out_records)} baseline records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
