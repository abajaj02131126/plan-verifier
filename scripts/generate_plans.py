"""CLI: generate LLM candidate plans for problems under prompt conditions,
auto-label them, write as JSONL.

Usage:
    python -m scripts.generate_plans \\
        --problems verifier/data/synthetic/blocksworld_problems.jsonl \\
        --conditions baseline,goal_omission,resource_blind,distractor \\
        --n 25 --seed 0 \\
        --out verifier/data/synthetic/blocksworld_plans.jsonl

--n limits how many problems (from the top of the file) are used, keeping API
cost controlled during development; omit it for the full file. Resource caps
are tightened to (gold consumption * --resource-slack) by default so the
resource-infeasibility failure mode is actually representable (see
tighten_resource_caps docstring); pass --no-tighten to keep original caps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verifier.generation.harness import generate_plan_record, tighten_resource_caps
from verifier.generation.prompts import CONDITIONS
from verifier.llm import DEFAULT_MODEL, get_client
from verifier.schema import Problem


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="generate_plans", description=__doc__)
    p.add_argument("--problems", required=True, type=Path, help="input problems .jsonl")
    p.add_argument("--conditions", default="baseline", help="comma-separated condition names")
    p.add_argument("--n", type=int, default=None, help="max problems to use (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--resource-slack",
        type=float,
        default=1.25,
        help="tighten caps to gold-consumption * this factor (default 1.25)",
    )
    p.add_argument("--no-tighten", action="store_true", help="keep original generous caps")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            print(f"unknown condition '{c}'; valid: {sorted(CONDITIONS)}", file=sys.stderr)
            return 2

    records = [json.loads(line) for line in args.problems.open()]
    if args.n is not None:
        records = records[: args.n]

    client = get_client()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out.open("w") as f:
        for i, rec in enumerate(records):
            problem = Problem.model_validate(rec["problem"])
            gold_plan = rec["gold_plan"]
            if not args.no_tighten:
                problem = tighten_resource_caps(problem, gold_plan, slack=args.resource_slack)
            for condition in conditions:
                plan_record = generate_plan_record(
                    client,
                    problem,
                    gold_plan,
                    condition,
                    model=args.model,
                    seed=args.seed,
                    temperature=args.temperature,
                )
                f.write(plan_record.model_dump_json() + "\n")
                written += 1
            if (i + 1) % 5 == 0:
                print(f"  ...{i + 1}/{len(records)} problems done ({written} records)")

    print(f"wrote {written} plan records ({len(records)} problems x {len(conditions)} conditions) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
