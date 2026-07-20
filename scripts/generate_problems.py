"""CLI: generate synthetic planning problems + gold plans, write as JSONL.

Usage:
    python -m scripts.generate_problems --domain blocksworld --n 100 --seed 0 \\
        --out verifier/data/synthetic/blocksworld_problems.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verifier.domains import DOMAIN_REGISTRY
from verifier.schema.state import GroundAction


def ground_action_to_dict(ga: GroundAction) -> dict:
    return {"action": ga.schema_name, "args": list(ga.args)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_problems",
        description="Generate synthetic PDDL-style planning problems with gold plans.",
    )
    parser.add_argument("--domain", required=True, choices=sorted(DOMAIN_REGISTRY))
    parser.add_argument("--n", type=int, required=True, help="number of problems to generate")
    parser.add_argument("--seed", type=int, default=0, help="master seed for reproducibility")
    parser.add_argument("--out", required=True, type=Path, help="output .jsonl path")
    parser.add_argument("--min-plan-len", type=int, default=3, help="minimum gold plan length")
    parser.add_argument("--max-plan-len", type=int, default=8, help="maximum gold plan length")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    spec = DOMAIN_REGISTRY[args.domain]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out.open("w") as f:
        for i in range(args.n):
            problem, gold_plan = spec.generate_problem(
                seed=args.seed,
                index=i,
                min_plan_len=args.min_plan_len,
                max_plan_len=args.max_plan_len,
            )
            record = {
                "problem": problem.model_dump(),
                "gold_plan": [ground_action_to_dict(ga) for ga in gold_plan],
                "plan_length": len(gold_plan),
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"wrote {written} {args.domain} problems (seed={args.seed}) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())