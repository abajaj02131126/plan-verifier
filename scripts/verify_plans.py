"""CLI: run the symbolic verifier over plan records, write verdicts as JSONL,
and cross-check verdicts against the Phase 2 oracle labels.

Usage:
    python -m scripts.verify_plans \\
        --plans verifier/data/synthetic/blocksworld_plans.jsonl \\
        --out verifier/data/synthetic/blocksworld_verdicts.jsonl

--parser rule (default) re-parses raw_llm_plan with the rule-based parser;
--parser llm uses the Phase 3 LLM extractor with k-resampled self-consistency
(slower, costs API calls) and records the extractor's uncertainty signals in
the output — those are the Phase 5 learned model's features.

The cross-check (always reported for --parser rule) compares this verifier's
overall_valid against the oracle label's overall_valid record by record; any
mismatch means a bug in one of the two independent implementations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verifier.generation.parser import parse_plan
from verifier.llm import DEFAULT_MODEL
from verifier.schema import Problem
from verifier.symbolic import verify


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verify_plans", description=__doc__)
    p.add_argument("--plans", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--parser", choices=["rule", "llm"], default="rule")
    p.add_argument("--k", type=int, default=3, help="self-consistency resamples (llm parser)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="extractor model (llm parser)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = [json.loads(line) for line in args.plans.open()]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    llm_client = None
    if args.parser == "llm":
        from verifier.llm import get_client

        llm_client = get_client()

    agree = 0
    disagree = []
    written = 0

    with args.out.open("w") as f:
        for i, rec in enumerate(records):
            problem = Problem.model_validate(rec["problem"])

            extraction_meta = {}
            if args.parser == "rule":
                parse = parse_plan(rec["raw_llm_plan"], problem.domain)
                actions, input_errors = parse.actions, parse.errors
            else:
                from verifier.extraction import extract_plan

                px = extract_plan(
                    llm_client, problem.domain, rec["raw_llm_plan"], k=args.k, model=args.model
                )
                actions = px.to_ground_actions()
                input_errors = [
                    f"step {j + 1} ({px.step_texts[j]!r}): {s.extraction.validation_error}"
                    for j, s in enumerate(px.steps)
                    if not s.extraction.valid
                ]
                extraction_meta = {
                    "mean_confidence": px.mean_confidence,
                    "min_confidence": px.min_confidence,
                    "mean_agreement": px.mean_agreement,
                    "per_step": [
                        {
                            "text": text,
                            "action_type": s.extraction.action_type,
                            "args": s.extraction.args,
                            "confidence": s.extraction.extractor_confidence,
                            "agreement_exact": s.agreement_exact,
                            "agreement_action_type": s.agreement_action_type,
                            "valid": s.extraction.valid,
                        }
                        for text, s in zip(px.step_texts, px.steps)
                    ],
                }

            verdict = verify(actions, problem, input_errors=input_errors)

            oracle_valid = rec["labels"]["overall_valid"]
            if args.parser == "rule":
                if verdict.overall_valid == oracle_valid:
                    agree += 1
                else:
                    disagree.append((i, rec["problem_name"], rec["condition"]))

            out_rec = {
                **rec,
                "verifier_parser": args.parser,
                "verdict": verdict.model_dump(),
                "extraction": extraction_meta,
            }
            f.write(json.dumps(out_rec) + "\n")
            written += 1
            if args.parser == "llm" and (i + 1) % 10 == 0:
                print(f"  ...{i + 1}/{len(records)} records extracted+verified")

    print(f"wrote {written} verdict records to {args.out}")
    if args.parser == "rule":
        total = agree + len(disagree)
        print(f"cross-check vs oracle labels: {agree}/{total} agree")
        if disagree:
            print("DISAGREEMENTS (bug in verifier or labeler — investigate):")
            for idx, name, cond in disagree:
                print(f"  record {idx}: {name} [{cond}]")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
