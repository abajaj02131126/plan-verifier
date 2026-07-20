"""CLI: rewrite each plan's constrained-format text as prose, producing the
input that the PRODUCTION verification path (LLM extractor -> symbolic
verifier) actually sees.

Why this exists (documented design decision, see PROGRESS.md Phase 5): with
the constrained "Step N: action(args)" format, the LLM extractor's verdict
agreed with the rule-based reference on 240/240 records — the NL->schema
translation was lossless, so the trust label had no negative class and the
learned layer had nothing to model. Real deployed LLMs emit prose plans; this
step recreates that lossy-translation regime while keeping ground truth
anchored to the original constrained text (the oracle labels in each record
are computed from `raw_llm_plan_original` and are NOT touched here).

Output records: raw_llm_plan <- prose paraphrase (one sentence per line,
same actions/objects/order), raw_llm_plan_original <- the original text.

Usage:
    python -m scripts.paraphrase_plans \\
        --plans verifier/data/synthetic/blocksworld_plans.jsonl \\
        --out verifier/data/synthetic/blocksworld_plans_nl.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verifier.llm import DEFAULT_MODEL, get_client

_PARAPHRASE_PROMPT = """\
Rewrite the step-by-step plan below as natural prose instructions, the way a
human assistant would describe them. Requirements:
- one sentence per line, one sentence per original step, same order
- refer to the same objects by the same names
- do NOT use the "Step N:" format or write action names with parentheses;
  describe each action in words
- do not add, remove, merge, or reorder steps; no commentary

Plan:
{plan}
"""


def paraphrase(client, text: str, model: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.7,
        messages=[{"role": "user", "content": _PARAPHRASE_PROMPT.format(plan=text)}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="paraphrase_plans", description=__doc__)
    p.add_argument("--plans", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args(argv)

    records = [json.loads(line) for line in args.plans.open()]
    client = get_client()

    def work(rec):
        original = rec["raw_llm_plan"]
        if original.strip():
            prose = paraphrase(client, original, args.model)
        else:
            prose = ""
        return {**rec, "raw_llm_plan": prose, "raw_llm_plan_original": original}

    with ThreadPoolExecutor(max_workers=8) as pool:
        out_records = list(pool.map(work, records))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(out_records)} paraphrased plan records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
