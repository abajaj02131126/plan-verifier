"""CLI: train + calibrate the trust model on --parser llm verdict records,
report test metrics (incl. ECE and the fusion threshold sweep), save model.

Usage:
    python -m scripts.train_trust_model \\
        --verdicts verifier/data/synthetic/blocksworld_verdicts_llm.jsonl \\
                   verifier/data/synthetic/logistics_verdicts_llm.jsonl \\
        --out verifier/data/models/trust_model.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from verifier.fusion import sweep_threshold
from verifier.learned import (
    TrustModel,
    expected_calibration_error,
    make_label,
    split_by_problem,
    train_trust_model,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_trust_model", description=__doc__)
    p.add_argument("--verdicts", required=True, nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", type=Path, default=None, help="optional metrics JSON output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    records: list[dict] = []
    for path in args.verdicts:
        records.extend(json.loads(line) for line in path.open())
    print(f"loaded {len(records)} llm-parser verdict records from {len(args.verdicts)} file(s)")

    labels_all = np.array([make_label(r) for r in records])
    print(
        f"label balance: {labels_all.sum()}/{len(labels_all)} faithful "
        f"({labels_all.mean():.0%}) — label 0 means the production verdict "
        f"disagrees with the rule-based reference"
    )

    train, val, test = split_by_problem(records, seed=args.seed)
    print(f"split by problem: {len(train)} train / {len(val)} val / {len(test)} test records")

    model = train_trust_model(train, val)
    model.save(args.out)
    print(f"saved model to {args.out}")

    probs = model.predict_proba(test)
    y_test = np.array([make_label(r) for r in test])
    ece, bins = expected_calibration_error(probs, y_test)
    acc = float(((probs >= 0.5).astype(int) == y_test).mean())
    print(f"test: n={len(test)} accuracy@0.5={acc:.3f} ECE={ece:.3f}")

    # Fusion sweep on the test split: how does gating on trust change flaw P/R/F1?
    symbolic_valids = [r["verdict"]["overall_valid"] for r in test]
    flawed_true = [not r["labels"]["overall_valid"] for r in test]
    rows = sweep_threshold(symbolic_valids, list(probs), flawed_true)
    print("fusion threshold sweep (positive class = flawed plan):")
    for row in rows:
        if row["threshold"] in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
            print(
                f"  th={row['threshold']:.2f}  P={row['precision']:.3f} "
                f"R={row['recall']:.3f} F1={row['f1']:.3f}"
            )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "n_records": len(records),
                    "label_balance_faithful": float(labels_all.mean()),
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    "test_accuracy_at_0.5": acc,
                    "test_ece": ece,
                    "reliability_bins": bins,
                    "threshold_sweep": rows,
                },
                indent=2,
            )
        )
        print(f"wrote metrics report to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
