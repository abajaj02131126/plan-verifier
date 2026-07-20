"""CLI entry point for the pre-execution plan verifier.

Pipeline stages (generation, extraction, symbolic checking, learned
scoring, fusion, evaluation) will be wired in as subcommands as they
are implemented. For now this is a stub so the package is runnable.
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verifier",
        description="Pre-execution verifier for LLM-generated plans.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="plan-verifier 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("status", help="Print a readiness summary (placeholder).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        print("plan-verifier: skeleton only, no pipeline stages implemented yet.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
