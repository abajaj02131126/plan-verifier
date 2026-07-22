"""Task 11 detection experiment on real ALFWorld tasks (API-free).

The synthetic domains inject flaws by PROMPTING an LLM under four conditions.
ALFWorld's object identifiers are machine-encoded (Book_bar__plus_00...) and
its init states have 200+ atoms, which makes faithful LLM plan GENERATION
over readable names unreliable without an aliasing layer that would itself
distort results. So for ALFWorld we inject flaws DETERMINISTICALLY into the
real expert plans (matching the same flaw taxonomy), and run the real
pipeline components — oracle labeler and symbolic checker — with the same
independent-oracle cross-check required for every other domain. This
methodology difference is disclosed, not hidden.

Flaw injections (each yields a genuinely flawed plan of a taxonomy type):
  * goal_incompleteness: drop the final PutObject (target never placed).
  * inconsistency (precondition slip): remove a required OpenObject so a
    later Pickup/Put violates its accessibility precondition; if the plan has
    no open, drop a GotoLocation so a later action violates atLocation.
  * hallucinated_action: insert an unsupported ground action.

Reports, in the same shape as the other domains' detection results: recall
per flaw type, and the oracle cross-check agreement (labeler==checker) over
every record — valid expert plans plus injected-flawed variants.
"""
from __future__ import annotations
import collections
import json
from pathlib import Path

from verifier.generation.labeler import label_plan
from verifier.generation.parser import ParsedStep, ParseResult
from verifier.symbolic.checker import verify
from verifier.schema import Problem
from verifier.schema.state import GroundAction

DATA = Path("verifier/data/alfworld/alfworld_problems.jsonl")


def as_parse_result(plan):
    return ParseResult(steps=[ParsedStep(raw_line=repr(g), action=g) for g in plan])


def _drop_final_put(plan):
    for i in range(len(plan) - 1, -1, -1):
        if plan[i].schema_name == "PutObject":
            return plan[:i] + plan[i + 1:], "goal_incompleteness"
    return None, None


def _remove_open_or_goto(plan):
    for i, g in enumerate(plan):
        if g.schema_name == "OpenObject":
            return plan[:i] + plan[i + 1:], "inconsistency"
    for i, g in enumerate(plan):
        if g.schema_name == "GotoLocation" and i > 0:
            return plan[:i] + plan[i + 1:], "inconsistency"
    return None, None


def _insert_hallucination(plan):
    bad = GroundAction(schema_name="TeleportObject", args=("agent1", "nowhere"))
    return plan[:1] + [bad] + plan[1:], "hallucinated_action"


def main():
    if not DATA.exists():
        print("no alfworld problems; run pivot.alfworld.translate_report first")
        return
    records = [json.loads(l) for l in DATA.open()]
    print(f"loaded {len(records)} translated ALFWorld problems")

    rows = []  # (kind, flaw_type, labeler_valid, checker_valid)
    agree = 0
    total = 0
    per_type_caught = collections.Counter()
    per_type_total = collections.Counter()

    def record(problem, plan, expect_valid, flaw_type):
        nonlocal agree, total
        labels = label_plan(problem, as_parse_result(plan))
        verdict = verify(plan, problem)
        total += 1
        if labels.overall_valid == verdict.overall_valid:
            agree += 1
        rows.append((expect_valid, flaw_type, labels.overall_valid, verdict.overall_valid))
        if not expect_valid:
            per_type_total[flaw_type] += 1
            if not verdict.overall_valid:  # checker caught the flaw
                per_type_caught[flaw_type] += 1

    for rec in records:
        problem = Problem.model_validate(rec["problem"])
        gold = [GroundAction(schema_name=a["action"], args=tuple(a["args"])) for a in rec["gold_plan"]]
        record(problem, gold, True, None)  # valid expert plan
        for inject in (_drop_final_put, _remove_open_or_goto, _insert_hallucination):
            flawed, ftype = inject(gold)
            if flawed is not None:
                record(problem, flawed, False, ftype)

    n_valid = sum(1 for r in rows if r[0])
    n_flawed = sum(1 for r in rows if not r[0])
    # detection: checker rejects flawed (recall), accepts valid (precision proxy)
    tp = sum(1 for r in rows if not r[0] and not r[3])
    fn = sum(1 for r in rows if not r[0] and r[3])
    fp = sum(1 for r in rows if r[0] and not r[3])
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None

    out = {
        "domain": "alfworld",
        "n_problems": len(records),
        "n_records": total,
        "n_valid_expert": n_valid,
        "n_flawed_injected": n_flawed,
        "oracle_cross_check_agreement": f"{agree}/{total}",
        "detection": {"precision": round(precision, 3) if precision else None,
                      "recall": round(recall, 3) if recall else None,
                      "f1": round(f1, 3) if f1 else None},
        "per_flaw_type_recall": {
            ft: {"recall": round(per_type_caught[ft] / per_type_total[ft], 3),
                 "support": per_type_total[ft]}
            for ft in sorted(per_type_total)
        },
        "methodology": "deterministic flaw injection (see module docstring); "
                       "same labeler+checker oracle cross-check as other domains",
    }
    Path("pivot/results").mkdir(parents=True, exist_ok=True)
    Path("pivot/results/alfworld_detection.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
