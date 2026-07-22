"""Translate a random sample of real ALFRED trials and report the honest
translation-success rate + failure reasons, with the oracle cross-check
(labeler AND checker must agree the expert plan is valid) as the gate."""
from __future__ import annotations
import json, random, collections
from pathlib import Path

from pivot.alfworld.adapt import translate_task, TranslationError
from verifier.generation.labeler import label_plan
from verifier.generation.parser import ParsedStep, ParseResult
from verifier.symbolic.checker import verify
from verifier.schema.state import initial_state, simulate, goal_satisfied

CACHE = Path("/Users/aribajaj/.cache/alfworld/json_2.1.1")
N = 80


def sample_trials(n: int, seed: int = 0) -> list[Path]:
    trials = []
    for split in ["valid_seen", "valid_unseen", "train"]:
        d = CACHE / split
        if d.exists():
            trials += [p.parent for p in d.glob("*/*/traj_data.json")]
    rng = random.Random(seed)
    rng.shuffle(trials)
    return trials[:n]


def as_parse_result(plan):
    return ParseResult(steps=[ParsedStep(raw_line=repr(g), action=g) for g in plan])


def main():
    trials = sample_trials(N)
    ok, fails = [], collections.Counter()
    fam_ok = collections.Counter(); fam_total = collections.Counter()
    xcheck_ok = 0
    details = []
    for t in trials:
        fam = t.parent.name.split("-")[0]
        fam_total[fam] += 1
        try:
            problem, gold = translate_task(t)
        except TranslationError as e:
            fails[str(e).split(":")[0][:45]] += 1
            continue
        except Exception as e:
            fails[f"OTHER:{type(e).__name__}:{str(e)[:40]}"] += 1
            continue
        # oracle cross-check: both simulators agree the expert plan is VALID
        try:
            labels = label_plan(problem, as_parse_result(gold))
            verdict = verify(gold, problem)
            state = initial_state(problem)
            final = simulate(problem.domain, state, gold)
            schema_valid = goal_satisfied(problem, final)
        except Exception as e:
            fails[f"SIMERR:{type(e).__name__}:{str(e)[:40]}"] += 1
            continue
        agree = (labels.overall_valid == verdict.overall_valid)
        if labels.overall_valid and verdict.overall_valid and schema_valid and agree:
            ok.append((t, problem, gold))
            fam_ok[fam] += 1
            xcheck_ok += 1
        else:
            fails[f"EXPERT_NOT_VALID(lab={labels.overall_valid},chk={verdict.overall_valid},schema={schema_valid})"] += 1
            if len(details) < 6:
                details.append((t.parent.name, labels.overall_valid, verdict.overall_valid, schema_valid,
                                labels.consistency_violations[:2], [str(u) for u in labels.unmet_goals[:2]]))

    print(f"=== ALFWorld translation report (n={N} random real ALFRED trials) ===")
    print(f"fully translated + oracle-cross-checked VALID: {len(ok)}/{N} ({len(ok)/N:.0%})")
    print(f"oracle agreement (labeler==checker) on translated: {xcheck_ok}/{len(ok) if ok else 0}")
    print("\nby family (ok/total):")
    for fam in sorted(fam_total):
        print(f"  {fam}: {fam_ok[fam]}/{fam_total[fam]}")
    print("\nfailure reasons:")
    for r, c in fails.most_common():
        print(f"  {c:3d}  {r}")
    if details:
        print("\nsample expert-plan-not-valid diagnostics:")
        for d in details:
            print(" ", d)

    # save the translated set for the pipeline
    out = Path("verifier/data/alfworld"); out.mkdir(parents=True, exist_ok=True)
    from scripts.generate_problems import ground_action_to_dict
    with (out / "alfworld_problems.jsonl").open("w") as f:
        for t, problem, gold in ok:
            f.write(json.dumps({"problem": problem.model_dump(),
                                "gold_plan": [ground_action_to_dict(g) for g in gold],
                                "plan_length": len(gold),
                                "source_trial": str(t)}) + "\n")
    print(f"\nwrote {len(ok)} translated problems to {out}/alfworld_problems.jsonl")


if __name__ == "__main__":
    main()
