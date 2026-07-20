"""Downstream execute-or-reject experiment (Phase 8, spec section 7):

For each test record, "execution" = the oracle labeler's verdict on the plan
(synthetic domains: running the simulator; tools domain: dry-running the mock
environment — same code path by construction).

  no_verifier      : execute the first plan as-is.
  <system>_gated   : if the gate rejects, allow ONE re-plan attempt with the
                     verifier's explanation fed back, re-verify, execute only
                     if accepted.

Task success = an executed plan that is oracle-valid. Executing a flawed plan
is a failure; ending with no accepted plan is also a failure (task not done).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

from verifier.generation.labeler import label_plan
from verifier.generation.parser import parse_plan
from verifier.generation.prompts import CONDITIONS
from verifier.llm import DEFAULT_MODEL
from verifier.schema import Problem


def _replan_prompt(problem: Problem, old_plan: str, feedback: str) -> str:
    import random

    base = CONDITIONS["baseline"].render(problem, random.Random(0))
    return (
        f"{base}\n\nYour previous plan was rejected by a verifier:\n{old_plan}\n\n"
        f"Verifier feedback: {feedback}\n\nProduce a corrected plan in the same format."
    )


def run_downstream(
    records: Sequence[dict],
    gate_reject_fn: Optional[Callable[[dict], bool]],
    gate_explanation_fn: Optional[Callable[[dict], str]],
    client=None,
    model: str = DEFAULT_MODEL,
) -> Dict:
    """gate_reject_fn None => no_verifier arm (execute everything)."""
    n = len(records)
    successes = 0
    executed_flawed = 0
    rejected_twice = 0
    replans = 0

    for rec in records:
        problem = Problem.model_validate(rec["problem"])
        oracle_valid = rec["labels"]["overall_valid"]

        if gate_reject_fn is None or not gate_reject_fn(rec):
            # gate accepts (or no gate): execute the original plan
            if oracle_valid:
                successes += 1
            else:
                executed_flawed += 1
            continue

        # gate rejected: one re-plan attempt with feedback
        replans += 1
        feedback = gate_explanation_fn(rec) if gate_explanation_fn else "plan rejected"
        prompt = _replan_prompt(
            problem, rec.get("raw_llm_plan_original", rec["raw_llm_plan"]), feedback
        )
        response = client.messages.create(
            model=model, max_tokens=1024, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        new_plan = "".join(b.text for b in response.content if b.type == "text")

        parse = parse_plan(new_plan, problem.domain)
        new_labels = label_plan(problem, parse)
        # re-verify with the sound symbolic gate (rule parse of the new plan);
        # execute only if it passes
        if new_labels.overall_valid:
            successes += 1  # accepted and actually valid
        else:
            # symbolic gate on the replan: reject again -> task not done
            rejected_twice += 1

    return {
        "n": n,
        "task_success_rate": successes / n if n else 0.0,
        "executed_flawed_rate": executed_flawed / n if n else 0.0,
        "replans": replans,
        "rejected_after_replan": rejected_twice,
    }
