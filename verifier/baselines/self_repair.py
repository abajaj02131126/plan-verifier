"""Baseline 4 (spec section 6): Reflexion-style single-pass self-repair —
re-prompt the LLM with its own plan, ask it to find and fix any errors, then
check whether the repaired plan is actually valid per the oracle labeler.
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel

from verifier.generation.labeler import PlanLabels, label_plan
from verifier.generation.parser import parse_plan
from verifier.generation.prompts import (
    _ACTION_PROSE,
    _FORMAT_INSTRUCTIONS,
    _describe_goal_literals,
    _describe_init,
    _describe_resources,
)
from verifier.llm import DEFAULT_MODEL
from verifier.schema import Problem


class SelfRepairResult(BaseModel):
    system: str = "self_repair"
    repaired_plan: str
    repaired_labels: PlanLabels
    predicted_valid: bool  # what execution of the REPAIRED plan would yield

    @property
    def repaired_is_valid(self) -> bool:
        return self.repaired_labels.overall_valid


def _repair_prompt(problem: Problem, plan_text: str) -> str:
    domain = problem.domain.name
    return "\n".join(
        [
            "You previously produced the candidate plan below for this problem. "
            "Carefully critique your plan: check each action's requirements in "
            "sequence, whether the whole goal is achieved, and the resource limits. "
            "Then output a corrected plan (or the same plan if you find no errors).",
            f"\nActions:\n{_ACTION_PROSE[domain]}",
            f"\nResource limits:\n{_describe_resources(problem)}",
            f"\nInitial state:\n{_describe_init(problem)}",
            f"\nGoal (ALL must hold at the end):\n{_describe_goal_literals(problem)}",
            f"\nYour previous plan:\n{plan_text}",
            "\nFirst write a brief critique, then output the final plan after a "
            f"line reading exactly 'FINAL PLAN:'.\n{_FORMAT_INSTRUCTIONS}",
        ]
    )


def self_repair(
    client: anthropic.Anthropic,
    problem: Problem,
    plan_text: str,
    model: str = DEFAULT_MODEL,
) -> SelfRepairResult:
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        temperature=0.0,
        messages=[{"role": "user", "content": _repair_prompt(problem, plan_text)}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    repaired = raw.split("FINAL PLAN:", 1)[1] if "FINAL PLAN:" in raw else raw
    labels = label_plan(problem, parse_plan(repaired, problem.domain))
    return SelfRepairResult(
        repaired_plan=repaired.strip(),
        repaired_labels=labels,
        predicted_valid=labels.overall_valid,
    )
