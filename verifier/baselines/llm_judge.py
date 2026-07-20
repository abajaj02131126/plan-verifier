"""Baseline 1 (spec section 6): pure LLM-judge — ask the same base LLM
"is this plan valid?", zero-shot and CoT variants, parsed into the same
overall_valid boolean as every other system for uniform evaluation.
"""

from __future__ import annotations

import re
from typing import Optional

import anthropic
from pydantic import BaseModel

from verifier.generation.prompts import _describe_goal_literals, _describe_init, _describe_resources
from verifier.generation.prompts import _ACTION_PROSE
from verifier.llm import DEFAULT_MODEL
from verifier.schema import Problem

_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(VALID|INVALID)", re.IGNORECASE)


class JudgeResult(BaseModel):
    system: str
    predicted_valid: Optional[bool]  # None = unparseable judge output
    raw_response: str

    @property
    def reject(self) -> bool:
        """Fail-open on unparseable output (mirrors how a naive deployment
        would treat a judge that didn't answer: let the plan through)."""
        return self.predicted_valid is False


def _judge_prompt(problem: Problem, plan_text: str, cot: bool) -> str:
    domain = problem.domain.name
    parts = [
        "You are a plan validity judge. Below is a planning problem and a candidate plan.",
        f"\nActions:\n{_ACTION_PROSE[domain]}",
        f"\nResource limits:\n{_describe_resources(problem)}",
        f"\nInitial state:\n{_describe_init(problem)}",
        f"\nGoal (ALL must hold at the end):\n{_describe_goal_literals(problem)}",
        f"\nCandidate plan:\n{plan_text}",
        "\nIs this plan valid? A valid plan only uses available actions whose "
        "requirements are satisfied when executed in order, achieves the entire "
        "goal, and never exceeds any resource limit.",
    ]
    if cot:
        parts.append(
            "\nThink step by step: simulate the plan action by action, tracking the "
            "state and resource totals, then check each goal condition. After your "
            "reasoning, end with exactly one line: VERDICT: VALID or VERDICT: INVALID"
        )
    else:
        parts.append("\nRespond with exactly one line: VERDICT: VALID or VERDICT: INVALID")
    return "\n".join(parts)


def parse_judge_verdict(text: str) -> Optional[bool]:
    matches = _VERDICT_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper() == "VALID"


def judge_plan(
    client: anthropic.Anthropic,
    problem: Problem,
    plan_text: str,
    cot: bool = False,
    model: str = DEFAULT_MODEL,
) -> JudgeResult:
    system = "llm_judge_cot" if cot else "llm_judge_zeroshot"
    response = client.messages.create(
        model=model,
        # Both variants get the same generous budget: haiku ignores the
        # zero-shot "exactly one line" instruction and simulates the plan
        # anyway, so a tight budget (64, then 512) truncated before any
        # VERDICT line and made most outputs unparseable (360/360 then
        # 146/360 in dev runs). The zs/CoT contrast is the *prompt* (no
        # think-step-by-step instruction), not the token budget.
        max_tokens=2048,
        temperature=0.0,
        messages=[{"role": "user", "content": _judge_prompt(problem, plan_text, cot)}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    return JudgeResult(system=system, predicted_valid=parse_judge_verdict(raw), raw_response=raw)
