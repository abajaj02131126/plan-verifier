"""Plan-level wrapper around the step extractor: split a raw NL plan into
candidate step lines and extract each with k-resampled self-consistency.

Wired in as the flagged alternative to the Phase 2 rule-based parser
(scripts/verify_plans.py --parser llm), so "rule-based parse + oracle label"
can be compared against "LLM extraction + oracle label", and the extractor's
uncertainty signals feed the Phase 5 learned model.
"""

from __future__ import annotations

import re
from typing import List

import anthropic
from pydantic import BaseModel, Field

from verifier.extraction.extractor import SelfConsistentExtraction, extract_step_self_consistent
from verifier.llm import DEFAULT_MODEL
from verifier.schema import Domain
from verifier.schema.state import GroundAction


class PlanExtraction(BaseModel):
    steps: List[SelfConsistentExtraction] = Field(default_factory=list)
    step_texts: List[str] = Field(default_factory=list)

    def to_ground_actions(self) -> List[GroundAction]:
        """Ground actions for the valid steps (invalid extractions dropped —
        their absence is itself a signal the labeler/verifier will see)."""
        actions = []
        for sc in self.steps:
            e = sc.extraction
            if e.valid and e.action_type is not None:
                actions.append(GroundAction(schema_name=e.action_type, args=tuple(e.args)))
        return actions

    @property
    def mean_confidence(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.extraction.extractor_confidence for s in self.steps) / len(self.steps)

    @property
    def min_confidence(self) -> float:
        if not self.steps:
            return 0.0
        return min(s.extraction.extractor_confidence for s in self.steps)

    @property
    def mean_agreement(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.agreement_exact for s in self.steps) / len(self.steps)


_STEPLIKE_RE = re.compile(r"^\s*(step\s*\d+|\d+\s*[:.)-])", re.IGNORECASE)


def split_plan_lines(raw_plan: str) -> List[str]:
    """Candidate step lines: numbered/step-prefixed lines, or any line
    containing a call-like 'name(...)' pattern. If NO line matches (prose
    plans — e.g. the paraphrased free-text regime), fall back to treating
    every non-empty line as one step; the extractor's confidence signal
    handles lines that turn out not to be actions."""
    all_nonempty = [line.strip() for line in raw_plan.splitlines() if line.strip()]
    lines = [
        line
        for line in all_nonempty
        if _STEPLIKE_RE.match(line) or re.search(r"[\w-]+\s*\([^)]*\)", line)
    ]
    return lines if lines else all_nonempty


def extract_plan(
    client: anthropic.Anthropic,
    domain: Domain,
    raw_plan: str,
    k: int = 3,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> PlanExtraction:
    from concurrent.futures import ThreadPoolExecutor

    step_texts = split_plan_lines(raw_plan)
    with ThreadPoolExecutor(max_workers=4) as pool:
        steps = list(
            pool.map(
                lambda text: extract_step_self_consistent(
                    client, domain, text, k=k, model=model, temperature=temperature
                ),
                step_texts,
            )
        )
    return PlanExtraction(steps=steps, step_texts=step_texts)
