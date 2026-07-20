"""LLM plan-generation harness: Problem -> prompt (per condition) -> Anthropic
API -> raw NL plan -> rule-based parse -> oracle labels.

Cost note: defaults to claude-haiku-4-5 (see verifier/llm.py) because the full
pipeline calls this thousands of times; model is configurable per call.
"""

from __future__ import annotations

import random
import time
from typing import Optional

import anthropic
from pydantic import BaseModel, Field

from verifier.generation.labeler import PlanLabels, label_plan
from verifier.generation.parser import ParseResult, parse_plan
from verifier.generation.prompts import CONDITIONS
from verifier.llm import DEFAULT_MODEL, get_client
from verifier.schema import Problem


class PlanRecord(BaseModel):
    """One (problem, condition) -> candidate plan record, JSONL-serializable."""

    problem_name: str
    domain: str
    condition: str
    model: str
    problem: dict
    gold_plan: list
    raw_llm_plan: str
    parsed_actions: list = Field(default_factory=list)
    parse_errors: list = Field(default_factory=list)
    labels: PlanLabels


def tighten_resource_caps(problem: Problem, gold_plan: list[dict], slack: float) -> Problem:
    """Return a copy of ``problem`` whose resource caps/initials are tightened
    to (gold plan consumption * slack), rounded up to the next integer.

    Rationale (design decision, see PROGRESS.md): the generator's default caps
    are generous enough that nearly no plan can overrun them, which would
    leave the resource-infeasibility failure mode unrepresented in the
    dataset. Tightening to slightly above the gold plan's actual consumption
    keeps every optimal plan feasible while making wasteful plans infeasible.
    """
    import math

    consumption: dict[str, float] = {}
    domain = problem.domain
    for step in gold_plan:
        schema = domain.action_by_name(step["action"])
        for res, delta in schema.resource_deltas.items():
            consumption[res] = consumption.get(res, 0.0) - delta  # deltas are negative

    dump = problem.model_dump()
    for dim in dump["domain"]["resource_dimensions"]:
        used = consumption.get(dim["name"], 0.0)
        tightened = math.ceil(used * slack) if used > 0 else dim["initial"]
        tightened = max(tightened, 1.0)
        dim["initial"] = float(tightened)
        dim["cap"] = float(tightened)
    return Problem.model_validate(dump)


def request_plan(
    client: anthropic.Anthropic,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    max_attempts: int = 4,
) -> str:
    """One plan-generation call with retry on rate limits / transient errors.

    The SDK already retries 429/5xx twice internally; this outer loop adds a
    slower second layer for long batch runs.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if b.type == "text")
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            last_exc = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"plan generation failed after {max_attempts} attempts") from last_exc


def generate_plan_record(
    client: anthropic.Anthropic,
    problem: Problem,
    gold_plan: list[dict],
    condition: str,
    model: str = DEFAULT_MODEL,
    seed: int = 0,
    temperature: float = 1.0,
) -> PlanRecord:
    """Full pipeline for one (problem, condition): render prompt, call the LLM,
    parse, label against the TRUE problem (full goal, true resource caps)."""
    cond = CONDITIONS[condition]
    rng = random.Random(f"plangen:{problem.name}:{condition}:{seed}")
    prompt = cond.render(problem, rng)
    raw = request_plan(client, prompt, model=model, temperature=temperature)
    parse: ParseResult = parse_plan(raw, problem.domain)
    labels: PlanLabels = label_plan(problem, parse)
    return PlanRecord(
        problem_name=problem.name,
        domain=problem.domain.name,
        condition=condition,
        model=model,
        problem=problem.model_dump(),
        gold_plan=gold_plan,
        raw_llm_plan=raw,
        parsed_actions=[
            {"action": s.action.schema_name, "args": list(s.action.args)} if s.action else {"raw": s.raw_line}
            for s in parse.steps
        ],
        parse_errors=parse.errors,
        labels=labels,
    )


__all__ = [
    "PlanRecord",
    "generate_plan_record",
    "request_plan",
    "tighten_resource_caps",
    "get_client",
]
