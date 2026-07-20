"""LLM-based structured extractor (spec section 4.1): maps one NL plan step
into a validated schema object using the Anthropic API's structured-output
mode (output_config.format json_schema).

This is the general alternative to the Phase 2 rule-based parser: it works on
less-constrained text (and the tool-use domain later), reports its own
confidence, and supports k-resampled self-consistency — whose agreement
signal is a key feature for the Phase 5 learned trust model. The rule-based
parser remains available as the fast, free, high-fidelity reference for the
synthetic domain.

Failure policy: if the model's output does not validate against the domain
(unknown action, wrong arity), retry ONCE with the validation error fed back;
if it still fails, return a null/low-confidence extraction rather than raise.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import anthropic
from pydantic import BaseModel, Field

from verifier.llm import DEFAULT_MODEL
from verifier.schema import Domain

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "description": "The action name, exactly as listed in the available actions.",
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Object names, in the action's parameter order.",
        },
        "preconditions_referenced": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Conditions this step's text states or implies must hold before it, as predicate(args) strings. Empty if none stated.",
        },
        "resource_deltas": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "description": "Leave empty unless the step text explicitly states a resource cost.",
        },
        "confidence": {
            "type": "number",
            "description": "Your confidence (0.0-1.0) that this extraction faithfully captures the step.",
        },
    },
    "required": ["action_type", "args", "preconditions_referenced", "resource_deltas", "confidence"],
    "additionalProperties": False,
}


class StepExtraction(BaseModel):
    """One plan step mapped into the domain schema, plus extractor metadata."""

    action_type: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    preconditions_referenced: List[str] = Field(default_factory=list)
    resource_deltas: Dict[str, float] = Field(default_factory=dict)
    extractor_confidence: float = 0.0
    valid: bool = False
    validation_error: Optional[str] = None


class SelfConsistentExtraction(BaseModel):
    """Modal extraction over k resamples plus the agreement signal."""

    extraction: StepExtraction
    k: int
    agreement_exact: float  # fraction of resamples matching the modal (action_type, args)
    agreement_action_type: float  # looser: fraction agreeing on action_type alone
    all_extractions: List[StepExtraction] = Field(default_factory=list)


def _domain_prompt(domain: Domain, step_text: str, feedback: Optional[str]) -> str:
    action_lines = []
    for a in domain.action_schemas:
        params = ", ".join(f"{p.name}: {p.type}" for p in a.parameters)
        action_lines.append(f"- {a.name}({params})")
    parts = [
        "Extract the structured action from one step of a plan.",
        f"The available actions in domain '{domain.name}' are:",
        "\n".join(action_lines),
        f"\nPlan step text:\n{step_text}",
        "\nIf the step does not correspond to any available action, still fill in "
        "your best guess for action_type and set confidence near 0.",
    ]
    if feedback:
        parts.append(f"\nYour previous extraction was invalid: {feedback}\nCorrect it.")
    return "\n".join(parts)


def _validate(domain: Domain, action_type: str, args: List[str]) -> Optional[str]:
    try:
        schema = domain.action_by_name(action_type)
    except KeyError:
        return f"unknown action '{action_type}' (valid: {[a.name for a in domain.action_schemas]})"
    if len(args) != len(schema.parameters):
        return f"action '{action_type}' expects {len(schema.parameters)} args, got {len(args)}"
    return None


def _call_once(
    client: anthropic.Anthropic,
    domain: Domain,
    step_text: str,
    model: str,
    temperature: float,
    feedback: Optional[str],
) -> Tuple[Optional[dict], Optional[str]]:
    """One structured-output API call. Returns (parsed json, error)."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=temperature,
            messages=[{"role": "user", "content": _domain_prompt(domain, step_text, feedback)}],
            output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text), None
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        return None, f"api error: {e}"
    except (StopIteration, json.JSONDecodeError) as e:
        return None, f"malformed output: {e}"


def extract_step(
    client: anthropic.Anthropic,
    domain: Domain,
    step_text: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> StepExtraction:
    """Extract one step, with one validation-feedback retry, then fallback."""
    feedback: Optional[str] = None
    for _attempt in range(2):
        data, err = _call_once(client, domain, step_text, model, temperature, feedback)
        if data is None:
            feedback = err
            time.sleep(0.5)
            continue
        action_type = str(data.get("action_type", ""))
        args = [str(a) for a in data.get("args", [])]
        validation_error = _validate(domain, action_type, args)
        confidence = float(data.get("confidence", 0.0))
        confidence = min(max(confidence, 0.0), 1.0)
        if validation_error is None:
            return StepExtraction(
                action_type=action_type,
                args=args,
                preconditions_referenced=[str(p) for p in data.get("preconditions_referenced", [])],
                resource_deltas={},
                extractor_confidence=confidence,
                valid=True,
            )
        feedback = validation_error

    # fell through both attempts: null/low-confidence extraction, never a crash
    return StepExtraction(
        action_type=None,
        extractor_confidence=0.0,
        valid=False,
        validation_error=feedback,
    )


def extract_step_self_consistent(
    client: anthropic.Anthropic,
    domain: Domain,
    step_text: str,
    k: int = 3,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> SelfConsistentExtraction:
    """Call the extractor k times at temperature > 0 and record agreement.

    The modal extraction (most common (action_type, args) pair) is returned as
    the answer; agreement_exact / agreement_action_type are stored as features
    for the learned trust model, not discarded.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=k) as pool:
        extractions = list(
            pool.map(
                lambda _: extract_step(client, domain, step_text, model=model, temperature=temperature),
                range(k),
            )
        )
    keys = [(e.action_type, tuple(e.args)) for e in extractions]
    modal_key, modal_count = Counter(keys).most_common(1)[0]
    modal = next(e for e, key in zip(extractions, keys) if key == modal_key)

    action_types = [e.action_type for e in extractions]
    modal_action, action_count = Counter(action_types).most_common(1)[0]

    return SelfConsistentExtraction(
        extraction=modal,
        k=k,
        agreement_exact=modal_count / k,
        agreement_action_type=action_count / k,
        all_extractions=extractions,
    )
