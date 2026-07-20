"""Rule-based parser for the constrained plan format the generation prompt
requests: one action per line, "Step N: action-name(arg1, arg2)".

This is the fast, free, deterministic ground-truth-proxy parser (Phase 2).
The LLM-based structured extractor (Phase 3) is the general alternative for
less-constrained input; both stay available so they can be compared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from verifier.schema import Domain
from verifier.schema.state import GroundAction

# Matches "Step 3: stack(b1, b2)" and tolerant variants: optional "Step N:",
# optional numbering like "3." or "3)", surrounding whitespace.
_STEP_RE = re.compile(
    r"^\s*(?:step\s*\d+\s*[:.)-]\s*|\d+\s*[:.)-]\s*)?"
    r"([a-zA-Z][\w-]*)\s*\(\s*([^)]*)\s*\)\s*[.,;]?\s*$",
    re.IGNORECASE,
)


@dataclass
class ParsedStep:
    """One line of the LLM's plan, either parsed into an action or not."""

    raw_line: str
    action: Optional[GroundAction] = None
    error: Optional[str] = None


@dataclass
class ParseResult:
    steps: List[ParsedStep] = field(default_factory=list)

    @property
    def actions(self) -> List[GroundAction]:
        return [s.action for s in self.steps if s.action is not None]

    @property
    def errors(self) -> List[str]:
        return [s.error for s in self.steps if s.error is not None]

    @property
    def fully_parsed(self) -> bool:
        return all(s.action is not None for s in self.steps)


def parse_plan(text: str, domain: Domain) -> ParseResult:
    """Parse the LLM's plan text into ground actions, validating action names
    and arity against the domain. Lines that are not plan steps at all (blank
    lines, prose commentary) are skipped; lines that look like steps but fail
    validation are kept as errored steps so the labeler can penalize them.
    """
    result = ParseResult()
    known_actions = {a.name: a for a in domain.action_schemas}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _STEP_RE.match(line)
        if m is None:
            # Only treat it as a failed step if it plausibly tried to be one
            # (mentions "step" or starts with a number); pure prose is skipped.
            if re.match(r"^\s*(step\s*\d+|\d+\s*[:.)-])", line, re.IGNORECASE):
                result.steps.append(ParsedStep(raw_line=raw_line, error=f"unparseable step line: {line!r}"))
            continue

        name = m.group(1).lower()
        args_text = m.group(2).strip()
        args = tuple(a.strip() for a in args_text.split(",") if a.strip()) if args_text else ()

        schema = known_actions.get(name)
        if schema is None:
            result.steps.append(
                ParsedStep(raw_line=raw_line, error=f"unknown action '{name}' (not in domain '{domain.name}')")
            )
            continue
        if len(args) != len(schema.parameters):
            result.steps.append(
                ParsedStep(
                    raw_line=raw_line,
                    error=(
                        f"action '{name}' expects {len(schema.parameters)} args, got {len(args)}"
                    ),
                )
            )
            continue

        result.steps.append(ParsedStep(raw_line=raw_line, action=GroundAction(schema_name=name, args=args)))

    return result
