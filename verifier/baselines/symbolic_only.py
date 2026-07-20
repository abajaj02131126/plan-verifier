"""Baseline 2 (spec section 6): pure symbolic checker — the Phase 4 verifier
with the fusion threshold forced to 0.0 so the trust score is ignored. This
already exists implicitly as sweep_threshold's th=0.0 row; here it is a named
system emitting the shared record schema for the Phase 8 results table.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from verifier.fusion import decide


class SystemResult(BaseModel):
    """Shared output schema for every system/baseline (Phase 8 consumes this)."""

    system: str
    predicted_valid: Optional[bool]
    score: Optional[float] = None  # trust/confidence score where applicable

    @property
    def reject(self) -> bool:
        return self.predicted_valid is False


def symbolic_only(verdict_overall_valid: bool) -> SystemResult:
    decision = decide(symbolic_valid=verdict_overall_valid, trust_score=1.0, threshold=0.0)
    return SystemResult(system="symbolic_only", predicted_valid=decision.accept)
