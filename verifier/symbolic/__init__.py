"""Deterministic symbolic verifier — the sound, explainable hard gate."""

from verifier.symbolic.checker import (
    ResourceViolationDetail,
    Verdict,
    consistency_check,
    goal_completeness_check,
    resource_feasibility_check,
    verify,
)

__all__ = [
    "ResourceViolationDetail",
    "Verdict",
    "consistency_check",
    "goal_completeness_check",
    "resource_feasibility_check",
    "verify",
]
