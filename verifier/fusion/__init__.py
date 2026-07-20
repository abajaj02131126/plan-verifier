"""Fusion of the symbolic hard gate and the learned trust score (spec 4.3)."""

from verifier.fusion.decision import FusionDecision, decide, prf, sweep_threshold

__all__ = ["FusionDecision", "decide", "prf", "sweep_threshold"]
