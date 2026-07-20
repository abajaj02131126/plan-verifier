"""Learned trust layer: features + calibrated classifier (spec section 4.2)."""

from verifier.learned.features import FEATURE_NAMES, extract_features, make_label
from verifier.learned.model import (
    TrustModel,
    expected_calibration_error,
    split_by_problem,
    train_trust_model,
)

__all__ = [
    "FEATURE_NAMES",
    "extract_features",
    "make_label",
    "TrustModel",
    "expected_calibration_error",
    "split_by_problem",
    "train_trust_model",
]
