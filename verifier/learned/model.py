"""Learned trust model (spec section 4.2): a small calibrated classifier over
the Phase 5 feature vectors.

- Logistic regression (scikit-learn) — trains in seconds on a laptop.
- Split BY PROBLEM, not by record, so resampled conditions of the same
  problem never leak across train/val/test.
- Calibrated with Platt scaling (sigmoid) on the validation split; isotonic
  needs more data than we have per split.
- Reports ECE and reliability-diagram bins.
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PlattCalibrated:
    """Explicit Platt scaling: a 1-feature logistic regression mapping the
    base model's raw probability to a calibrated one, fit on the val split.
    Implemented directly (rather than CalibratedClassifierCV) because the
    val splits here are small and sklearn's calibrator insists on further
    cross-validation splits; this is transparent and version-proof."""

    def __init__(self, base: Pipeline, calibrator: LogisticRegression):
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.calibrator.predict_proba(raw)

from verifier.learned.features import FEATURE_NAMES, extract_features, make_label


@dataclass
class TrustModel:
    pipeline: "Pipeline | PlattCalibrated"
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def predict_proba(self, records: Sequence[dict]) -> np.ndarray:
        X = np.stack([extract_features(r) for r in records])
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "TrustModel":
        with path.open("rb") as f:
            model = pickle.load(f)
        assert isinstance(model, TrustModel)
        return model


def split_by_problem(
    records: List[dict], seed: int = 0, frac_train: float = 0.6, frac_val: float = 0.2
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Deterministic train/val/test split on problem_name (never on record)."""
    problems = sorted({r["problem_name"] for r in records})
    rng = random.Random(f"trust-split:{seed}")
    rng.shuffle(problems)
    n = len(problems)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)
    train_p = set(problems[:n_train])
    val_p = set(problems[n_train : n_train + n_val])
    train = [r for r in records if r["problem_name"] in train_p]
    val = [r for r in records if r["problem_name"] in val_p]
    test = [r for r in records if r["problem_name"] not in train_p | val_p]
    return train, val, test


def train_trust_model(train: List[dict], val: List[dict]) -> TrustModel:
    X_train = np.stack([extract_features(r) for r in train])
    y_train = np.array([make_label(r) for r in train])
    X_val = np.stack([extract_features(r) for r in val])
    y_val = np.array([make_label(r) for r in val])

    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    base.fit(X_train, y_train)

    # Platt-scale on the held-out val split when it contains both classes;
    # otherwise (tiny/degenerate val) fall back to the uncalibrated model.
    if len(np.unique(y_val)) == 2:
        raw_val = base.predict_proba(X_val)[:, 1].reshape(-1, 1)
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(raw_val, y_val)
        return TrustModel(pipeline=PlattCalibrated(base, calibrator))
    return TrustModel(pipeline=base)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[Dict]]:
    """ECE plus per-bin data for reliability diagrams."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows: List[Dict] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_rows.append({"lo": float(lo), "hi": float(hi), "count": 0, "conf": None, "acc": None})
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += (count / len(probs)) * abs(conf - acc)
        bin_rows.append({"lo": float(lo), "hi": float(hi), "count": count, "conf": conf, "acc": acc})
    return float(ece), bin_rows
