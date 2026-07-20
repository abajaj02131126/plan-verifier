"""Baseline 3 (spec section 6): pure learned classifier — predicts plan
validity directly from the Phase 5 feature vector WITH ALL SYMBOLIC-VERDICT
FEATURES REMOVED (no symbolic grounding), trained on the oracle
overall_valid label (not the trust label). Ablation showing what the
symbolic layer adds.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from verifier.learned.features import FEATURE_NAMES, extract_features

# every feature derived from the symbolic verdict is excluded
_SYMBOLIC_FEATURES = {
    "verdict_consistent",
    "verdict_goal_complete",
    "verdict_resource_feasible",
    "n_consistency_violations",
    "n_unmet_goals",
}
_KEEP_IDX = [i for i, name in enumerate(FEATURE_NAMES) if name not in _SYMBOLIC_FEATURES]
LEARNED_ONLY_FEATURES = [FEATURE_NAMES[i] for i in _KEEP_IDX]


def _features_no_symbolic(record: dict) -> np.ndarray:
    return extract_features(record)[_KEEP_IDX]


@dataclass
class LearnedOnlyModel:
    pipeline: Pipeline

    def predict_valid_proba(self, records: Sequence[dict]) -> np.ndarray:
        X = np.stack([_features_no_symbolic(r) for r in records])
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "LearnedOnlyModel":
        with path.open("rb") as f:
            model = pickle.load(f)
        assert isinstance(model, LearnedOnlyModel)
        return model


def train_learned_only(train: List[dict]) -> LearnedOnlyModel:
    X = np.stack([_features_no_symbolic(r) for r in train])
    y = np.array([1 if r["labels"]["overall_valid"] else 0 for r in train])
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X, y)
    return LearnedOnlyModel(pipeline=pipeline)
