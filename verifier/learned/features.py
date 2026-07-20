"""Feature extraction for the learned trust model (spec section 4.2).

Input: one verdict record from scripts/verify_plans.py --parser llm — raw LLM
plan text, per-step extractions with confidence + self-consistency agreement
(Phase 3), and the symbolic verdict computed from those extractions (Phase 4).
Output: a fixed-size numeric feature vector.

Feature notes / design decisions:
- Embedding similarity between stated and schema-expected preconditions uses
  TF-IDF character-n-gram cosine similarity (scikit-learn, already a dep)
  rather than a neural sentence embedding: laptop-fast, zero new
  dependencies/model downloads, and for short predicate strings like
  "clear(b1)" lexical similarity captures most of the signal. Documented
  here and in PROGRESS.md per the spec's "document whichever you pick".
- Token-level log-probabilities are NOT available from the Anthropic API
  (it does not expose logprobs), so that spec-suggested feature is omitted
  rather than faked. Recorded in LIMITATIONS.md.
- The oracle labels are NEVER features — they are (part of) the target.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from verifier.schema import Problem

FEATURE_NAMES: List[str] = [
    "mean_confidence",
    "min_confidence",
    "mean_agreement_exact",
    "min_agreement_exact",
    "mean_agreement_action_type",
    "n_steps",
    "n_distinct_actions",
    "frac_invalid_steps",
    "n_resource_types_touched",
    "precondition_similarity",
    "raw_plan_chars",
    "verdict_consistent",
    "verdict_goal_complete",
    "verdict_resource_feasible",
    "n_consistency_violations",
    "n_unmet_goals",
    "domain_is_blocksworld",
]


def _precondition_similarity(problem: Problem, per_step: List[dict]) -> float:
    """Mean cosine similarity (TF-IDF char 2-4-grams) between the
    preconditions the extractor says a step references and the schema's
    actual preconditions for the extracted action type. Steps that reference
    no preconditions, or whose action is invalid, contribute 0 similarity
    only if the schema HAS preconditions to state (else they're skipped)."""
    domain = problem.domain
    pairs: List[tuple[str, str]] = []
    for step in per_step:
        action_type = step.get("action_type")
        if not action_type or not step.get("valid"):
            continue
        try:
            schema = domain.action_by_name(action_type)
        except KeyError:
            continue
        expected = " ".join(
            f"{'not ' if lit.negated else ''}{lit.atom.predicate}({', '.join(lit.atom.args)})"
            for lit in schema.preconditions
        )
        stated = " ".join(str(p) for p in step.get("preconditions_referenced", []) or [])
        if not expected:
            continue
        pairs.append((stated, expected))

    if not pairs:
        return 0.0

    sims = []
    for stated, expected in pairs:
        if not stated.strip():
            sims.append(0.0)
            continue
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            m = vec.fit_transform([stated, expected])
            sims.append(float(cosine_similarity(m[0], m[1])[0, 0]))
        except ValueError:
            sims.append(0.0)
    return float(np.mean(sims))


def extract_features(record: Dict) -> np.ndarray:
    """Feature vector for one --parser llm verdict record."""
    problem = Problem.model_validate(record["problem"])
    extraction = record.get("extraction", {}) or {}
    per_step: List[dict] = extraction.get("per_step", []) or []
    verdict = record["verdict"]

    n_steps = len(per_step)
    confidences = [s.get("confidence", 0.0) for s in per_step]
    agreements = [s.get("agreement_exact", 0.0) for s in per_step]
    action_agreements = [s.get("agreement_action_type", 0.0) for s in per_step]
    valid_flags = [bool(s.get("valid")) for s in per_step]
    action_types = {s.get("action_type") for s in per_step if s.get("valid") and s.get("action_type")}

    touched_resources = set()
    for s in per_step:
        if s.get("valid") and s.get("action_type"):
            try:
                schema = problem.domain.action_by_name(s["action_type"])
                touched_resources.update(schema.resource_deltas)
            except KeyError:
                pass

    features = [
        float(np.mean(confidences)) if confidences else 0.0,
        float(np.min(confidences)) if confidences else 0.0,
        float(np.mean(agreements)) if agreements else 0.0,
        float(np.min(agreements)) if agreements else 0.0,
        float(np.mean(action_agreements)) if action_agreements else 0.0,
        float(n_steps),
        float(len(action_types)),
        float(1.0 - (sum(valid_flags) / n_steps)) if n_steps else 1.0,
        float(len(touched_resources)),
        _precondition_similarity(problem, per_step),
        float(len(record.get("raw_llm_plan", ""))),
        1.0 if verdict["is_consistent"] else 0.0,
        1.0 if verdict["is_goal_complete"] else 0.0,
        1.0 if verdict["is_resource_feasible"] else 0.0,
        float(len(verdict.get("consistency_violations", []))),
        float(len(verdict.get("unmet_goals", []))),
        1.0 if record.get("domain") == "blocksworld" else 0.0,
    ]
    assert len(features) == len(FEATURE_NAMES)
    return np.asarray(features, dtype=np.float64)


def make_label(record: Dict) -> int:
    """Trust label construction (spec section 3 item 4, operationalized):

    The trust score predicts whether the PRODUCTION pipeline's symbolic
    verdict (LLM extraction -> symbolic verify) is faithful to what the plan
    actually does. The reference for "what the plan actually does" is the
    rule-based parse + oracle label — higher-fidelity on this synthetic
    domain because the prompt's constrained format was designed for it.

        label = 1  iff  verdict.overall_valid == labels.overall_valid

    i.e. label 0 marks records where trusting the production verdict would
    mean accepting a flawed plan or rejecting a good one purely because the
    NL->schema translation was lossy. This is exactly the residual risk the
    learned layer is supposed to rank (spec 4.3).
    """
    return int(record["verdict"]["overall_valid"] == record["labels"]["overall_valid"])
