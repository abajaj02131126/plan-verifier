"""Phase 5 tests: deterministic features, model round-trip with probabilities
in [0, 1], leak-free problem-level splitting."""

import numpy as np
import pytest

from verifier.learned import (
    FEATURE_NAMES,
    TrustModel,
    extract_features,
    make_label,
    split_by_problem,
    train_trust_model,
)


def _fake_record(problem_name="p1", valid=True, oracle_valid=True, conf=0.9, agree=1.0, seed=0):
    """A minimal but schema-complete --parser llm verdict record."""
    from verifier.domains.blocksworld import build_domain
    from verifier.schema import Literal, PredicateAtom, Problem, TypedObject

    domain = build_domain(energy_cap=6.0)
    problem = Problem(
        name=problem_name,
        domain=domain,
        objects=[TypedObject(name="b1", type="block"), TypedObject(name="b2", type="block")],
        init=[
            PredicateAtom(predicate="on-table", args=["b1"]),
            PredicateAtom(predicate="on-table", args=["b2"]),
            PredicateAtom(predicate="clear", args=["b1"]),
            PredicateAtom(predicate="clear", args=["b2"]),
            PredicateAtom(predicate="crane-empty", args=[]),
        ],
        goal=[Literal.pos("on", "b1", "b2")],
    )
    return {
        "problem_name": problem_name,
        "domain": "blocksworld",
        "condition": "baseline",
        "problem": problem.model_dump(),
        "raw_llm_plan": "Step 1: pick-up(b1)\nStep 2: stack(b1, b2)",
        "labels": {"overall_valid": oracle_valid},
        "verdict": {
            "is_consistent": valid,
            "is_goal_complete": valid,
            "is_resource_feasible": True,
            "overall_valid": valid,
            "consistency_violations": [] if valid else ["step 1: boom"],
            "unmet_goals": [],
        },
        "extraction": {
            "per_step": [
                {
                    "text": "Step 1: pick-up(b1)",
                    "action_type": "pick-up",
                    "args": ["b1"],
                    "confidence": conf,
                    "agreement_exact": agree,
                    "agreement_action_type": agree,
                    "valid": True,
                    "preconditions_referenced": ["on-table(b1)", "clear(b1)"],
                },
                {
                    "text": "Step 2: stack(b1, b2)",
                    "action_type": "stack",
                    "args": ["b1", "b2"],
                    "confidence": conf - 0.1 * seed % 0.5,
                    "agreement_exact": agree,
                    "agreement_action_type": agree,
                    "valid": True,
                    "preconditions_referenced": ["holding(b1)"],
                },
            ]
        },
    }


def test_features_are_deterministic_and_right_shape():
    rec = _fake_record()
    f1 = extract_features(rec)
    f2 = extract_features(rec)
    assert f1.shape == (len(FEATURE_NAMES),)
    assert np.array_equal(f1, f2)


def test_label_construction():
    assert make_label(_fake_record(valid=True, oracle_valid=True)) == 1
    assert make_label(_fake_record(valid=False, oracle_valid=True)) == 0
    assert make_label(_fake_record(valid=False, oracle_valid=False)) == 1  # both say invalid: faithful


def test_split_by_problem_never_leaks():
    records = [
        _fake_record(problem_name=f"p{i}", seed=j) for i in range(10) for j in range(4)
    ]
    train, val, test = split_by_problem(records, seed=0)
    names = lambda rs: {r["problem_name"] for r in rs}
    assert names(train) & names(val) == set()
    assert names(train) & names(test) == set()
    assert names(val) & names(test) == set()
    assert len(train) + len(val) + len(test) == len(records)


def test_model_trains_saves_loads_and_outputs_probabilities(tmp_path):
    # synthetic training set with signal: low confidence => unfaithful
    records = []
    for i in range(24):
        faithful = i % 2 == 0
        rec = _fake_record(
            problem_name=f"p{i}",
            valid=True,
            oracle_valid=faithful,  # verdict says valid; oracle agrees only when faithful
            conf=0.9 if faithful else 0.2,
            agree=1.0 if faithful else 1 / 3,
        )
        records.append(rec)
    train, val, test = records[:16], records[16:20], records[20:]
    model = train_trust_model(train, val)

    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = TrustModel.load(path)

    probs = loaded.predict_proba(test)
    assert probs.shape == (len(test),)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    # the trained model must have learned the confidence signal
    hi = loaded.predict_proba([_fake_record(conf=0.95, agree=1.0)])[0]
    lo = loaded.predict_proba([_fake_record(conf=0.1, agree=1 / 3)])[0]
    assert hi > lo
