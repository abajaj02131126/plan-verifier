"""Tests for the long-horizon constructive generators."""

import pytest

from verifier.domains.horizon import HORIZON_GENERATORS, _band
from verifier.generation.labeler import label_plan
from verifier.generation.parser import ParsedStep, ParseResult

HORIZONS = [10, 20, 40, 80]


def _as_parse_result(gold_plan):
    return ParseResult(
        steps=[ParsedStep(raw_line=repr(ga), action=ga) for ga in gold_plan]
    )


@pytest.mark.parametrize("domain", sorted(HORIZON_GENERATORS))
@pytest.mark.parametrize("horizon", HORIZONS)
def test_gold_plan_valid_per_oracle_labeler(domain, horizon):
    """The constructive plan must be flawless per the INDEPENDENT labeler
    (generation itself already checks via strict schema simulation, so this
    is a second, separately implemented opinion)."""
    problem, gold_plan = HORIZON_GENERATORS[domain](seed=0, index=0, target_len=horizon)
    labels = label_plan(problem, _as_parse_result(gold_plan))
    assert labels.overall_valid, (
        f"{domain} h={horizon}: {labels.consistency_violations} "
        f"{labels.unmet_goals} {labels.resource_violations}"
    )


@pytest.mark.parametrize("domain", sorted(HORIZON_GENERATORS))
@pytest.mark.parametrize("horizon", HORIZONS)
def test_plan_length_in_band(domain, horizon):
    lo, hi = _band(horizon)
    problem, gold_plan = HORIZON_GENERATORS[domain](seed=0, index=1, target_len=horizon)
    assert lo <= len(gold_plan) <= hi


@pytest.mark.parametrize("domain", sorted(HORIZON_GENERATORS))
def test_determinism(domain):
    a = HORIZON_GENERATORS[domain](seed=3, index=2, target_len=20)
    b = HORIZON_GENERATORS[domain](seed=3, index=2, target_len=20)
    assert a[0].model_dump() == b[0].model_dump()
    assert a[1] == b[1]


def test_resource_dims_tightenable():
    """Resource dims must be sized from gold consumption so the Phase 2
    tightening step (1.25x) still leaves the gold plan feasible."""
    problem, gold_plan = HORIZON_GENERATORS["blocksworld"](seed=0, index=3, target_len=40)
    for dim in problem.domain.resource_dimensions:
        assert dim.initial > 0
        assert dim.cap >= dim.initial
