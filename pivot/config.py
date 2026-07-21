"""Task 2 (pivot): stakes parameter S and decision-model sweep ranges.

IMPORTANT — these are STATED MODELING ASSUMPTIONS set by the authors, NOT
measured, inferred, elicited, or calibrated quantities. No LLM was asked to
rate severity; there is no ground truth for "stakes" in this paper. They are
illustrative anchors chosen a priori to make the decision-theoretic model
concrete, and they were fixed BEFORE seeing the Sonnet-5 result — they are
not tuned to make any crossover or Pareto boundary land favorably.

S = stakes = the cost incurred when a flawed plan is executed (a false
accept / missed flaw), in abstract cost units. Higher S = more
irreversible / expensive failure.

The domain anchors below are illustrative markers on a continuous swept
range; the analysis sweeps S over the whole range and treats the domain
values only as annotated reference points.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Stakes anchors (illustrative, author-set modeling assumptions — NOT data)
# ---------------------------------------------------------------------------
DOMAIN_STAKES: dict[str, float] = {
    "blocksworld": 1.0,   # sandbox / pure simulation — a bad plan costs ~nothing
    "logistics": 10.0,    # budgeted but reversible physical operations
    "tools": 50.0,        # real spend / partially irreversible external side effects
}

# Continuous sweep over stakes, independent of the anchors above.
STAKES_SWEEP = np.logspace(0, np.log10(200.0), 60)  # 1 .. 200, log-spaced

# ---------------------------------------------------------------------------
# Replan cost r: the cost of one unnecessary replan (a false reject). A small
# constant relative to stakes; exposed as a sweepable parameter, not hardcoded
# at the call sites. Default is a stated assumption, not fit to data.
# ---------------------------------------------------------------------------
REPLAN_COST_DEFAULT: float = 1.0
REPLAN_COST_SWEEP = np.array([0.5, 1.0, 2.0, 5.0])

# ---------------------------------------------------------------------------
# lambda: converts measured dollar/token call cost into the same abstract cost
# units as S and r (Task 3 amendment). Sweepable; default is a stated
# assumption. lambda=0 recovers the original (cost-free-judging) model.
# ---------------------------------------------------------------------------
LAMBDA_DEFAULT: float = 1.0
LAMBDA_SWEEP = np.logspace(-2, 2, 60)  # 0.01 .. 100, log-spaced

# ---------------------------------------------------------------------------
# Measured per-1k-output-token dollar prices (Anthropic list prices, USD),
# used to turn measured output-token counts into a real billed call cost in
# Task 3. These are published prices, not estimates of our own spend, and are
# the only externally-sourced numbers here.
#   claude-haiku-4-5 : $5.00 / 1M output tokens
#   claude-sonnet-5  : $15.00 / 1M output tokens (list; intro pricing lower)
# Input-token cost is omitted: prompts are identical across tiers at a given
# horizon, so it cancels in tier-vs-tier comparison; documented, not hidden.
# ---------------------------------------------------------------------------
OUTPUT_PRICE_PER_TOKEN_USD: dict[str, float] = {
    "claude-haiku-4-5": 5.00 / 1_000_000,
    "claude-sonnet-5": 15.00 / 1_000_000,
    "symbolic": 0.0,  # no API call at decision time
}
