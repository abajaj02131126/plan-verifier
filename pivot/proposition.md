# Proposition (Task 8): conditional expected-cost dominance of the checker

This supplements — does not replace — the empirical 3600-cell Pareto grid
(`pivot/results/figures/pareto_H*.png`). The grid measures which strategy is
expected-cost-optimal across (S, λ); the proposition explains algebraically
*why* the grid looks the way it does wherever its premises hold.

## Setup

The expected-cost model implemented in `pivot/decision_model.py` is, verbatim:

    expected_cost(tier, H, S, r, λ) = (1 − recall(tier,H))·S
                                    + false_reject_rate(tier,H)·r
                                    + call_cost_usd(tier,H)·λ

with S ≥ 0 (missed-flaw stakes), r ≥ 0 (replan cost), λ ≥ 0 (dollar-cost
weight). No other cost formula is introduced for the proof.

## Proposition

**Let H be a horizon at which a judge tier J satisfies recall(J,H) = 1 and
false_reject_rate(J,H) = 0, and at which the symbolic checker satisfies
recall(sym,H) = 1, false_reject_rate(sym,H) = 0, and call_cost(sym,H) = 0
while call_cost(J,H) = c > 0. Then**

    expected_cost(sym, H, S, r, λ) ≤ expected_cost(J, H, S, r, λ)
    for all S, r, λ ≥ 0, with equality iff λ = 0.

## Proof

Substitute the premises into the exact formula:

    expected_cost(sym, H, S, r, λ) = (1−1)·S + 0·r + 0·λ = 0
    expected_cost(J,   H, S, r, λ) = (1−1)·S + 0·r + c·λ = c·λ

Hence expected_cost(J) − expected_cost(sym) = c·λ. Since c > 0 and λ ≥ 0,
this difference is ≥ 0 for all S, r, λ ≥ 0, and equals 0 iff λ = 0. ∎

The missed-flaw term (S) and replan term (r) both vanish under the shared
recall = 1, fr = 0 premises, so S and r drop out entirely: the whole gap is
the judge's per-call dollar cost c·λ. This is exactly why the H=20/40/80
heatmaps are a single symbolic-optimal region for every λ > 0 and a tie at
λ = 0 — the empirical grid and the algebra agree.

## Scope limits (stated directly, as required)

- **Conditional, not universal.** The proposition assumes the empirical
  premises hold for the specific tier at the specific horizon. It does **not**
  assert that symbolic checking always dominates judging.
- **Where the premises hold in our data:** Sonnet-5 at H = 20, 40, and 80
  (measured recall 1.0, fr 0). There the proposition applies and the grid is
  symbolic-optimal for all λ > 0.
- **Where a premise fails, the proposition says nothing — and the grid
  reflects that:**
  - *Weaker tier (Haiku):* recall(Haiku,H) < 1 at every H ≥ 10, so the
    premise recall(J)=1 fails; the missed-flaw term (1−recall)·S is strictly
    positive and the proposition does not apply. (Symbolic still wins there
    empirically, but for the different reason that it has strictly higher
    recall — not covered by this proposition.)
  - *Short horizon (H = 5):* the symbolic strategy has
    false_reject_rate(sym, 5) = 0.068 (LLM-extraction noise on 3
    hyphenated-arg tool plans), so the premise fr(sym)=0 fails. Here the
    proposition does **not** hold, and indeed the H=5 grid shows the cheap
    Haiku zero-shot judge expected-cost-optimal in 3060/3600 cells — the
    algebra correctly predicts its own boundary of applicability.
- **The λ = 0 tie is a real degenerate case**, not hidden: with dollar cost
  ignored, a judge that meets the premises ties the checker exactly.

The proposition therefore earns exactly the claim the data supports —
*conditional* cost + soundness dominance — and no more.
