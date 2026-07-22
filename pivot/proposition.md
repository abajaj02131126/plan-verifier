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

The family below has three propositions, one per regime the data actually
exhibits. Common notation: write $c=\text{call\_cost}(J,H)>0$ for the judge's
per-record dollar cost, $\rho=\text{recall}(J,H)$, and
$\phi=\text{false\_reject\_rate}(\text{sym},H)$. All three are proved by
substitution into the single formula above; none introduces a new one.

### Proposition 1 (judge ties the checker on accuracy — the base case)

**If at horizon $H$ both the checker and a judge tier $J$ have recall $1$ and
false-reject rate $0$, and $\text{call\_cost}(\text{sym},H)=0$ while
$\text{call\_cost}(J,H)=c>0$, then**
$$\text{cost}_\text{sym} \le \text{cost}_J \quad\text{for all } S,r,\lambda\ge 0,\ \text{equality iff } \lambda=0.$$

*Proof.* $\text{cost}_\text{sym}=(1{-}1)S+0\cdot r+0\cdot\lambda=0$ and
$\text{cost}_J=(1{-}1)S+0\cdot r+c\lambda=c\lambda$; so the gap is $c\lambda\ge 0$,
zero iff $\lambda=0$. The $S$ and $r$ terms vanish under the shared premises —
the whole gap is the judge's per-call cost. $\square$

*Premises hold in the data at:* Sonnet-5, $H=20,40,80$ (measured recall $1$,
fr $0$). There the grid is symbolic-optimal in all $3600/3600$ cells for
$\lambda>0$ and ties at $\lambda=0$ (verified). Says nothing about weaker
tiers or shorter horizons.

### Proposition 2 (checker has residual false-rejects, judge accuracy-perfect)

**If at horizon $H$ the checker has recall $1$ but false-reject rate
$\phi>0$ (and call cost $0$), while a judge tier has recall $1$, fr $0$, and
call cost $c>0$, then the judge is expected-cost-optimal iff**
$$\lambda < \frac{\phi\,r}{c},$$
**the checker iff $\lambda>\phi r/c$, with a tie on the boundary; the
threshold is independent of the stakes $S$.**

*Proof.* $\text{cost}_\text{sym}=0\cdot S+\phi r+0=\phi r$ and
$\text{cost}_J=0\cdot S+0+c\lambda=c\lambda$. Then $\text{cost}_J<\text{cost}_\text{sym}
\iff c\lambda<\phi r \iff \lambda<\phi r/c$. Both recalls are $1$, so the $S$
term is absent from both sides — the boundary is a horizontal line in
$\lambda$. $\square$

*Premises hold in the data at:* $H=5$, where LLM-extraction noise gives the
checker $\phi=0.068$ (three hyphenated-arg tool plans) and both Haiku judges
have recall $1$, fr $0$. With $r=1$: $\lambda^\star=\phi/c=27.0$ (zero-shot,
$c=\$2.53\mathrm{e}{-}3$) and $21.4$ (CoT). The zero-shot boundary
$\lambda^\star{=}27$ over the swept $\lambda\in[0.01,100]$ predicts the judge
wins the lower $\approx\!85\%$ of $\lambda$ values at every $S$ — i.e.
$3060/3600$ grid cells — matching the measured H=5 grid exactly (verified).

### Proposition 3 (judge recall below 1 — the original Haiku case)

**If at horizon $H$ the checker has recall $1$ and false-reject rate $0$ (so
call cost aside its cost is $0$), and a judge tier has recall $\rho<1$, then**
$$\text{cost}_\text{sym}=0 \le \text{cost}_J = (1{-}\rho)S+\text{fr}_J\,r+c\lambda$$
**for all $S,r,\lambda\ge 0$, and the domination is witnessed by the
accuracy term alone: at $\lambda=0$ the gap is already
$(1{-}\rho)S+\text{fr}_J r$, strictly positive whenever $S>0$.**

*Proof.* Immediate: $\text{cost}_\text{sym}=0$ and every term of
$\text{cost}_J$ is non-negative, with $(1{-}\rho)S>0$ when $\rho<1,S>0$.
Thus the checker dominates *without needing the call-cost term* — unlike
Prop. 1, where the accuracy terms vanish and only $c\lambda$ separates them. $\square$

*Premises hold in the data at:* Haiku, $H=10,20,40$ (recall
$0.94\!\to\!0.88$), where the $\lambda{=}0$ gap is $2.4$–$5.9$ at $S{=}50,r{=}1$
(verified) — the checker wins on missed-flaw cost before call cost enters,
formalizing the paper's original Haiku-degradation observation.

### Reading the family together

The three propositions partition by which premise the data satisfies, and
each maps to a real measured horizon/tier: Prop. 1 $\to$ Sonnet $H\ge20$
(cost-only separation), Prop. 2 $\to$ $H=5$ (a genuine judge-optimal region,
bounded in $\lambda$), Prop. 3 $\to$ Haiku $H\ge10$ (accuracy-only
separation). None claims universal dominance; each states its premises and is
confirmed against the grid before inclusion.

## Scope limits (stated directly, as required)

- **None of the three is universal.** Each is conditional on its stated
  premises for a specific tier at a specific horizon; together they do **not**
  assert that symbolic checking always dominates judging. Prop. 2 in
  particular *proves a region where a judge is optimal* ($H=5$, low $\lambda$).
- **Every proposition corresponds to at least one real measured cell** (P1:
  Sonnet $H\ge20$; P2: $H=5$; P3: Haiku $H\ge10$) — none is included on
  premises no data point satisfies.
- **The regimes are exclusive on the checker's false-reject rate and the
  judge's recall**, the two quantities that decide which term separates the
  strategies: P1/P3 need $\phi=0$ (checker clean, $H\ge10$), P2 needs
  $\phi>0$ (extraction noise, $H=5$); P1/P2 need $\rho=1$ (Sonnet, or Haiku at
  $H=5$), P3 needs $\rho<1$ (Haiku, $H\ge10$).
- **The $\lambda=0$ tie in Prop. 1 is a real degenerate case**, not hidden.
- **Outside these premises the family is silent** (e.g. a tier that is both
  recall $<1$ *and* faces a checker with $\phi>0$ simultaneously is not one of
  the three — no measured cell exhibits it, so it is not claimed).
