# Decision-Theoretic Pivot — Final Summary

All five tasks (plus the amendment's Task 1b) are implemented, each in a
separate `pivot/` module leaving the original pipeline untouched (`make
reproduce` unaffected; 106 offline tests still pass). Every number below is
traceable to an existing result artifact or the new Sonnet-5 runs — nothing
is estimated or interpolated.

## What each task produced

| Task | Module | Output |
|---|---|---|
| 1 | `sonnet_judge_h20.py` | `results/sonnet_judge_h20.json` |
| 1b | `sonnet_judge_h40.py` | `results/sonnet_judge_h40.json` |
| 2 | `config.py` | stakes/replan/lambda constants + prices |
| 3 | `decision_model.py` | `decision_table.json`, `decision_findings.json`, 3 figures |
| 4 | `conformal.py` | `conformal_report.json` |
| 5 | `taxonomy.py` | `guardrail_taxonomy.json` + `.md` |

## Headline results

1. **Sonnet-5 does NOT degrade with horizon** (Task 1/1b). Same prompts,
   same 8192-token budget Haiku got. Zero-shot recall **1.0 at both H=20 and
   H=40**, vs Haiku 0.914 / 0.824 on identical records. 0 parse failures.
   The recall-vs-horizon degradation is **model-strength-specific (Haiku),
   not a horizon-structural property of judging.**

2. **On expected cost, the symbolic checker weakly dominates every judge
   tier at H≥10** (Task 3) — including Sonnet-5. But the win is **zero call
   cost + guaranteed soundness, NOT higher accuracy**: Sonnet-5 ties the
   checker on recall (1.0) and false-rejects (0) at H=20/40. The Pareto
   frontier over (S∈[1,200], λ∈[0.01,100]) is degenerate: symbolic-optimal
   in 3600/3600 grid cells.

3. **The guardrail-collapse taxonomy is the durable, tier-independent
   argument** (Task 5). The checker structurally cannot enter the observed
   silent-failure modes (truncation, fail-open) — no budget to truncate,
   total deterministic verdict, fails closed, zero marginal cost — no matter
   how strong the judge model is.

## Results that came out AGAINST the original thesis (reported, not softened)

- **The core "judges degrade structurally with horizon" premise is
  contradicted by the stronger model.** Sonnet-5 shows no degradation at
  either tested horizon. This is a first-class finding, in the results at
  full visibility, with both Sonnet points plotted alongside Haiku (Task 3
  figures / this summary), not a footnote.
- **The "stakes-pricing moves the crossover earlier" claim does NOT hold**
  (Task 3). At every horizon ≥10 the checker already dominates on accuracy
  alone (recall 1.0, fr 0) *before* stakes enter, so there is no
  judge-favorable region for stakes to shrink. Stakes only matter in the
  H<10 regime where strong judges edge the checker on accuracy.
- **At λ=0 (dollar cost ignored), Sonnet-5 ties the checker** at H=20/40;
  the checker is not strictly optimal there.
- **The conformal trust gate produces 0 beneficial decision flips** on the
  test split (Task 4). At α=0.1 it is vacuous (τ=1.0, flags everything → 41
  false rejects, 0 good flips); at the tightest non-vacuous α≈0.167 it flags
  0 symbolic-passes → changes nothing. Same "trust layer adds nothing at dev
  scale" conclusion as the original paper, now shown with rigorous coverage
  bounds. Binding limit: 5 calibration positives.
- **Coverage guarantees are too wide to be useful** (Task 4). Realized test
  coverage 3/3 but Clopper-Pearson 95% CI = [0.29, 1.0] on 3 points; the
  distribution-free 90% guarantee is unreachable from 5 positives except by
  flagging everything.

## Data gaps flagged, NOT fabricated

- **H=5 judge token cost was never logged** (output-token logging postdates
  the Phase-8 run). This is exactly the one regime where a judge could be
  cost-optimal (H=5, checker has fr>0 from extraction noise), so that cell of
  the Pareto analysis is left as a symbolic boundary in the unknown cost, not
  filled with a guess.
- **The 512-token budget row's per-record P/R** was not separately tabulated
  in the original dev log beyond the 41% unparseable rate; reported as a gap.

## Scope boundaries held (flagged, not silently expanded)

- Task 1b ran **only** Sonnet-5 zero-shot at **only** H=40; no H=10/80, no
  third model (Opus), no short-horizon re-run.
- No new budget sweeps, models, or records for Task 5.
- No LLM elicitation of stakes (Task 2); no synthetic inflation of the
  13-example calibration set (Task 4).

## Follow-ups worth doing (deliberately NOT done here)

- Re-run H=5 judges logging output tokens, to close the one open cell of the
  decision model.
- Sonnet-5 at H=80 (and a third tier) to test whether *any* judge eventually
  degrades, or whether the checker's accuracy tie holds arbitrarily far out.
- Expand the unfaithful-translation calibration set (was scoped out) so the
  conformal gate can reach a non-vacuous 90% guarantee.
- Empirically test adversarial prompt-injection collapse (claimed for the
  checker on structural grounds only, untested here).

## One deviation from the written brief, by necessity

The brief guessed the judge budget was "likely 2048"; it is actually **8192**
(raised during the earlier horizon experiment, which produced the H=20/40
data). The binding rule — "give Sonnet the same budget Haiku got" — required
8192, so that is what was used, and the script asserts the value it reads
matches, so an unequal-budget confound cannot silently return. Sonnet-5 also
deprecates the `temperature` parameter, so `judge_plan` now omits it for the
newest tiers via a backward-compatible guard; the Haiku path is byte-identical.
