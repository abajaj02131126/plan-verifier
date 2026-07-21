# Limitations

Honest caveats, from PROJECT_SPEC.md section 10 plus everything discovered
during the build. Anything here that materially affects a paper claim is also
flagged at the point of use in PROGRESS.md.

## Dataset & failure injection

- **Sample sizes are dev-scale.** 30 problems x 4 conditions per domain
  (360 plan records total; test split ~70 records after problem-level
  splitting). Every headline number needs the full-scale run (`make
  reproduce FULL=1`) plus ideally multiple seeds before submission.
- **The trust-label negative class exists only in the prose regime.** With
  the constrained "Step N: action(args)" output format, the LLM extractor's
  verdict agreed with the rule-based reference on 240/240 records — the
  NL->schema translation was lossless and the learned layer had nothing to
  model. The pipeline therefore verifies a PROSE PARAPHRASE of each plan
  (scripts/paraphrase_plans.py) while ground truth stays anchored to the
  original constrained text. This mirrors real free-text plans, but it means
  the trust layer's value is demonstrated in the prose regime specifically;
  in deployments where plans can be forced into rigidly parseable formats, a
  rule-based parser + symbolic checking alone may genuinely suffice.
- **The paraphrase step can itself introduce label noise**: if the
  paraphrase changes plan content, the oracle label (computed from the
  original) no longer describes the text the production systems see. All
  compared systems consume the same paraphrased text, so comparisons remain
  like-for-like, but absolute flaw rates in the prose regime inherit this
  noise.
- **resource_blind failed to raise overruns on logistics** (5 baseline -> 4
  induced): logistics plans are short enough that cost-ignorant plans still
  fit under the 1.25x-tightened caps. Needs a tighter slack or longer-horizon
  problems for that cell of the taxonomy.
- **The distractor condition is weak** (0 -> 2 hallucinated actions per 30
  records per domain with haiku): current models mostly resist obviously
  invalid actions listed in the prompt. Realistic hallucination pressure may
  require distractors that shadow real actions more closely.
- Failure modes are induced by prompt manipulation on one model
  (claude-haiku-4-5 by default). The naturally-occurring failure
  distribution of stronger models may differ; the pilot-calibration
  comparison should be re-run per plan-generation model.

## Long-horizon experiment design

- **Long-horizon gold plans are valid but NOT optimal.** The constructive
  planners (`verifier/domains/horizon.py`) tear down and rebuild
  (blocksworld) or route sequentially (logistics/tools), so "horizon H" is
  a band on the REFERENCE plan length; LLM plans for the same problems may
  legitimately be shorter. Consequently the 1.25x resource-cap tightening
  is computed against a wasteful reference, leaving looser caps than at
  short horizons — resource-infeasibility is expected to be rarer in the
  long-horizon flaw mix.
- **The h40/h80 extraction cells are 12-record stratified subsets** (3 per
  condition per domain) — the tiered design keeps k*steps extraction cost
  bounded, at the price of wide error bars on the hybrid/extraction-path
  numbers at those horizons. Judges, self-repair, and the rule-parsed
  symbolic check cover all 40 records per cell.
- **Judge budgets were raised to 8192 output tokens** (and plan
  generation/paraphrase/self-repair budgets raised accordingly) so that
  80-step simulation is not budget-truncated; short- vs long-horizon judge
  comparisons therefore use different budgets than the original dev run.
- **The 100% extraction-fidelity rate at h10-80 should not be read as
  "translation gets more reliable with plan length."** It measures
  agreement between the LLM-extraction verdict and a rule-based reference
  verdict of the SAME plan, not ground-truth extraction accuracy against
  human judgment. The long-horizon constructive plans are mechanically
  repetitive (the same authenticate/search/book or load/drive/unload
  subsequence repeated many times), which plausibly makes the k=3
  self-consistency vote more reproducible even if individual steps are
  still imperfectly extracted — a consistent mis-extraction would still
  register as "faithful" here. Unresolved; noted as a question for
  follow-up rather than a finding.
- **The h10/h20 downstream soundness-gap result (judge-gated arm lets
  0.05/0.042 of plans execute flawed) is not yet demonstrated at h40/h80**,
  where the judge recall trend suggests it would be larger; running it
  there was deferred for API cost.

## Architecture & features

- **No token-level log-probability feature.** The Anthropic API does not
  expose logprobs, so the spec-suggested feature is omitted rather than
  approximated.
- **Precondition similarity is lexical (TF-IDF char n-grams), not neural.**
  Chosen for laptop-friendliness and zero new dependencies; short predicate
  strings make lexical overlap a reasonable proxy, but paraphrased
  precondition mentions ("the crane is free" vs "crane-empty") are
  under-credited.
- **The DSL has no disjunctive preconditions, conditional effects, or typed
  return values.** The tool-use domain threads return values as boolean
  trip-scoped predicates and pins auth scopes via is-<scope> identity
  predicates — adequate for the flaw taxonomy but a simplification of real
  tool APIs (no value passing, no per-call rate windows, no auth expiry).
- **Oracle labeler and symbolic verifier share design semantics** (lenient
  progression) though not code. A semantics-level error made identically in
  both designs would not be caught by their cross-check; the hand-verified
  fixture tests are the main defense. An external VAL check was skipped at
  dev scale (documented in PROGRESS.md Phase 4).

## Dev-scale results that cut against the pitch

- **Both LLM judges score a perfect 1.0 P/R/F1 on the 72-record test
  split** once given an adequate token budget: 3–8-step plans are short
  enough for haiku to simulate flawlessly. The claim "hybrid beats
  LLM-judge on detection" is unsupported at dev scale; the hybrid's
  measured edges are downstream task success (0.931 vs 0.875, fewer replan
  dead-ends thanks to symbolic explanations), zero-API-cost symbolic
  decisions, determinism, and machine-readable explanations. Whether
  detection separates at longer horizons is an open full-scale question.
- **The trust layer never fires at dev scale.** The val-picked threshold is
  0.0 (hybrid ≡ symbolic-only); the test split contains zero
  symbolically-clean-but-flawed records; only 13/360 pooled records have
  unfaithful verdicts. The trust score does correctly rank the 3
  false-reject mistranslations lowest (0.74–0.82) — but the fusion rule
  only lets trust ADD rejects, so it cannot rescue a symbolic reject caused
  by a bad extraction. A bidirectional rule (low-trust reject → re-extract
  or fall back to the rule parser) is the obvious next design, untested
  here.

## Evaluation

- **Downstream experiment simplification:** the re-plan arm gates the second
  plan with the sound rule-based/oracle check (constrained format), so
  "accepted replan" implies "valid replan" by construction; the interesting
  quantity is how often the replan succeeds at all, plus the cost of
  executing flawed plans in the ungated arm.
- **The "zero-shot" judge is zero-shot in prompt only.** haiku ignores the
  "respond with exactly one line" instruction and simulates the plan
  step-by-step regardless, so both judge variants get the same 2048-token
  budget (tight budgets — 64, then 512 — truncated before the VERDICT line
  and made 100% / 41% of outputs unparseable in dev runs). The measured
  zs-vs-CoT gap therefore reflects the presence of an explicit
  reasoning+verdict-format instruction, not reasoning vs no reasoning.
- **Self-repair's detection metric is indirect** (did the action sequence
  change), which conflates cosmetic rewrites with true flaw detection; its
  repair-quality numbers (flawed->fixed, valid->broken) are the more
  meaningful lens.
- **Single judge/extractor model family.** All LLM roles (planner,
  paraphraser, extractor, judge, repairer) default to claude-haiku-4-5;
  cross-model generality (e.g. a stronger judge) is untested in this run.
- Latency/cost comparisons report API call counts and local decision time,
  not end-to-end wall-clock under production batching/caching.
