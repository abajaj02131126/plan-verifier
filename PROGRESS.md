# Progress Log

Running log for the pre-execution plan verifier project. One entry per work
session, dated, newest at the bottom. Append only — never overwrite past
entries.

---

## 2026-07-19

**Built:**
- Repo scaffolding at `~/plan-verifier`, git-initialized.
- Full directory skeleton per `PROJECT_SPEC.md`: `verifier/{schema,domains,
  generation,symbolic,extraction,learned,fusion,baselines,eval,data,configs}`,
  `scripts/`, `tests/` (mirroring `verifier/` package structure).
- `PROJECT_SPEC.md` committed as the canonical research design doc.
- Toolchain: `uv` installed (no system Python 3.11+ was available — system
  Python was 3.9.6); `uv python install 3.11` + `.python-version` pin project
  to Python 3.11.15. Documented both `uv` and plain venv+pip setup paths in
  README.md.
- `pyproject.toml`: dependencies `anthropic`, `pydantic`, `numpy`,
  `scikit-learn`, `pyyaml`; dev dependency `pytest`. `verifier` console
  script entry point. `hatchling` build backend.
- `verifier/__main__.py`: minimal argparse CLI stub (`--help`, `--version`,
  a placeholder `status` subcommand). No pipeline logic yet.
- `tests/test_cli.py`: two trivial tests exercising the CLI stub.
- `.gitignore`: standard Python ignores, plus `verifier/data/*` except
  `verifier/data/fixtures/`.
- README.md with setup instructions (uv and venv+pip), directory layout,
  and `ANTHROPIC_API_KEY` env var note (not hardcoded anywhere).

**Tests passing:**
- `uv run pytest` — 2/2 passed.
- `uv run python -m verifier --help` — runs, prints usage.

**Explicitly not done (by design, per task scope):**
- No domain, schema, symbolic-checker, extractor, learned-model, fusion, or
  baseline logic. This session is scaffolding only.

**Next:**
- Design the schema DSL (`verifier/schema/`) for typed predicates,
  resources, and action types.
- Build the synthetic PDDL-style domain generator + a simple gold planner
  (`verifier/domains/`, per spec section 5, primary domain).
- Stand up the LLM plan-generation harness skeleton in `verifier/generation/`
  (no real API calls yet — interface + config first).

---

## 2026-07-20

**Built — synthetic PDDL-style planning foundation (spec section 5, primary domain):**
- `verifier/schema/`: typed pydantic DSL —
  `predicates.py` (`PredicateDefinition`, `PredicateAtom`, `Literal`),
  `actions.py` (`Parameter`, `ActionSchema`: params, preconditions, add/del
  effects, `resource_deltas: dict[str, float]`),
  `resources.py` (`ResourceDimension`: name, initial, cap, floor),
  `domain.py` (`Domain`: cross-validates that every predicate/resource an
  action references is declared, arity matches, and parameter types match),
  `problem.py` (`Problem`: embeds a `Domain` instance + typed `objects` +
  `init` + `goal`, validated against the domain's predicate/type
  declarations), `state.py` (grounding, forward simulation/progression,
  resource floor/cap checking at every step — `PreconditionViolation` /
  `ResourceViolation` raised on the first invalid step of a plan).
- `verifier/domains/blocksworld.py`: classic 4-operator blocksworld
  (pick-up/put-down/stack/unstack) extended with a single `energy` resource
  dimension (crane energy, capped per plan) — 2 energy/stack-unstack,
  1/pick-up-put-down. `generate_problem(seed, index, ...)` builds a random
  valid block arrangement, does a random walk to a reachable state, uses the
  walked state's `on`/`on-table` atoms as the goal, then calls the gold
  planner and retries (new derived seed) until the optimal plan length lands
  in `[min_plan_len, max_plan_len]` (default 3-8).
- `verifier/domains/logistics.py`: 2 cities x 2 locations (1 airport each),
  1 truck/city, 1 plane, 2-3 packages; 6 action schemas (load/unload-truck,
  load/unload-plane, drive-truck, fly-plane) over two resource dimensions,
  `fuel` (consumed by movement, more by flying) and `budget` (consumed by
  every action). Same generate-walk-then-BFS approach as blocksworld, goal
  is a subset of `at-package` atoms. Deliberately kept to 2 cities (not 3+)
  to keep the reachable-state space small enough for uninformed BFS to stay
  laptop-fast (sub-10ms/problem even with retries).
- `verifier/domains/planner.py`: pure-Python BFS gold planner
  (`bfs_plan`) — since BFS expands states in order of increasing depth, the
  first goal state found is reached by a shortest/optimal plan; no external
  planner dependency (Fast Downward etc.) was needed at this scale. Also
  houses `candidate_ground_actions` (type-safe grounding, forbids an action
  binding two different parameters to the same object) and `random_walk`
  (shared by both domain generators to reach a goal-reachable state).
- `scripts/generate_problems.py`: CLI —
  `python -m scripts.generate_problems --domain {blocksworld,logistics}
  --n N --seed S --out path.jsonl [--min-plan-len] [--max-plan-len]`.
  Writes one JSON object per line: `{problem, gold_plan, plan_length}`.
- Generated `verifier/data/synthetic/{blocksworld,logistics}_problems.jsonl`,
  100 problems each, seed 0 (gitignored per existing `.gitignore` rule —
  only `verifier/data/fixtures/` is tracked; regenerate with the CLI command
  above). Blocksworld plan lengths: min 4, max 8, avg 4.46. Logistics: min
  3, max 7, avg 3.4 — both within the target 3-8 range.
- Tests (24, all passing): `tests/schema/test_schema.py` (Domain/Problem
  validation — undeclared predicate/resource references, arity mismatches,
  parameter-type mismatches, resource initial-above-cap, round-trip via
  `model_dump`/`model_validate`); `tests/domains/test_blocksworld.py` and
  `test_logistics.py` (generation determinism under a fixed seed, gold
  plans satisfy the goal and never violate resource floor/cap when
  re-simulated from scratch, custom plan-length ranges respected);
  `tests/domains/test_planner.py` (hand-built problems with a known-optimal
  plan, and an unreachable goal correctly returns `None`).

**Bug caught and fixed during this session:** the first pass at
`generate_problem` built the goal literal list by iterating a `frozenset`
of atoms directly. `frozenset`/`set` iteration order depends on Python's
per-process string hash seed (`PYTHONHASHSEED`, randomized by default), so
in-process determinism tests passed but two separate CLI invocations with
the same `--seed` produced JSONL files that differed in goal-literal order.
Fixed by sorting before converting to a list in both `blocksworld.py` and
`logistics.py`. Added `tests/domains/test_generation_cross_process.py`,
which spawns fresh subprocesses (so it actually exercises different
`PYTHONHASHSEED` values) to catch this class of bug in the future —
verified manually too by running the CLI under `PYTHONHASHSEED=111` vs
`PYTHONHASHSEED=222` and diffing the output. **Lesson for later phases:**
anywhere a `set`/`frozenset` (or dict keyed by a set) feeds into serialized
output, sort before emitting, and prefer a subprocess-based check over an
in-process one when testing "does this reproduce identically."

**Explicitly not done (by design, per task scope):**
- No LLM plan-generation harness, structured extractor, symbolic verifier,
  learned trust model, fusion, or baselines yet. No induced-failure-mode
  prompting. No third domain (Gripper-style) — 2 domains satisfies the
  spec's "2-3" and keeps this phase small.
- No external planner (Fast Downward/VAL) dependency — pure-Python BFS is
  fast enough at this action-space size; revisit only if a later phase
  needs longer-horizon plans where BFS stops being competitive.

**Next:**
- Symbolic verifier (consistency/goal-completeness/resource-feasibility
  checks) operating on the DSL in `verifier/schema/` — largely already
  exercised by `state.py`'s `simulate`/`applicable`, but needs to be wrapped
  as a standalone checker that reports *which* conjunct/precondition/
  resource failed, not just pass/fail, per spec section 3.
- LLM plan-generation harness (`verifier/generation/`): prompt an LLM with
  a problem's NL description, get back a candidate plan, including
  prompt variations designed to induce goal omission / precondition slip /
  resource overrun / hallucinated effect failure modes (spec section 5).
- Structured extractor skeleton (`verifier/extraction/`) to map LLM
  free-text/tool-call plans into the ground-action form this session's
  planner already consumes.

---

## 2026-07-20 — Phase 2: LLM plan generation + failure injection + oracle labeler

**Built:**
- `verifier/llm.py`: shared Anthropic client; API key from env or gitignored
  `.env`. Default model `claude-haiku-4-5` (fast/cheap tier, per spec —
  thousands of calls); every call site takes a model override.
- `verifier/generation/prompts.py`: Problem → NL prompt (prose action
  descriptions, not schema dumps) with a constrained output format
  (`Step N: action(args)`). Four conditions, each with a docstring stating
  the failure mode it elicits: `baseline` (honest prompt — natural failures),
  `goal_omission` (one goal conjunct silently dropped from the prompt; labels
  computed against the FULL goal), `resource_blind` (caps and per-action
  costs stripped), `distractor` (plausible invalid actions listed, e.g. a
  one-step `move(x, y)` in blocksworld).
- `verifier/generation/parser.py`: rule-based parser for the constrained
  format (tolerant of `1.` / `1)` / bare `action(args)` variants); validates
  action name + arity against the domain; unknown/malformed steps become
  errored steps, not crashes.
- `verifier/generation/labeler.py` [SOUNDNESS-CRITICAL]: oracle labeler —
  an INDEPENDENT forward-simulation implementation (deliberately does not
  reuse `verifier/schema/state.py`) producing `is_consistent` (+violations),
  `is_goal_complete` (+unmet conjuncts), `is_resource_feasible` (+resource/
  step violations), `overall_valid`. Documented semantics: lenient
  progression (a failed precondition is recorded but effects still apply, so
  one early slip doesn't mask later goal omissions); unparseable/unknown
  steps count against consistency; prefix-checked resources; closed world.
- `scripts/generate_plans.py` CLI. Design decision: by default resource caps
  are tightened to ceil(gold-plan consumption × 1.25) (`--resource-slack`,
  `--no-tighten` to disable) — the Phase 1 generator's caps were generous
  enough that almost no plan could overrun them, which would have left
  resource-infeasibility unrepresented; tightening keeps optimal plans
  feasible while wasteful ones overrun.
- Tests: 16 in `tests/generation/` — parser format tolerance + validation,
  and adversarial labeler fixtures covering every flaw type (valid plan,
  precondition slip, goal omission, resource overrun (goal reached but
  budget blown), hallucinated action, wrong arity, unknown object, empty
  plan, early-slip-doesn't-mask-goal, multiple simultaneous flaws).

**Data generated** (30 problems × 4 conditions × 2 domains = 240 records,
claude-haiku-4-5, temperature 1.0, seed 0):

| domain | condition | valid | inconsistent | goal-incomp. | res-infeas. | halluc. action |
|---|---|---|---|---|---|---|
| blocksworld | baseline       | 16/30 (53%) | 13 | 4  | 6 | 0 |
| blocksworld | goal_omission  |  8/30 (27%) | 15 | 16 | 5 | 0 |
| blocksworld | resource_blind | 13/30 (43%) | 15 | 5  | 9 | 0 |
| blocksworld | distractor     | 15/30 (50%) | 15 | 5  | 5 | 2 |
| logistics   | baseline       | 17/30 (57%) | 12 | 3  | 5 | 0 |
| logistics   | goal_omission  | 16/30 (53%) | 11 | 10 | 2 | 0 |
| logistics   | resource_blind | 20/30 (67%) | 10 | 2  | 4 | 0 |
| logistics   | distractor     | 17/30 (57%) | 13 | 2  | 1 | 2 |

**Pilot calibration (per spec section 10):** the baseline condition IS the
natural-failure pilot: 43–47% of unprompted plans are flawed, dominated by
precondition slips (~40%), with goal-incompleteness and resource overruns
both occurring naturally. The induced conditions amplify failure types that
already occur naturally rather than injecting alien ones — goal_omission
quadruples goal-incompleteness (4→16 bw, 3→10 log), resource_blind raises
overruns on blocksworld (6→9). **Honest caveats:** (1) resource_blind did
NOT raise overruns on logistics (5→4) — logistics plans are short enough
that even cost-ignorant plans stay under the 1.25× cap; (2) the distractor
effect is weak (0→2 hallucinated actions per 30) — haiku mostly resists the
invalid actions. Both noted for LIMITATIONS.md; conditions may need
sharpening (tighter slack for logistics, more tempting distractors) before
the paper's full-scale run.

**Example records (sanity-check evidence):**
- goal_omission, `blocksworld-0-0`: prompt omitted `on(b3, b2)`; model
  produced `unstack(b3,b2); put-down(b3); pick-up(b2); stack(b2,b1)` —
  consistent, resource-feasible, but labeled goal-incomplete with unmet
  conjunct `on(b3, b2)`. Exactly the intended failure signature.
- resource_blind, `blocksworld-0-6`: 7-step plan reaches the goal but spends
  10 energy against a tightened cap of 8 → `is_resource_feasible=false`
  (`step 7: energy dropped to -2`), plus a natural double-pick-up
  inconsistency at step 2. Multiple flaw dimensions labeled independently.

**First live smoke test also caught a natural failure**: the very first
baseline call produced `unstack(b3,b2); pick-up(b2); ...` — a precondition
slip (picking up while still holding b3) — confirming natural failures are
plentiful enough for the "symbolically clean but flawed" analysis later.

---

## 2026-07-20 — Phase 3: LLM structured extractor

**Built:**
- `verifier/extraction/extractor.py`: `extract_step()` maps one NL plan step
  into `{action_type, args, preconditions_referenced, resource_deltas,
  extractor_confidence}` using the API's structured-output mode
  (`output_config.format` json_schema — supported on haiku-4.5). Validates
  action name + arity against the domain; on validation failure retries ONCE
  with the error fed back, then falls back to a null extraction with
  confidence 0.0 (never crashes). `extract_step_self_consistent()` resamples
  k times (default 3) at temperature 0.7 in a thread pool and records
  `agreement_exact` (modal (action, args) fraction) and
  `agreement_action_type` — stored alongside the extraction as Phase 5
  features, not discarded.
- `verifier/extraction/plan_extractor.py`: plan-level wrapper — splits raw
  plan text into step-like lines, extracts each (parallelized), exposes
  `mean/min_confidence` and `mean_agreement` plan-level features.
- Wired in behind a flag: `scripts/verify_plans.py --parser {rule,llm}` so
  "rule-based parse + oracle label" vs "LLM extraction + verdict" can be
  compared, per the spec. Rule-based stays the free ground-truth proxy.
- Tests: 7 in `tests/extraction/` — pure-logic validation tests always run;
  live-API tests (skipped automatically when no key) check the extractor
  recovers the same actions as the rule-based parser on clean fixture input,
  degrades gracefully on garbage, and agrees with itself under resampling.

**Example extractions (live, claude-haiku-4-5):**
- `"Step 2: stack(b1, b2)"` → `stack(b1, b2)`, confidence 0.95, valid.
- `"Now carefully place block b3 on top of block b1"` → `stack(b3, b1)`,
  confidence 0.95, preconditions_referenced `[holding(b3), clear(b1)]` —
  correct extraction from free NL, no rigid format needed.
- `"Step 5: frobnicate the quux with gusto"` → `pick-up(quux)`, confidence
  0.05 — degrades to a *low-confidence* guess rather than crashing; the
  confidence signal is what the learned layer consumes.
- Self-consistency on `"pick up block b2 from the table"`: k=3 →
  `pick-up(b2)`, agreement 1.0.

---

## 2026-07-20 — Phase 4: symbolic verifier + oracle cross-check

**Built:**
- `verifier/symbolic/checker.py` [SOUNDNESS-CRITICAL]: `consistency_check`,
  `goal_completeness_check`, `resource_feasibility_check` (running-sum per
  dimension, checked at every prefix, structured
  `ResourceViolationDetail(resource, step, value, limit, kind)`), and
  `verify()` bundling all three into a `Verdict` with `overall_valid` and a
  human-readable explanation string (for the paper's qualitative examples /
  demo). `verify()` accepts `input_errors` so upstream parse/extraction
  failures count against consistency (an executor couldn't run those steps).
- Progression semantics intentionally match the Phase 2 labeler (lenient:
  record violations, still apply effects) so verdicts are comparable 1:1 —
  but the implementations are independent (labeler is self-contained; the
  verifier builds on `verifier/schema/state.py` grounding helpers), so
  agreement between them is evidence, not tautology.
- `scripts/verify_plans.py` CLI: writes verdict JSONL; with `--parser rule`
  it also cross-checks every verdict against the oracle label and exits
  nonzero on any disagreement. `--parser llm` runs the Phase 3 extractor
  (k-resampled) and records per-step confidence/agreement in the output.
- Tests: 11 in `tests/symbolic/` — valid plan, empty plan, precondition
  violation with step number, unknown action/object, arity mismatch,
  resource floor violation at an interior prefix (step 7 of 8), exactly-at-
  floor legality, input_errors counting against consistency, negated
  preconditions + no-precondition actions (hand-built toy domain, since the
  shipped domains have neither), a replenishing resource that violates its
  cap (positive delta), and an in-process fixture cross-check against the
  labeler on every flaw type.

**Cross-check evidence (the soundness gate for everything downstream):**
- Ran the verifier over all 240 plan records (both domains, rule parser):
  **240/240 agreement** with the oracle labeler on `overall_valid`, and
  **0 mismatches across all 960 per-dimension comparisons** (240 records ×
  {consistent, goal-complete, resource-feasible, overall}). One test bug was
  found and fixed during this phase (the *test* wrongly expected 2 unmet
  goal conjuncts on the empty plan; one conjunct already held in the initial
  state — the verifier was right).
- VAL (external PDDL validator) sanity check: skipped as permitted by the
  task instructions — two independent in-repo implementations agreeing
  everywhere, plus hand-verified fixtures, was judged sufficient at this
  stage; revisit for the paper's camera-ready if reviewers want it.

---

## 2026-07-20 — Phase 6: tool-use domain (built early, out of order)

Note on ordering: the tool-use domain was BUILT before Phase 5's results
because Phase 5's training data (LLM-extraction verdicts) takes ~15 min of
API time per domain to generate, and building the third domain first let all
three domains' data generate in one pass. No Phase 5 decision depended on
Phase 6 and vice versa; results are still reported per phase below.

**Built:**
- `verifier/domains/tools/`: 8 mock travel-booking tools (authenticate,
  search_flights, book_flight, search_hotels, book_hotel, charge_card,
  send_confirmation_email, create_calendar_event) expressed in the SAME
  schema DSL: tool call = action, auth scopes + prerequisite-call
  constraints = predicates (a call's "return value" is threaded to later
  calls as a trip-scoped boolean predicate, e.g. search_flights adds
  flight-options(trip) which book_flight requires), rate limits and spend =
  resource dimensions (api-quota: every call -1; budget: flights 300,
  hotels 200). Generator varies 1-2 trips, goal milestones, and partially-
  progressed sessions (pre-granted scopes / pre-run searches).
- **Generality evidence (for the paper):** NO changes to the Phases 1-5
  code were required. The DSL fit as-is; the "mock environment dry-run"
  that auto-labels flaws is literally the Phase 2 labeler / Phase 4
  verifier running on this domain — zero new labeling code. The one
  representational simplification: tool return values are boolean
  availability facts rather than typed values threaded between calls
  (adequate for the flaw taxonomy: invalid ordering, missing prerequisite,
  quota exceeded, wrong arg type via the DSL's type checking, budget
  exceeded — each covered by a dedicated test). One DSL-shaped wrinkle:
  the DSL has no constant action arguments, so tools take an explicit
  scope parameter pinned by is-<scope> identity predicates — mildly
  awkward but no schema change.
- Tests: 7 in `tests/domains/test_tools.py` — determinism, gold-plan
  validity/feasibility (15 problems), and one hand-written fixture per
  tool-use flaw type labeled through the shared labeler.
- Data: 100 problems (optimal plans 3-8 steps, avg 5.26); 30 problems × 4
  conditions of LLM plans; symbolic verdicts cross-check **120/120
  agreement** with the oracle labeler (bringing the Phase 4 cross-check
  total to 360/360 records, 1440/1440 dimension comparisons).

---

## 2026-07-20 — Phase 7: baselines

**Built (all emit the shared SystemResult/record schema for Phase 8):**
- `verifier/baselines/llm_judge.py`: zero-shot and CoT variants of "is this
  plan valid?" on the same base LLM; verdict parsed from a required
  `VERDICT: VALID|INVALID` line (last occurrence wins, so CoT reasoning that
  *mentions* a verdict mid-thought doesn't confuse parsing); unparseable
  output → predicted_valid=None → fail-open (counted as accept), mirroring a
  naive deployment.
- `verifier/baselines/symbolic_only.py`: the Phase 4 verifier with the
  fusion threshold forced to 0.0 (trust ignored) — the named ablation row.
- `verifier/baselines/learned_only.py`: classifier on the Phase 5 feature
  vector WITH ALL FIVE SYMBOLIC-VERDICT FEATURES REMOVED, trained to
  predict oracle overall_valid directly — the no-symbolic-grounding
  ablation.
- `verifier/baselines/self_repair.py`: Reflexion-style single-pass
  critique+fix; the repaired plan is oracle-labeled. Two lenses recorded:
  detection (did the action sequence change — an implicit "I found a flaw")
  and repair quality (flawed→fixed and valid→broken rates).
- `scripts/run_baselines.py`: runs judge×2 + self-repair over the SAME
  prose plan text the hybrid pipeline verifies (like-for-like).
- Tests: 7 in `tests/baselines/` — verdict parsing incl. malformed and
  case/spacing variants and CoT-mentions-both, fail-open on None, symbolic
  wiring, and a check that no symbolic feature leaks into learned-only.

**Design note (documented ambiguity call):** self-repair has no native
accept/reject verdict, so for the shared P/R table its "reject" is
change-detection (rule-parse of original vs repaired action sequences);
this conflates cosmetic rewrites with detection, so its repair-quality
numbers are reported separately in results/self_repair_quality.json and
the caveat is in LIMITATIONS.md.

---

## 2026-07-20 — Phase 5: learned trust model + fusion (first end-to-end result)

**The critical finding first:** the initial trust-label extraction produced
**240/240 "faithful"** — with the constrained `Step N: action(args)` output
format, the LLM extractor's verdict NEVER disagreed with the rule-based
reference, so the label had one class and training crashed. Diagnosis: the
constrained format makes NL→schema translation essentially lossless, which
is good news for deployments that can force a format, but leaves the learned
layer nothing to model. **Pivot (documented ambiguity call):** the production
pipeline now verifies a PROSE PARAPHRASE of each plan
(`scripts/paraphrase_plans.py` rewrites each plan as free-flowing sentences;
`raw_llm_plan_original` retains the constrained text), while oracle labels
stay anchored to the original constrained text. This creates the realistic
lossy-translation regime the trust model is FOR — and after the pivot the
label balance is **347/360 faithful (96.4%), 13 unfaithful** — sparse but
real. Recorded in LIMITATIONS.md: the trust layer's demonstrated value is
specific to the prose regime.

**Built:**
- `verifier/learned/features.py`: 17 features — extraction confidence
  (mean/min), k-resample agreement (exact + action-type), plan-shape stats,
  fraction of invalid extraction steps, TF-IDF char-n-gram cosine between
  referenced and schema preconditions (chosen over neural embeddings: zero
  new deps, laptop-fast; noted lexical-only in LIMITATIONS.md), the symbolic
  verdict bits + violation counts, domain indicator. No token logprobs — the
  API doesn't expose them (documented omission).
- Label = `verdict.overall_valid == oracle.overall_valid` (is the
  production-pipeline verdict FAITHFUL to the rule-based reference).
- `verifier/learned/model.py`: problem-level 0.6/0.2/0.2 split (records from
  one problem never straddle splits), StandardScaler + class-balanced
  logistic regression, then Platt calibration on the val split. sklearn 1.9
  removed `cv="prefit"` from CalibratedClassifierCV, so Platt scaling is a
  small custom class (1-feature LR on the base model's probabilities).
- `verifier/fusion/decision.py`: `decide()` — a hard symbolic failure
  ALWAYS rejects regardless of trust score (dedicated invariant test sweeps
  trust=0..1 and asserts reject); among symbolic passes, trust < threshold
  rejects. `sweep_threshold()` for the PR curve; th=0.0 reduces exactly to
  symbolic-only (asserted in tests).
- Tests: 13 in `tests/learned/` + `tests/fusion/` — feature extraction on
  fixture records, split-by-problem leakage check, the fusion soundness
  invariant, sweep monotonicity/endpoints.

**First end-to-end result (360 prose-regime records, claude-haiku-4-5
extraction with k=3, split 216/72/72 by problem):**
- Trust classifier: test accuracy@0.5 **0.958**, ECE **0.067** (reliability
  mass sits in the 0.9–1.0 bin at acc 1.0; the few low-trust predictions
  land in 0.7–0.9 bins at acc 0.5–0.6 — over-confident there, but the
  ranking is right).
- Fusion threshold sweep on test (positive class = flawed plan):
  th=0.0 (≡ symbolic-only) P=0.903 R=1.000 F1=0.949 (tp=28 fp=3 fn=0),
  **flat through th=0.95**, collapsing at th=1.0 (P=0.389).
- **Honest flag, as instructed:** on THIS 72-record test split the learned
  layer adds nothing over symbolic-only — the LLM-extraction pipeline's
  symbolic verdict already catches all 28 flawed plans (its 3 errors are
  false REJECTS, which no amount of extra rejecting fixes), and trust scores
  are so bimodal that no threshold below 1.0 flips any decision. With only
  13 negative labels in 360 records this is expected at dev scale; whether
  the trust layer earns its keep must be judged on (a) the pooled
  clean-but-flawed analysis in Phase 8 and (b) the full-scale run. The
  claim "hybrid > symbolic-only" is NOT supported at dev scale and the
  paper draft must not assert it without the larger run.

---

## 2026-07-20 — Phase 8: unified evaluation + downstream experiment

**Built:**
- `verifier/eval/harness.py`: per-system/per-domain P/R/F1 (positive class =
  flawed plan; reject = positive prediction), per-flaw-type recall
  (inconsistency / goal-incompleteness / resource-infeasibility, with
  supports), threshold picking on val, trust calibration (ECE + reliability
  bins), clean-but-flawed analysis, latency measurement.
- `verifier/eval/downstream.py`: execute-or-reject — arms: no verifier
  (execute everything), hybrid-gated, CoT-judge-gated; a rejected plan gets
  ONE replan with the rejecting system's feedback (the hybrid passes its
  symbolic explanation string; the judge passes its critique), and the
  replan is oracle-checked. Task success = an executed plan that is valid.
- `scripts/run_full_eval.py`: joins verdicts + baselines by (problem_name,
  condition), pooled problem-level split, trains trust + learned-only on
  train, picks threshold on val, evaluates 6 systems on the 72-record test
  split, writes results/ (metrics.csv, summary.json, threshold_sweep.json,
  calibration.json, clean_but_flawed.json, self_repair_quality.json,
  latency_cost.json, downstream.json, qualitative_examples.md, figures/
  pr_curve+reliability+ablation in PNG+SVG). Every reported number is
  regenerated from the JSONL data by this one script.

**Bug found via the eval and fixed (documented in LIMITATIONS.md):** the
zero-shot judge's original 64-token budget made **360/360 outputs
unparseable** — haiku ignores "respond with exactly one line" and simulates
the plan step-by-step, so the budget truncated before any VERDICT line and
fail-open turned the judge into accept-everything (P=R=F1=0.0). At 512
tokens still 41% unparseable; at 2048 (same as CoT) 0% unparseable. The
judge rows below use the 2048 budget; the zs-vs-CoT contrast is therefore
prompt-only. Deployment lesson worth a paragraph in the paper: a fail-open
LLM judge with format non-compliance silently degrades to "no verifier".

**Headline numbers (72-record pooled test split, positive = flawed):**

| system | P | R | F1 |
|---|---|---|---|
| hybrid (trust th=0.0 picked on val) | 0.903 | 1.000 | 0.949 |
| symbolic-only | 0.903 | 1.000 | 0.949 |
| learned-only (no symbolic features) | 0.618 | 0.750 | 0.677 |
| LLM judge zero-shot (2048 tok) | 1.000 | 1.000 | 1.000 |
| LLM judge CoT | 1.000 | 1.000 | 1.000 |
| self-repair (change-detection) | 0.966 | 1.000 | 0.982 |

Per-flaw-type recall: hybrid catches 20/20 inconsistency, 10/10
goal-incompleteness, 5/5 resource-infeasibility. Per-domain: hybrid perfect
on blocksworld + logistics; tools precision 0.833 (3 false rejects, see
below).

**Honest flags (as instructed — several results cut against the pitch):**
1. **Both LLM judges are PERFECT on this test split** — at dev scale
   (3–8-step plans, haiku judging haiku) a properly-budgeted judge
   simulates these short plans flawlessly. "Hybrid beats LLM-judge on
   detection" is NOT supported at dev scale; the hybrid's measured
   advantages are downstream task success (below), zero API cost at
   decision time for the symbolic path, determinism, and machine-readable
   explanations. Longer-horizon problems are the obvious full-scale test.
2. **The trust layer never fires**: threshold picked on val is 0.0, so
   hybrid ≡ symbolic-only on every test record; the test split has ZERO
   clean-but-flawed records (41 symbolic passes, all genuinely valid) for
   it to catch. Only 13/360 unfaithful-translation records exist pooled.
3. **But the trust signal is real, just pointed the wrong way by the fusion
   rule**: the hybrid's only 3 errors are false REJECTS on the tools domain
   where the extractor mangles hyphenated args (`flights-scope` →
   `flights`) — and those exact records get the trust model's LOWEST scores
   (0.74–0.82 vs 0.9+ elsewhere; the reliability table's low bins are
   exactly these). The current rule only lets trust ADD rejects; a
   "low-trust symbolic reject → re-extract or fall back to rule parser"
   repair path would have fixed all 3 errors. Noted as the concrete
   architectural finding of the dev run.
4. Learned-only (no symbolic features) at F1 0.677 confirms the symbolic
   grounding carries the detection signal — the ablation the spec wanted.

**Downstream execute-or-reject (72 test records, one replan allowed):**

| arm | task success | executed-flawed rate | replans | dead-ends |
|---|---|---|---|---|
| no verifier | 0.611 | 0.389 | 0 | 0 |
| hybrid-gated | **0.931** | **0.000** | 31 | 5 |
| CoT-judge-gated | 0.875 | 0.000 | 28 | 9 |

Both gates eliminate flawed execution entirely; the hybrid converts more of
its rejects into successful replans (5 dead-ends vs 9) — its symbolic
explanation ("step 3: unknown object...", "goal conjunct X unmet") is more
actionable replan feedback than the judge's free-text critique. This is
where the hybrid's edge actually shows at dev scale.

**Other results:** self-repair fixes 23/28 flawed plans and breaks 0/44
valid ones (surprisingly strong; its weakness is detection precision, not
repair). Calibration ECE 0.067. Latency: symbolic check ~0.2µs/plan local
with zero API calls; hybrid adds k×steps extraction calls (~12–24,
batchable); judges 1 call each (~2048-token generations).

**Data/figures:** results/ regenerated end-to-end twice (before/after the
judge-budget fix); figures pr_curve, reliability, ablation in PNG+SVG;
qualitative_examples.md sections: (a) hybrid-caught-judge-missed — EMPTY
(judges perfect), (b) trust-caught-symbolic-missed — EMPTY (no
clean-but-flawed in test), (c) the 3 tools false-rejects with trust scores
and mangled extractions shown. The empty sections are kept in the file
deliberately — they are themselves dev-scale findings.

---

## 2026-07-20 — Phase 9: reproducibility + final coherence pass

**Built/verified:**
- Pinned environment: `uv.lock` + `.python-version` (3.11.15); README
  documents both `uv sync` and plain venv+pip (with the caveat that only
  uv reproduces locked versions).
- `Makefile`: `make reproduce` (smoke config: 20 problems / 8 plans per
  condition per domain) and `make reproduce FULL=1` (100/30 — the dev
  numbers above used 30 problems × 4 conditions per domain). Chain: test →
  problems → plans → paraphrase → verdicts (rule + LLM parser) → baselines
  → eval. Every stage is a CLI; every number in results/ comes from
  `scripts.run_full_eval` reading the JSONL data.
- README rewritten (pitch, setup, reproduce instructions, artifact table,
  pipeline stages, layout, soundness anchors). LIMITATIONS.md finalized —
  including the two dev-scale findings that cut against the pitch (perfect
  judges on short plans; trust layer never fires + one-directional fusion)
  and the zero-shot-judge token-budget lesson.
- Final coherence pass: full offline suite 85/85 passing; grepped for stale
  references after the judge-budget fix (none outside the intentional
  historical note); latency_cost.json note updated in the script AND the
  generated file to match; every number quoted in PROGRESS.md Phases 5/8
  traced back to trust_model_report.json / results/*.json / metrics.csv.

**Not done (deliberate, recorded):** full-scale run (`FULL=1`) and
multi-seed replication — the dev-scale numbers are the honest current
state; VAL external validation (rationale in Phase 4); neural embedding
features (lexical TF-IDF stands in, rationale in LIMITATIONS.md).

---

## 2026-07-20 — Horizon experiment: 10/20/40/80-step plans

**Motivation:** the Phase 8 conclusion flagged an open question — both LLM
judges scored a perfect F1=1.0 on 3-8-step plans, so "hybrid beats judge on
detection" was unsupported at that horizon. This experiment asks whether
judge detection degrades as plans get longer while sound symbolic
simulation does not.

**Built:**
- `verifier/domains/horizon.py`: constructive gold planners for each domain
  (BFS is infeasible past ~10 steps). Blocksworld tears every tower down to
  the table then rebuilds goal towers bottom-up; logistics routes each
  package sequentially through truck/airport/plane legs; tools
  authenticates each needed scope once then runs every trip's milestone
  prerequisite chain in dependency order. Plans are valid by construction
  AND independently strict-simulated at generation time
  (`verifier/schema/state.py:simulate`, raises on any violation) plus a
  goal check — a bug in a constructive planner cannot silently ship a bad
  reference plan. Honest caveat: these plans are valid but NOT optimal
  (blocksworld especially), so "horizon H" is a nominal band
  ([0.85H, 1.2H]) on the reference length, not an optimality claim.
- `scripts/generate_problems.py --horizon H`: switches to the constructive
  generators.
- `scripts/horizon_pipeline.sh`: idempotent tiered pipeline — full 6-system
  pipeline (10 problems x 4 conditions = 40 records/cell) at every horizon
  for judges/self-repair/rule-parsed-symbolic; the expensive
  k-resampled LLM-extraction path (k x steps calls/plan) runs on all 40
  records at h10/h20 but only a 12-record stratified subset (3/condition)
  at h40/h80 to bound API cost. Token budgets raised throughout (plan
  generation/paraphrase 4096, judge/self-repair 8192) so 80-step plans
  don't get silently truncated the way the Phase 8 zero-shot judge did at
  64 tokens.
- `scripts/run_horizon_eval.py`: trains the trust model and learned-only
  baseline ONLY on the original short-horizon data (zero long-horizon
  leakage — this also measures the trust model's out-of-distribution
  generalization to longer plans) and evaluates all 6 systems at
  horizons {5 (short-horizon reference, Phase 8 test split), 10, 20, 40,
  80}. Emits per-horizon metrics, diagnostics (judge token cost, LLM plan
  length, extraction fidelity), 5 figures (F1/recall/precision vs horizon,
  judge cost vs horizon, extraction fidelity vs horizon), and downstream
  execute-or-reject at h10/h20.
- 28 new tests (`tests/domains/test_horizon.py`): every domain x horizon
  cell's gold plan validated by the INDEPENDENT oracle labeler (a second,
  separately-implemented opinion beyond the generation-time strict
  simulation), band membership, cross-process-style determinism, resource
  dimensions non-degenerate. 106 offline tests total now pass.

**Result — the headline finding, and it reverses the Phase 8 caveat:**

| horizon | hybrid/symbolic F1 | judge-zs F1 | judge-zs recall | judge-CoT F1 | judge-CoT recall |
|---|---|---|---|---|---|
| 5 (short-horizon ref) | 0.949 | 1.000 | 1.000 | 1.000 | 1.000 |
| 10 | **1.000** | 0.967 | 0.936 | 0.960 | 0.923 |
| 20 | **1.000** | 0.950 | 0.914 | 0.976 | 0.952 |
| 40 | **1.000** | 0.937 | 0.882 | 0.962 | 0.927 |
| 80 | **1.000** | 0.914 | **0.842** | 0.938 | 0.883 |

The hybrid (and rule-parsed symbolic-only) hold perfect F1=1.000 and
recall=1.000 at every horizon from 10 to 80 steps — as expected, since
forward simulation's cost is linear in plan length and correctness doesn't
degrade with length. Both LLM judges degrade monotonically in recall as
horizon grows (zero-shot 1.00 -> 0.94 -> 0.91 -> 0.88 -> 0.84; CoT 1.00 ->
0.92 -> 0.95 -> 0.93 -> 0.88, noisier but the same downward drift) — they
increasingly fail to CATCH flawed plans (false negatives), not fail to
parse: judge mean output tokens rise with horizon (zs 634->854, CoT
765->1192 tokens) and 0/480 judge calls were unparseable at any horizon, so
this is a genuine simulation-accuracy failure, not a formatting artifact.
The flaw rate among LLM-generated plans also rises sharply with horizon
(38.9% at h5 -> 65% at h10 -> 87.5% at h20 -> 100% at h80) — longer
LLM-generated plans are both harder to verify AND more likely to be flawed,
compounding the value of a sound checker exactly where judges get worse.
Self-repair's "detection" (change-on-repair) is noisy across horizon
(0.98->0.98->0.88->0.91->0.92) with no clean trend — consistent with the
Phase 8 caveat that it's an indirect signal.

**Extraction fidelity vs horizon:** 100% faithful (LLM-extraction verdict
== rule-based reference) at every long horizon (10/20/40/80), vs 95.8% on
the short-horizon reference split. This is a genuinely surprising result
worth flagging rather than smoothing over: it does NOT mean translation
gets easier at length. The more likely explanation (recorded as a
follow-up question, not resolved here) is that the k=3 self-consistency
modal vote and/or the mechanical, template-like structure of the
constructive long-horizon plans (repeated authenticate/search/book or
load/drive/unload subsequences) make the extractor's output more
reproducible even when individual steps are still imperfect — i.e., this
metric measures agreement between two extraction paths of the SAME
underlying plan, not ground-truth extraction accuracy, and a
mechanically-repetitive plan can be consistently mis-extracted the same
way both times. Flagged in LIMITATIONS.md rather than presented as
"extraction gets better with scale."

**Downstream execute-or-reject at h10/h20** (120 records/horizon, same
methodology as Phase 8 — one feedback-driven replan):

| horizon | arm | task success | executed-flawed |
|---|---|---|---|
| 10 | no verifier | 0.350 | 0.650 |
| 10 | hybrid-gated | **0.667** | 0.000 |
| 10 | CoT-judge-gated | 0.525 | 0.050 |
| 20 | no verifier | 0.125 | 0.875 |
| 20 | hybrid-gated | **0.383** | 0.000 |
| 20 | CoT-judge-gated | 0.283 | 0.042 |

Two things worth flagging honestly. First, absolute task success falls
sharply with horizon for every arm (the underlying LLM planner gets much
worse at 10-20 step plans, which is expected and not a verifier property).
Second, and importantly: unlike at h5 where BOTH gates achieved 0%
executed-flawed, at h10/h20 the **judge-gated arm lets flawed plans through
0.05 (h10) / 0.042 (h20) of the time** — a real, nonzero soundness gap that
directly instantiates the recall degradation above, while the **hybrid
stays at exactly 0.000 executed-flawed at every horizon tested (5 through
20)** — the one guarantee a judge fundamentally cannot make. The hybrid
also converts more of its rejections into successful replans at both
horizons (task success 0.667 vs 0.525 at h10; 0.383 vs 0.283 at h20).

**What this changes in the paper's story:** the Phase 8 conclusion said
"whether detection separates at longer horizons is the central open
question" and declined to claim hybrid > judge on detection. This
experiment answers that question with real data: it does separate, in the
hybrid's favor, and the separation shows up first as a recall gap (10-80
steps) and then as an actual soundness violation in the downstream
experiment (10-20 steps, the only horizons where a live downstream run was
affordable). The paper has been updated to report this as a real finding
rather than an open question, while keeping the honest caveats: long-horizon
extraction cells are 12-record subsets (wide error bars), gold plans there
are valid-but-not-optimal, and the downstream soundness-gap result is only
demonstrated at h10/h20 (not yet run at h40/h80, where it would likely be
larger given the recall trend).
