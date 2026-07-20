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
