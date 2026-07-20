# Pre-Execution Plan Verifier

A lightweight, model-agnostic verifier that sits between "LLM proposes a
plan" and "system executes it": a sound symbolic checker (consistency,
goal-completeness, resource-feasibility via forward simulation) fused with a
small calibrated learned model that estimates how much to trust the symbolic
checker's own parse of the LLM's free-text plan. Hard symbolic failures
always reject; among symbolic passes, the trust score ranks the residual
risk that the NL→schema translation was lossy — catching plans that are
symbolically clean but actually flawed. Laptop-scale throughout: no
fine-tuning, no GPU, no external planner.

See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full research design,
[`PROGRESS.md`](PROGRESS.md) for the per-phase build log with evidence, and
[`LIMITATIONS.md`](LIMITATIONS.md) for honest caveats.

## Setup

Targets **Python 3.11+**, managed with [`uv`](https://docs.astral.sh/uv/)
(exact dependency versions are pinned in `uv.lock`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
uv sync                                            # installs Python 3.11 + all deps
uv run pytest                                      # offline test suite
```

Plain venv+pip also works (`python3.11 -m venv .venv && pip install -e .`),
but only `uv sync` reproduces the locked versions.

**API key**: create a gitignored `.env` at the repo root containing
`ANTHROPIC_API_KEY=<your key>` (or export it). All LLM stages default to
`claude-haiku-4-5` for cost; every CLI takes `--model`.

## Reproducing results/

```bash
make reproduce          # smoke config: small n (~15 min, ~$1 of API calls)
make reproduce FULL=1   # full-scale config behind the reported numbers
```

Both run the entire pipeline with fixed seeds: problem generation → LLM plan
generation under 4 failure-injection conditions → oracle labeling → prose
paraphrase → LLM extraction + symbolic verdicts → LLM baselines → training,
evaluation, figures. Every number in `results/` is produced by
`scripts/run_full_eval.py` from the JSONL data — no manual steps.

Key artifacts after a run:

| Artifact | Contents |
|---|---|
| `results/summary.json` | headline numbers: split sizes, trust threshold, ECE, clean-but-flawed catch rate, downstream success rates |
| `results/metrics.csv` | P/R/F1 per system per domain + per-flaw-type recall |
| `results/threshold_sweep.json` | fusion PR curve data (th=0.0 row = symbolic-only) |
| `results/calibration.json` | ECE + reliability-diagram bins |
| `results/self_repair_quality.json` | flawed→fixed / valid→broken repair rates |
| `results/downstream.json` | execute-or-reject task success rates |
| `results/qualitative_examples.md` | case studies: hybrid-caught-judge-missed, trust-caught-symbolic-missed, hybrid mistakes |
| `results/figures/` | PR curve, reliability diagram, ablation bar chart (PNG+SVG) |

## Pipeline stages (each is a CLI in `scripts/`)

```
generate_problems  --domain {blocksworld,logistics,tools} --n N --seed S --out ....jsonl
generate_plans     --problems ... --conditions baseline,goal_omission,resource_blind,distractor --out ...
paraphrase_plans   --plans ... --out ..._nl.jsonl        # prose regime (see PROGRESS.md Phase 5)
verify_plans       --plans ... --out ... [--parser rule|llm --k 3]
run_baselines      --plans ..._nl.jsonl --out ..._baselines.jsonl
train_trust_model  --verdicts ..._verdicts_llm.jsonl ... --out ....pkl
run_full_eval      --out results/
```

## Directory layout

```
verifier/
  schema/        # typed DSL: predicates, actions, resources, Domain, Problem, simulation
  domains/       # blocksworld + logistics + tools domains, BFS gold planner, generators
  generation/    # prompt conditions, rule-based parser, ORACLE LABELER, LLM harness
  symbolic/      # sound symbolic verifier (the hard gate) with explanations
  extraction/    # LLM structured extractor with k-resampled self-consistency
  learned/       # trust features + calibrated trust model
  fusion/        # decision rule: symbolic gate + trust threshold; PR sweep
  baselines/     # LLM-judge (zs/CoT), symbolic-only, learned-only, self-repair
  eval/          # unified metrics, calibration, downstream execute-or-reject
  data/          # generated datasets (gitignored except fixtures)
scripts/         # one CLI per pipeline stage (see above)
tests/           # pytest, mirroring verifier/ (LLM tests auto-skip without a key)
```

## Soundness anchors

Two independent implementations of plan simulation exist on purpose: the
oracle labeler (`verifier/generation/labeler.py`, self-contained) and the
symbolic verifier (`verifier/symbolic/checker.py`, built on the schema
grounding helpers). `scripts/verify_plans.py --parser rule` cross-checks
them on every record and exits nonzero on any disagreement — at dev scale:
360/360 records, 1440/1440 per-dimension comparisons in agreement.

## Workflow notes

- Append to `PROGRESS.md` per work session/phase — never overwrite entries.
- `verifier/data/` stays gitignored except `verifier/data/fixtures/`.
