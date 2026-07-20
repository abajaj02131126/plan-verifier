# Pre-Execution Plan Verifier

Laptop-scale research codebase for a hybrid symbolic + learned "pre-execution
plan verifier" for LLM-generated plans, targeting an AAAI main-track
submission. See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full research
design (problem formalization, architecture, evaluation domains, baselines,
metrics). See [`PROGRESS.md`](PROGRESS.md) for the running build log.

## Status

Repo skeleton only — no domain, symbolic, extraction, or learned-layer logic
implemented yet. This commit exists to establish tooling, structure, and a
passing test/CLI baseline before any real code lands.

## Setup

This project targets **Python 3.11+** and uses [`uv`](https://docs.astral.sh/uv/)
for environment and dependency management. `uv` also manages the Python
interpreter itself, so you don't need a system Python 3.11 install.

### Option A: `uv` (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the repo root — installs Python 3.11 (pinned via .python-version)
# and creates .venv with all dependencies, including dev deps (pytest)
uv sync

# Run the CLI stub
uv run python -m verifier --help

# Run tests
uv run pytest
```

### Option B: plain venv + pip

Requires a Python 3.11+ interpreter already available on your PATH.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m verifier --help
pytest
```

Note: `pip install -e ".[dev]"` requires the `dev` optional-dependency group
to be declared as an extra rather than a PEP 735 dependency group if your
pip/setuptools version doesn't support `[dependency-groups]`. If that
install command fails on your pip version, use `uv` instead, or install
dev dependencies manually: `pip install -e . pytest`.

## Environment variables

- `ANTHROPIC_API_KEY` — required once the LLM plan-generation and structured
  extraction stages are implemented. Never hardcode this; export it in your
  shell or an untracked `.env` file.

## Directory layout

```
verifier/
  schema/        # domain predicate/resource/action-type schemas (DSL)
  domains/       # synthetic PDDL-style + tool-use domain definitions
  generation/    # LLM plan-generation harness, failure injection
  symbolic/      # consistency / goal-completeness / resource-feasibility checkers
  extraction/    # structured extractor (plan text -> schema objects)
  learned/       # feature extraction, trust model training + inference
  fusion/        # decision rule combining symbolic verdict + trust score
  baselines/     # LLM-judge, symbolic-only, learned-only, self-repair
  eval/          # metrics, evaluation harness, ablation runner
  data/          # generated datasets (gitignored except small fixtures)
  configs/       # yaml/json experiment configs
scripts/         # CLI entry points, one per pipeline stage
tests/           # pytest, mirroring the verifier/ package structure
```

## Contributing / workflow notes

- Append to `PROGRESS.md` at the end of each work session — never overwrite
  past entries.
- Keep `verifier/data/` gitignored except for small fixture files needed by
  tests (`verifier/data/fixtures/`).
