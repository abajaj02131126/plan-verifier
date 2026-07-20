"""Shared Anthropic client construction and model defaults.

All LLM-calling modules (generation harness, extractor, LLM-judge baselines)
go through get_client() so credential loading lives in one place. The API key
is read from the environment or from a .env file at the repo root (the .env
file is gitignored; never hardcode keys).

Model default: this pipeline makes thousands of calls (plan generation x
conditions x problems, k-resampled extractions, judge baselines), so per the
project spec the default is the fast/cheap tier, claude-haiku-4-5. Every
call site takes a model parameter so individual stages can be upgraded.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic

DEFAULT_MODEL = "claude-haiku-4-5"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv_key() -> str | None:
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def get_client() -> anthropic.Anthropic:
    """Anthropic client using ANTHROPIC_API_KEY from the environment or .env."""
    key = os.environ.get("ANTHROPIC_API_KEY") or _load_dotenv_key()
    if not key:
        raise RuntimeError(
            "No ANTHROPIC_API_KEY found in the environment or in .env at the repo "
            "root. Create .env containing 'ANTHROPIC_API_KEY=<your key>'."
        )
    return anthropic.Anthropic(api_key=key)
