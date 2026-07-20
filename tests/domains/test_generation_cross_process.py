"""Determinism must hold across separate Python processes, not just within one.

PYTHONHASHSEED (and therefore frozenset/set iteration order) is randomized
per-process by default. A generator that leaks that iteration order into its
output (e.g. building a goal list straight from a frozenset of atoms without
sorting) can look deterministic in-process, while `scripts/generate_problems`
run twice from the CLI on different days produces different JSONL. This test
actually spawns fresh interpreters to catch that class of bug.
"""

import json
import subprocess
import sys

_SNIPPET = """
import json
from verifier.domains.{module} import generate_problem
problem, plan = generate_problem(seed=42, index=3)
print(json.dumps({{"problem": problem.model_dump(), "plan": [(a.schema_name, a.args) for a in plan]}}))
"""


def _run_in_fresh_process(module: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET.format(module=module)],
        capture_output=True,
        text=True,
        check=True,
        cwd=__file__.rsplit("/tests/", 1)[0],
    )
    return json.loads(result.stdout)


def test_blocksworld_deterministic_across_processes():
    out1 = _run_in_fresh_process("blocksworld")
    out2 = _run_in_fresh_process("blocksworld")
    assert out1 == out2


def test_logistics_deterministic_across_processes():
    out1 = _run_in_fresh_process("logistics")
    out2 = _run_in_fresh_process("logistics")
    assert out1 == out2