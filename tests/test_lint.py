"""The repo proves its own health: lint and typecheck are part of `uv run pytest`.

Issue #31 was 'no linter, no type-checker'. Keeping them in the test run means
they can't quietly disappear again — if the checks are missing, this fails
rather than skipping.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    ("ruff check", [sys.executable, "-m", "ruff", "check", "src", "tests"]),
    ("ruff format", [sys.executable, "-m", "ruff", "format", "--check", "src", "tests"]),
    ("mypy", [sys.executable, "-m", "mypy"]),
]


@pytest.mark.parametrize("name,cmd", CHECKS, ids=[name for name, _ in CHECKS])
def test_check_passes(name: str, cmd: list[str]) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=120)
    except ModuleNotFoundError as exc:
        pytest.fail(f"{name} is not installed ({exc}); run: uv sync --group dev")

    if proc.returncode != 0:
        output = (proc.stdout + proc.stderr).strip()
        pytest.fail(f"{name} reported problems:\n\n{output}")
