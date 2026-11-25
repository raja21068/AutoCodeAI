"""
tests/test_env_loading.py — Credentials must reach the evaluation entrypoints.

Only main.py and experiments/setup_and_run.py used to call load_dotenv, so
`python -m eval.swebench_runner` ran with DEEPSEEK_API_KEY unset. LiteLLM sent
no credential, the provider replied 401, and the message read as "your key is
invalid" rather than "your key was never loaded".

The failure is nastier than a plain missing-config bug because it is not
reproducible in the obvious way: any preflight or probe that *does* load .env
succeeds moments before the run that does not, so it looks intermittent and
gets blamed on the provider. A real key can be rotated for nothing chasing it.

These tests run in a fresh subprocess, because an in-process test inherits
whatever pytest's own environment already holds and would pass regardless.

Run:
    python -m pytest tests/test_env_loading.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"


def run_snippet(code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env,
    )


def env_keys() -> set[str]:
    """Names (not values) defined in the repo .env."""
    keys = set()
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0].strip())
    return keys


@pytest.mark.skipif(not ENV_FILE.exists(), reason="no .env in this checkout")
def test_importing_llm_module_loads_dotenv():
    """Importing the LLM module must populate the environment on its own."""
    keys = env_keys()
    if not keys:
        pytest.skip(".env has no entries")
    probe = sorted(keys)[0]

    result = run_snippet(
        "import core.utils.llm, os; "
        f"print('FOUND' if os.getenv({probe!r}) else 'MISSING')"
    )
    assert result.returncode == 0, result.stderr
    assert "FOUND" in result.stdout, (
        f"{probe} absent after importing core.utils.llm — evaluation "
        f"entrypoints would run without credentials.\n{result.stderr}"
    )


@pytest.mark.skipif(not ENV_FILE.exists(), reason="no .env in this checkout")
def test_runner_entrypoint_sees_credentials():
    """The path that actually failed: import the runner, not the LLM module."""
    keys = env_keys()
    probe = "DEEPSEEK_API_KEY" if "DEEPSEEK_API_KEY" in keys else sorted(keys)[0]

    result = run_snippet(
        "import eval.swebench_runner, os; "
        f"print('FOUND' if os.getenv({probe!r}) else 'MISSING')"
    )
    assert result.returncode == 0, result.stderr
    assert "FOUND" in result.stdout, (
        f"{probe} absent after importing eval.swebench_runner.\n{result.stderr}"
    )


def test_real_environment_wins_over_dotenv():
    """
    override=False is deliberate: CI and per-run overrides set a real
    variable and must not be silently replaced by the checked-in file.
    """
    import os

    env = dict(os.environ)
    env["LLM_MODEL"] = "sentinel-model-from-environment"
    result = run_snippet(
        "import core.utils.llm, os; print(os.environ['LLM_MODEL'])", env=env
    )
    assert result.returncode == 0, result.stderr
    assert "sentinel-model-from-environment" in result.stdout
