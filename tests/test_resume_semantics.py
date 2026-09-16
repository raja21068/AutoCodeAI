"""
tests/test_resume_semantics.py — Resume means done, not attempted.

``--resume`` exists so an interrupted sweep can continue without repeating
completed work. It keyed only on the existence of a trajectory file, so an
instance that failed -- provider outage, exhausted account balance, container
error -- counted as complete forever. The retry then skipped precisely the
instances that needed retrying, and the run reported a resolve rate over a
population that had never actually run.

This was observed: a grid run died partway through on "Insufficient Balance",
leaving 22 of 50 instances as error records in every configuration. Resuming
after topping the account up would have preserved all 22 failures and
produced a number describing the billing incident.

Run:
    python -m pytest tests/test_resume_semantics.py -v
"""

from __future__ import annotations

import asyncio
import json

import pytest

from eval import swebench_runner
from services.repair_orchestrator import RepairConfig

INSTANCE = {
    "instance_id": "astropy__astropy-12907",
    "repo": "astropy/astropy",
    "base_commit": "d16bfe05",
    "problem_statement": "boom",
    "test_patch": "",
}


def write_record(tmp_path, **overrides):
    d = tmp_path / "trajectories"
    d.mkdir(parents=True, exist_ok=True)
    record = {
        "instance_id": INSTANCE["instance_id"],
        "model_patch": "diff --git a/x b/x\n",
        "stop_reason": "clean_run",
        "trajectory": [],
        "budget": {},
    }
    record.update(overrides)
    (d / f"{INSTANCE['instance_id']}.json").write_text(
        json.dumps(record), encoding="utf-8")
    return record


def run(tmp_path, monkeypatch, should_rerun_flag):
    """Invoke run_instance with the agent stubbed so no container is needed."""
    class StubAgent:
        async def run(self, instance, env):
            should_rerun_flag.append(True)
            return {
                "instance_id": instance["instance_id"],
                "model_patch": "diff --git a/new b/new\n",
                "stop_reason": "clean_run",
                "trajectory": [],
                "budget": {},
            }

    class StubEnv:
        exec_count = 0
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(swebench_runner, "make_agent", lambda kind, cfg: StubAgent())
    monkeypatch.setattr(swebench_runner, "InstanceEnv", lambda *a, **k: StubEnv())

    return asyncio.run(swebench_runner.run_instance(
        INSTANCE, RepairConfig(), "agentforge", tmp_path, resume=True))


def test_successful_record_is_not_rerun(tmp_path, monkeypatch):
    write_record(tmp_path)
    reran = []
    result = run(tmp_path, monkeypatch, reran)
    assert reran == [], "a completed instance must not be repeated"
    assert result["stop_reason"] == "clean_run"


def test_errored_record_is_retried(tmp_path, monkeypatch):
    write_record(tmp_path, model_patch="", stop_reason="error",
                 error="AuthenticationError: Insufficient Balance")
    reran = []
    result = run(tmp_path, monkeypatch, reran)
    assert reran == [True], "an errored instance must be retried on resume"
    assert result["model_patch"].startswith("diff --git a/new")
    assert not result.get("error")


def test_retry_clears_the_stale_error(tmp_path, monkeypatch):
    """The rewritten record must not keep the old failure alongside a patch."""
    write_record(tmp_path, model_patch="", stop_reason="error",
                 error="DeepseekException - Insufficient Balance")
    run(tmp_path, monkeypatch, [])
    on_disk = json.loads(
        (tmp_path / "trajectories" / f"{INSTANCE['instance_id']}.json")
        .read_text(encoding="utf-8"))
    assert not on_disk.get("error")
    assert on_disk["model_patch"].strip()
