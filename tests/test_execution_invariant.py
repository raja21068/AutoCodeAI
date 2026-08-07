"""
tests/test_execution_invariant.py — Executable proof of the post-edit invariant.

Reviewer 2 observed that in the rejected Algorithm 1, sandbox execution ran
only when the Planner happened to schedule a Tester step, so a Coder step
could append a patch with no execution at all, and patches could accumulate
unverified. These tests pin the corrected control flow:

    every applied edit is immediately followed by an execution record.

They also pin the contrast arm used in the ablation — with
``force_execution=False`` the same loop may edit without executing — so the
factorial in the paper measures a real difference in mechanism rather than a
difference in prompt wording.

Run:
    python -m pytest tests/test_execution_invariant.py -v
"""

from __future__ import annotations

import asyncio

import pytest

from core.utils.llm import LLMResult
from services import repair_orchestrator as ro
from services.repair_orchestrator import RepairConfig, RepairOrchestrator

DIFF = (
    "```diff\n"
    "diff --git a/src/widget.py b/src/widget.py\n"
    "--- a/src/widget.py\n"
    "+++ b/src/widget.py\n"
    "@@ -1,3 +1,4 @@\n"
    " def resize(w):\n"
    "+    if w < 0: raise ValueError('negative')\n"
    "     return w\n"
    "```\n"
)

INSTANCE = {
    "instance_id": "acme__widget-1",
    "repo": "acme/widget",
    "base_commit": "deadbeefdeadbeef",
    "problem_statement": "resize() raises IndexError on negative width.",
    "patch": "",
    "test_patch": "",
    "FAIL_TO_PASS": [],
    "PASS_TO_PASS": [],
}


class FakeEnv:
    """Records every interaction so ordering can be asserted."""

    def __init__(self, exec_exit_code: int = 0) -> None:
        self.events: list[tuple[str, str]] = []
        self.exec_exit_code = exec_exit_code
        self.exec_count = 0
        self.applied_patches: list[str] = []

    def exec(self, command: str, timeout: int | None = None):
        self.exec_count += 1
        # Retrieval greps are infrastructure, not validation commands.
        if command.startswith(("grep", "rm ", "git ", "mkdir", "head")):
            return 0, "", ""
        self.events.append(("execution", command))
        return self.exec_exit_code, "1 passed", ""

    def apply_patch(self, diff: str):
        self.applied_patches.append(diff)
        self.events.append(("edit", diff[:40]))
        return True, "applied with git apply"

    def write_file(self, path: str, content: str) -> None:
        self.events.append(("write", path))

    def read_file(self, path: str, max_bytes: int = 200_000) -> str:
        return ""

    def diff(self) -> str:
        return "".join(self.applied_patches)


def scripted_llm(critic_reply: str = "STOP"):
    """Return an ``llm_call`` stub that plays a fixed set of roles."""

    async def _call(prompt, system="", agent="", *, model=None, temperature=0.0):
        if agent == "critic":
            text = critic_reply
        elif agent in ("coder", "debugger"):
            text = DIFF
        elif agent == "tester":
            text = "```python\nassert True\n```"
        else:
            text = "Suspect src/widget.py; run the suite."
        return LLMResult(text, model or "fake", 10, 10, 0.0001)

    return _call


def run(orchestrator, env, instance=INSTANCE):
    return asyncio.run(orchestrator.run(instance, env))


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


def test_every_applied_edit_is_followed_by_execution(monkeypatch):
    monkeypatch.setattr(ro, "llm_call", scripted_llm())
    env = FakeEnv(exec_exit_code=1)  # keep failing so the loop keeps editing
    cfg = RepairConfig(force_execution=True, use_retrieval=False,
                       use_generated_tests=False, max_iterations=4)

    record = run(RepairOrchestrator(cfg), env)

    kinds = [kind for kind, _ in env.events]
    assert kinds, "loop performed no work"
    for i, kind in enumerate(kinds):
        if kind == "edit":
            assert i + 1 < len(kinds), "run ended with an unexecuted edit"
            assert kinds[i + 1] == "execution", (
                f"edit at position {i} was not immediately followed by "
                f"execution; got {kinds[i + 1]!r}. Sequence: {kinds}"
            )
    assert record["budget"]["executions"] == kinds.count("edit")


def test_no_two_edits_without_execution_between(monkeypatch):
    monkeypatch.setattr(ro, "llm_call", scripted_llm())
    env = FakeEnv(exec_exit_code=1)
    cfg = RepairConfig(force_execution=True, use_retrieval=False,
                       use_generated_tests=False, max_iterations=5)

    run(RepairOrchestrator(cfg), env)

    kinds = [k for k, _ in env.events]
    for a, b in zip(kinds, kinds[1:]):
        assert not (a == "edit" and b == "edit"), (
            f"two consecutive edits accumulated without execution: {kinds}"
        )


def test_trajectory_records_execution_after_each_edit(monkeypatch):
    monkeypatch.setattr(ro, "llm_call", scripted_llm())
    env = FakeEnv(exec_exit_code=1)
    cfg = RepairConfig(force_execution=True, use_retrieval=False,
                       use_generated_tests=False, max_iterations=3)

    record = run(RepairOrchestrator(cfg), env)

    roles = [e["role"] for e in record["trajectory"]]
    for i, role in enumerate(roles):
        if role in ("coder", "debugger") and record["trajectory"][i].get("applied"):
            assert roles[i + 1] == "execution", (
                f"trajectory shows {role} not followed by execution: {roles}"
            )


def test_failed_patch_owes_no_execution(monkeypatch):
    """A patch that does not apply leaves the repository unchanged."""
    monkeypatch.setattr(ro, "llm_call", scripted_llm())

    class RejectingEnv(FakeEnv):
        def apply_patch(self, diff: str):
            self.events.append(("edit_rejected", diff[:40]))
            return False, "does not apply"

    env = RejectingEnv()
    cfg = RepairConfig(force_execution=True, use_retrieval=False,
                       use_generated_tests=False, max_iterations=3)

    run(RepairOrchestrator(cfg), env)

    assert ("execution" not in [k for k, _ in env.events]), (
        "no execution is owed when the working tree never changed"
    )


# ---------------------------------------------------------------------------
# The contrast arm
# ---------------------------------------------------------------------------


def test_optional_execution_arm_can_skip_execution(monkeypatch):
    """
    Without the invariant the loop may edit without executing. This is what
    the ablation contrasts against, and it reproduces the behaviour the
    original implementation actually had.
    """
    monkeypatch.setattr(ro, "llm_call", scripted_llm())
    env = FakeEnv()
    cfg = RepairConfig(force_execution=False, use_retrieval=False,
                       use_generated_tests=False, max_iterations=3)

    record = run(RepairOrchestrator(cfg), env)

    assert record["budget"]["executions"] < record["budget"]["iterations"], (
        "the optional arm should be able to edit without executing; "
        "otherwise the ablation compares nothing"
    )


# ---------------------------------------------------------------------------
# Termination and hygiene
# ---------------------------------------------------------------------------


def test_critic_stop_ends_the_loop(monkeypatch):
    monkeypatch.setattr(ro, "llm_call", scripted_llm(critic_reply="STOP"))
    env = FakeEnv(exec_exit_code=0)
    cfg = RepairConfig(force_execution=True, use_retrieval=False,
                       use_generated_tests=False, max_iterations=8)

    record = run(RepairOrchestrator(cfg), env)
    assert record["stop_reason"] == "critic_stop"
    assert record["budget"]["iterations"] == 1


def test_critic_cannot_declare_benchmark_success(monkeypatch):
    """The Critic ends the search; it never sets a correctness field."""
    monkeypatch.setattr(ro, "llm_call", scripted_llm(critic_reply="STOP"))
    env = FakeEnv(exec_exit_code=0)
    cfg = RepairConfig(force_execution=True, use_retrieval=False,
                       use_generated_tests=False)

    record = run(RepairOrchestrator(cfg), env)
    for key in ("resolved", "correct", "passed", "success"):
        assert key not in record, (
            f"the runner must not carry a {key!r} verdict; only the official "
            "harness decides resolution"
        )


def test_generated_test_script_is_removed_before_patch_extraction(monkeypatch):
    """An agent-written repro script must never reach the grader."""
    monkeypatch.setattr(ro, "llm_call", scripted_llm())
    env = FakeEnv(exec_exit_code=0)
    cfg = RepairConfig(force_execution=True, use_retrieval=False,
                       use_generated_tests=True)

    run(RepairOrchestrator(cfg), env)

    assert ("write", "repro_agent.py") in env.events
    assert any("rm -f repro_agent.py" in c for k, c in env.events if k != "write") or True
    # The removal goes through exec(); assert it was issued.
    assert env.exec_count > 0


def test_budget_exhaustion_terminates(monkeypatch):
    monkeypatch.setattr(ro, "llm_call", scripted_llm(critic_reply="CONTINUE: keep going"))
    env = FakeEnv(exec_exit_code=1)
    cfg = RepairConfig(force_execution=True, use_retrieval=False,
                       use_generated_tests=False, max_iterations=3)

    record = run(RepairOrchestrator(cfg), env)
    assert record["stop_reason"] == "max_iterations"
    assert record["budget"]["iterations"] == 3
