"""
tests/test_ablation_semantics.py — What each role-removal ablation changes.

Reviewer 2 asked what exact changes are made to the pipeline in each
agent-removal ablation. The previous implementation answered that question
badly: it monkey-patched a role to a no-op, so the removed role's call still
happened and still cost tokens, and the ablation measured wasted spend as much
as missing structure.

These tests pin the current semantics, one assertion per documented effect,
so the description in the paper and the behaviour in the code cannot drift.

Run:
    python -m pytest tests/test_ablation_semantics.py -v
"""

from __future__ import annotations

import asyncio

import pytest

from services import repair_orchestrator as ro
from services.repair_orchestrator import RepairConfig, RepairOrchestrator
from tests.test_execution_invariant import INSTANCE, FakeEnv, scripted_llm


def run_with(cfg: RepairConfig, env, monkeypatch, critic_reply="STOP"):
    monkeypatch.setattr(ro, "llm_call", scripted_llm(critic_reply=critic_reply))
    return asyncio.run(RepairOrchestrator(cfg).run(INSTANCE, env))


def roles_called(record) -> list[str]:
    return [e["role"] for e in record["trajectory"]]


def test_no_planner_skips_the_triage_call(monkeypatch):
    cfg = RepairConfig(disabled_roles=frozenset({"planner"}),
                       use_retrieval=False, use_generated_tests=False)
    record = run_with(cfg, FakeEnv(exec_exit_code=0), monkeypatch)
    assert "planner" not in roles_called(record)


def test_no_tester_writes_no_reproduction_script(monkeypatch):
    env = FakeEnv(exec_exit_code=0)
    cfg = RepairConfig(disabled_roles=frozenset({"tester"}),
                       use_retrieval=False, use_generated_tests=True)
    record = run_with(cfg, env, monkeypatch)

    assert "tester" not in roles_called(record)
    assert not any(kind == "write" for kind, _ in env.events)


def test_no_critic_stops_on_a_clean_run(monkeypatch):
    cfg = RepairConfig(disabled_roles=frozenset({"critic"}),
                       use_retrieval=False, use_generated_tests=False)
    record = run_with(cfg, FakeEnv(exec_exit_code=0), monkeypatch)

    assert "critic" not in roles_called(record)
    assert record["stop_reason"] == "clean_run"


def test_no_debugger_keeps_editing_with_the_coder_prompt(monkeypatch):
    """Failures still drive another edit; they just do not switch prompts."""
    cfg = RepairConfig(disabled_roles=frozenset({"debugger"}),
                       use_retrieval=False, use_generated_tests=False,
                       max_iterations=3)
    record = run_with(cfg, FakeEnv(exec_exit_code=1), monkeypatch)

    called = roles_called(record)
    assert "debugger" not in called
    assert called.count("coder") >= 2, (
        "removing the Debugger must not stop the repair loop, only change "
        "which prompt produces the next edit"
    )


def test_role_removal_preserves_the_execution_invariant(monkeypatch):
    """No ablation is allowed to weaken the invariant under test."""
    for role in ("planner", "tester", "debugger", "critic"):
        env = FakeEnv(exec_exit_code=1)
        cfg = RepairConfig(disabled_roles=frozenset({role}),
                           use_retrieval=False, use_generated_tests=False,
                           max_iterations=3)
        run_with(cfg, env, monkeypatch)

        kinds = [k for k, _ in env.events]
        for a, b in zip(kinds, kinds[1:]):
            assert not (a == "edit" and b == "edit"), (
                f"disabling {role} broke the post-edit execution invariant: {kinds}"
            )


def test_role_removal_does_not_shorten_the_budget(monkeypatch):
    """
    A removed role frees its calls for the remaining loop rather than ending
    the run early — otherwise the ablation confounds structure with spend.
    """
    env_full = FakeEnv(exec_exit_code=1)
    full = run_with(RepairConfig(use_retrieval=False, use_generated_tests=False,
                                 max_iterations=3),
                    env_full, monkeypatch, critic_reply="CONTINUE: more")

    env_ablated = FakeEnv(exec_exit_code=1)
    ablated = run_with(RepairConfig(disabled_roles=frozenset({"planner"}),
                                    use_retrieval=False, use_generated_tests=False,
                                    max_iterations=3),
                       env_ablated, monkeypatch, critic_reply="CONTINUE: more")

    assert full["budget"]["iterations"] == ablated["budget"]["iterations"] == 3
    assert full["stop_reason"] == ablated["stop_reason"] == "max_iterations"


def test_every_named_config_is_constructible():
    """The CLI's config table must map onto real RepairConfig fields."""
    from eval.configs import CONFIGS

    for name, overrides in CONFIGS.items():
        cfg = RepairConfig(**overrides)
        assert cfg.role_structure in ("multi", "single"), name
        assert isinstance(cfg.disabled_roles, frozenset), name
