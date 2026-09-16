"""
services/repair_orchestrator.py — Benchmark repair loop (Algorithm 1).

This is the evaluated system. It is separate from ``services/orchestrator.py``,
which drives the interactive web application and streams to a UI.

Two properties distinguish it from the previous benchmark path:

1. **It operates on a real repository.** Every edit is applied to a checkout
   inside the instance container, and every command runs against that
   checkout. The patch submitted for grading is ``git diff`` against the base
   commit — repository state, not parsed model prose.

2. **The post-edit execution invariant is enforced by control flow.** After a
   state-changing edit, ``self._pending_execution`` is set and the loop
   refuses to dispatch another edit role until a command has run and its
   outcome has been appended to the trajectory. The Planner cannot waive it,
   because the Planner does not schedule execution — the loop does.

Configuration exposes the factorial the evaluation needs::

    RepairConfig(role_structure="multi", force_execution=True)   # AgentForge
    RepairConfig(role_structure="multi", force_execution=False)  # optional exec
    RepairConfig(role_structure="single", force_execution=True)
    RepairConfig(role_structure="single", force_execution=False)

All four share one budget object, so a comparison holds model, tokens, cost,
wall-clock and execution count constant by construction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from core.utils.llm import LLMResult, llm_call
from eval.task_adapter import SWEBenchTaskAdapter, assert_no_leakage
from services.repo_retrieval import RepoRetriever

logger = logging.getLogger(__name__)

EDIT_ROLES = {"coder", "debugger"}


# ---------------------------------------------------------------------------
# Configuration and accounting
# ---------------------------------------------------------------------------


@dataclass
class RepairConfig:
    """
    One cell of the experimental grid.

    ``disabled_roles`` states exactly what a role-removal ablation changes, so
    that "without the Planner" has a definition rather than an implementation
    accident. Removing a role removes its call and its effect on control flow;
    it never silently changes tool permissions, execution opportunities, or the
    context budget of the remaining roles:

        planner   the triage call is skipped; the edit prompt carries no
                  triage section. Retrieved context is unchanged.
        tester    no reproduction script is written; the post-edit validation
                  command falls back to the repository's own test entry point.
        debugger  failures no longer route to the repair prompt; edits are
                  always produced by the Coder prompt, which sees the same
                  execution history.
        critic    a clean run terminates the loop immediately instead of
                  asking whether to continue.

    Because the shared budget is unchanged, a removed role frees its calls for
    the remaining loop rather than shortening the run — which is what makes
    the ablation a comparison of structure rather than of spend.
    """

    role_structure: str = "multi"       # "multi" (five roles) | "single"
    force_execution: bool = True        # enforce the post-edit invariant
    use_retrieval: bool = True
    use_generated_tests: bool = True
    disabled_roles: frozenset = frozenset()

    model: str | None = None            # overrides per-agent routing when set
    max_iterations: int = 8
    max_executions: int = 12
    max_cost_usd: float = 1.50
    max_wall_clock_s: float = 900.0
    exec_timeout_s: int = 300

    def as_dict(self) -> dict:
        """
        JSON-safe view of the configuration.

        Every record this config is embedded in gets written with
        ``json.dumps``, and ``disabled_roles`` is a frozenset, which json
        rejects. Handing out ``vars(self)`` therefore produced a record that
        could not be serialised — and because the failure happens at write
        time, after the agent has finished, the whole instance was discarded
        once its work was already paid for. Sets are emitted sorted so the
        same configuration always serialises identically.
        """
        out: dict = {}
        for key, value in vars(self).items():
            out[key] = sorted(value) if isinstance(value, (set, frozenset)) else value
        return out


@dataclass
class Budget:
    """Shared resource ledger. Identical across compared configurations."""

    config: RepairConfig
    started_at: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    executions: int = 0
    iterations: int = 0

    def record(self, result: LLMResult) -> None:
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.cost_usd += result.cost_usd
        self.llm_calls += 1

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def exhausted(self) -> str | None:
        c = self.config
        if self.iterations >= c.max_iterations:
            return "max_iterations"
        if self.executions >= c.max_executions:
            return "max_executions"
        if self.cost_usd >= c.max_cost_usd:
            return "max_cost"
        if self.elapsed_s >= c.max_wall_clock_s:
            return "max_wall_clock"
        return None

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "llm_calls": self.llm_calls,
            "executions": self.executions,
            "iterations": self.iterations,
            "wall_clock_s": round(self.elapsed_s, 2),
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_SYS = (
    "You are a senior software engineer triaging a bug report against an "
    "unfamiliar repository. Given the issue and retrieved repository context, "
    "state in at most five sentences: the most likely responsible file(s), the "
    "suspected root cause, and one shell command that would demonstrate the bug. "
    "You are not given the fix. Reason only from the issue text and the code shown."
)

CODER_SYS = (
    "You are an expert software engineer. Produce a minimal unified diff in "
    "git format that fixes the described issue. Use exact paths relative to the "
    "repository root. Include at least three lines of context per hunk. "
    "Do not modify test files. Return ONLY the diff inside ```diff fences."
)

TESTER_SYS = (
    "You write a short reproduction script for a bug report. Return ONLY a "
    "runnable Python script that exercises the described behaviour and exits "
    "non-zero when the bug is present. This is a diagnostic aid, not a "
    "specification of correctness."
)

DEBUGGER_SYS = (
    "You are an expert debugger. You are given a patch, the command that was "
    "run, its exit status and its output. Diagnose why it failed and return a "
    "corrected minimal unified diff against the ORIGINAL files, in git format, "
    "inside ```diff fences. Return ONLY the diff."
)

CRITIC_SYS = (
    "You are a senior code reviewer. Given an issue, a cumulative diff and the "
    "commands executed with their outcomes, reply 'STOP' if the change plausibly "
    "resolves the issue and the evidence supports stopping, or 'CONTINUE: <what "
    "to investigate next>'. You are not deciding benchmark correctness."
)

SINGLE_SYS = (
    "You are an expert software engineer resolving an issue in a repository. "
    "Given the issue, repository context and the history of your own edits and "
    "command outputs, return the next minimal unified diff in git format inside "
    "```diff fences, or the single word DONE if the issue is resolved."
)


# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------


class RepairOrchestrator:
    def __init__(self, config: RepairConfig | None = None) -> None:
        self.config = config or RepairConfig()
        self.adapter = SWEBenchTaskAdapter()

    async def _ask(self, budget: Budget, system: str, prompt: str, agent: str) -> str:
        result = await llm_call(
            prompt, system=system, agent=agent, model=self.config.model
        )
        budget.record(result)
        return result.text

    async def run(self, instance: dict, env) -> dict:
        """
        Resolve one instance. Returns a record containing the model patch,
        the full trajectory and the resource ledger.
        """
        cfg = self.config
        budget = Budget(config=cfg)
        trajectory: list[dict] = []

        task = self.adapter.to_agent_input(instance)
        prompt_base = task["prompt"]
        assert_no_leakage(prompt_base, instance, where="task prompt")

        # -- Retrieval over the base checkout only ----------------------
        context = ""
        if cfg.use_retrieval:
            retriever = RepoRetriever(env)
            hits = retriever.retrieve(task["problem_statement"])
            context = retriever.format(hits)
            trajectory.append({"role": "retrieval", "files": [h.path for h in hits]})
            assert_no_leakage(context, instance, where="retrieved context")

        # -- Plan (multi-role only) -------------------------------------
        plan = ""
        if cfg.role_structure == "multi" and "planner" not in cfg.disabled_roles:
            plan = await self._ask(
                budget,
                PLANNER_SYS,
                f"{prompt_base}\n\n## Repository context\n{context}",
                "planner",
            )
            trajectory.append({"role": "planner", "output": plan})

        # -- Optional reproduction script -------------------------------
        repro_path = None
        if (
            cfg.role_structure == "multi"
            and cfg.use_generated_tests
            and "tester" not in cfg.disabled_roles
        ):
            script = await self._ask(
                budget, TESTER_SYS, f"{prompt_base}\n\n## Context\n{context}", "tester"
            )
            script = _strip_fences(script)
            if script:
                repro_path = "repro_agent.py"
                env.write_file(repro_path, script)
                trajectory.append({"role": "tester", "script_path": repro_path})

        # -- Main loop --------------------------------------------------
        self._pending_execution = False
        last_feedback = ""
        stop_reason = "budget"

        while True:
            halt = budget.exhausted()
            if halt:
                stop_reason = halt
                break
            budget.iterations += 1

            # ---- Edit step ------------------------------------------
            # INVARIANT: an edit role may only be dispatched when no
            # execution is outstanding.
            if cfg.force_execution and self._pending_execution:
                raise AssertionError(
                    "post-edit execution invariant violated: edit attempted "
                    "while execution was outstanding"
                )

            edit_prompt = self._build_edit_prompt(
                prompt_base, context, plan, trajectory, last_feedback
            )
            use_debugger = bool(last_feedback) and "debugger" not in cfg.disabled_roles
            role = "debugger" if use_debugger else "coder"
            if cfg.role_structure == "single":
                role, system = "coder", SINGLE_SYS
            else:
                system = DEBUGGER_SYS if use_debugger else CODER_SYS

            raw = await self._ask(budget, system, edit_prompt, role)

            if cfg.role_structure == "single" and raw.strip().upper().startswith("DONE"):
                stop_reason = "agent_done"
                break

            diff = self.adapter.extract_diff_block(raw)
            if not diff:
                trajectory.append({"role": role, "error": "no diff produced"})
                last_feedback = "Your previous reply contained no unified diff."
                continue

            applied, message = env.apply_patch(diff)
            trajectory.append(
                {"role": role, "applied": applied, "detail": message, "diff": diff}
            )

            if not applied:
                # The repository did not change, so no execution is owed.
                last_feedback = f"Your patch did not apply:\n{message}"
                continue

            self._pending_execution = True

            # ---- Mandatory execution --------------------------------
            if not cfg.force_execution and not self._should_execute_optionally(plan):
                # Ablation arm: execution is available but not compelled.
                self._pending_execution = False
                last_feedback = ""
                continue

            command = self._select_command(instance, repro_path)
            code, out, err = env.exec(command, timeout=cfg.exec_timeout_s)
            budget.executions += 1
            self._pending_execution = False

            observation = _truncate(f"$ {command}\nexit={code}\n{out}\n{err}")
            trajectory.append(
                {"role": "execution", "command": command, "exit_code": code,
                 "observation": observation}
            )

            if code == 0:
                if cfg.role_structure == "single" or "critic" in cfg.disabled_roles:
                    stop_reason = "clean_run"
                    break
                verdict = await self._ask(
                    budget,
                    CRITIC_SYS,
                    f"{prompt_base}\n\n## Cumulative diff\n{_truncate(env.diff())}\n\n"
                    f"## Evidence\n{observation}",
                    "critic",
                )
                trajectory.append({"role": "critic", "output": verdict})
                if verdict.strip().upper().startswith("STOP"):
                    stop_reason = "critic_stop"
                    break
                last_feedback = verdict
            else:
                last_feedback = observation

        # The invariant must hold on exit as well.
        assert not (cfg.force_execution and self._pending_execution), (
            "loop exited with an unexecuted edit outstanding"
        )

        # Remove our scratch artifacts so they never reach the grader.
        if repro_path:
            env.exec(f"rm -f {repro_path}")

        model_patch = self.adapter.extract_patch(env)
        return {
            "instance_id": instance["instance_id"],
            "model_patch": model_patch,
            "stop_reason": stop_reason,
            "trajectory": trajectory,
            "budget": budget.as_dict(),
            "config": self.config.as_dict(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_edit_prompt(
        self,
        prompt_base: str,
        context: str,
        plan: str,
        trajectory: list[dict],
        feedback: str,
    ) -> str:
        parts = [prompt_base]
        if context:
            parts.append(f"## Repository context\n{context}")
        if plan:
            parts.append(f"## Triage\n{plan}")

        history = [
            f"- {e['role']}: {e.get('detail') or e.get('command') or ''}"
            for e in trajectory
            if e["role"] in {"coder", "debugger", "execution"}
        ]
        if history:
            parts.append("## History\n" + "\n".join(history[-8:]))
        if feedback:
            parts.append(f"## Most recent outcome\n{_truncate(feedback, 4000)}")
        return "\n\n".join(parts)

    def _select_command(self, instance: dict, repro_path: str | None) -> str:
        """
        Choose the post-edit validation command.

        Deliberately conservative: it never consults FAIL_TO_PASS or
        PASS_TO_PASS. The agent validates with the repository's own test
        entry point and its own reproduction script.
        """
        if repro_path:
            return f"python {repro_path}"
        repo = instance.get("repo", "")
        if "django" in repo:
            return "python -m pytest tests/ -x -q --timeout=120 2>&1 | tail -40"
        return "python -m pytest -x -q 2>&1 | tail -40"

    @staticmethod
    def _should_execute_optionally(plan: str) -> bool:
        """
        In the optional-execution arm, execution happens only when the plan
        asked for it — reproducing the behaviour the original Algorithm 1
        actually had, where a Tester step had to be scheduled.
        """
        return "test" in (plan or "").lower()


def _strip_fences(text: str) -> str:
    import re

    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def _truncate(text: str, limit: int = 8000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-limit // 2 :]
    return f"{head}\n…[{len(text) - limit} chars elided]…\n{tail}"
