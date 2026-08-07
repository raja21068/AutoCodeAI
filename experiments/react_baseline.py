"""
experiments/react_baseline.py
-----------------------------
ReAct baseline (Yao et al., 2023) on the same footing as AgentForge.

The previous version of this file could not serve as a baseline. Its
``run_tests`` action returned the fixed string ``"Tests ran. Result: 3 passed
in 0.12s"`` no matter what the agent had written, and its ``read_file`` action
always answered ``"File not available in baseline (no repo access)"``. It
therefore had no repository, no execution, and fabricated observations telling
it that its work had succeeded. Any margin measured against it was an artifact
of that handicap.

This rewrite gives the ReAct loop exactly what AgentForge gets: the same
instance container, the same leakage-free prompt, the same tools, the same
budget object, and the same official grading path. What differs is only the
control structure — one agent interleaving thought and action, against five
roles with an enforced execution schedule. That is the comparison the paper
intends to make.

The agent exposes ``run(instance, env)`` and returns the same record shape as
:class:`services.repair_orchestrator.RepairOrchestrator`, so the runner can
dispatch to either without special-casing the output.
"""

from __future__ import annotations

import logging
import re
import shlex

from core.utils.llm import llm_call
from eval.task_adapter import SWEBenchTaskAdapter, assert_no_leakage
from services.repair_orchestrator import Budget, RepairConfig, _truncate
from services.repo_retrieval import RepoRetriever

logger = logging.getLogger(__name__)

REACT_SYSTEM = """You are an expert software engineer resolving an issue in a real repository, using the ReAct framework.

Each turn, output exactly:
Thought: <your reasoning about what to do next>
Action: <one of: read_file | search | apply_patch | run | finish>
Input: <input to the action>

Actions:
  read_file   Input is a path relative to the repository root.
  search      Input is a regular expression; returns matching files and lines.
  apply_patch Input is a unified diff in git format, inside ```diff fences.
  run         Input is a shell command executed in the repository.
  finish      Input is a short statement of what you changed.

You will receive an Observation after each action. The repository is real and
your patches persist. Locate the defect yourself; you are not told which file
is at fault. Do not modify test files.
"""

MAX_STEPS = 16


class ReactBaseline:
    """Single-agent ReAct loop with genuine repository access."""

    def __init__(self, config: RepairConfig | None = None) -> None:
        self.config = config or RepairConfig(role_structure="single")
        self.adapter = SWEBenchTaskAdapter()

    async def run(self, instance: dict, env) -> dict:
        cfg = self.config
        budget = Budget(config=cfg)
        trajectory: list[dict] = []

        task = self.adapter.to_agent_input(instance)
        assert_no_leakage(task["prompt"], instance, where="react task prompt")

        observation = task["prompt"]
        if cfg.use_retrieval:
            retriever = RepoRetriever(env)
            hits = retriever.retrieve(task["problem_statement"])
            context = retriever.format(hits)
            assert_no_leakage(context, instance, where="react retrieved context")
            if context:
                observation += f"\n\n## Repository context\n{context}"
            trajectory.append({"role": "retrieval", "files": [h.path for h in hits]})

        history: list[str] = []
        stop_reason = "budget"

        for _ in range(MAX_STEPS):
            halt = budget.exhausted()
            if halt:
                stop_reason = halt
                break
            budget.iterations += 1

            history.append(f"Observation: {_truncate(observation, 6000)}")
            result = await llm_call(
                "\n\n".join(history[-12:]),
                system=REACT_SYSTEM,
                agent="coder",
                model=cfg.model,
            )
            budget.record(result)
            history.append(result.text)

            action, payload = self._parse(result.text)
            if action is None:
                observation = (
                    "Error: could not parse an action. Reply with "
                    "'Action: <read_file|search|apply_patch|run|finish>'."
                )
                continue

            observation = self._act(action, payload, env, budget, trajectory)
            if action == "finish":
                stop_reason = "agent_done"
                break

        model_patch = self.adapter.extract_patch(env)
        return {
            "instance_id": instance["instance_id"],
            "model_patch": model_patch,
            "stop_reason": stop_reason,
            "trajectory": trajectory,
            "budget": budget.as_dict(),
            "config": {**vars(self.config), "agent": "react"},
        }

    # ------------------------------------------------------------------
    # Parsing and acting
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(text: str) -> tuple[str | None, str]:
        action_match = re.search(r"Action:\s*(\w+)", text)
        if not action_match:
            return None, ""
        action = action_match.group(1).strip()

        input_match = re.search(
            r"Input:\s*(.+?)(?=\nThought:|\nAction:|$)", text, re.DOTALL
        )
        return action, (input_match.group(1).strip() if input_match else "")

    def _act(self, action: str, payload: str, env, budget: Budget,
             trajectory: list[dict]) -> str:
        if action == "read_file":
            path = payload.strip().strip("`'\"")
            content = env.read_file(path)
            trajectory.append({"role": "read_file", "path": path})
            return (f"Contents of {path}:\n{content}" if content
                    else f"Could not read {path}. Check the path.")

        if action == "search":
            pattern = payload.strip().strip("`'\"")
            code, out, _ = env.exec(
                f"grep -rn --include='*.py' -E {shlex.quote(pattern)} . "
                f"2>/dev/null | head -40"
            )
            trajectory.append({"role": "search", "pattern": pattern})
            return out if out.strip() else f"No matches for {pattern!r}."

        if action == "apply_patch":
            diff = self.adapter.extract_diff_block(payload) or payload
            applied, message = env.apply_patch(diff)
            trajectory.append({"role": "apply_patch", "applied": applied,
                               "detail": message})
            return (f"Patch applied ({message})." if applied
                    else f"Patch did NOT apply:\n{message}")

        if action == "run":
            command = payload.strip().strip("`")
            code, out, err = env.exec(command, timeout=self.config.exec_timeout_s)
            budget.executions += 1
            trajectory.append({"role": "execution", "command": command,
                               "exit_code": code})
            return f"$ {command}\nexit={code}\n{out}\n{err}"

        if action == "finish":
            trajectory.append({"role": "finish", "summary": payload[:300]})
            return "Finished."

        return f"Unknown action {action!r}."


async def run_single_call_baseline(instance: dict, cfg: RepairConfig) -> dict:
    """
    Lower bound: one prompt, one response, no tools, no repository, no
    execution. Reported as a floor, never used to isolate the effect of
    decomposition or execution — it differs from AgentForge in every respect
    at once, so a difference against it attributes to nothing in particular.
    """
    adapter = SWEBenchTaskAdapter()
    budget = Budget(config=cfg)
    task = adapter.to_agent_input(instance)
    assert_no_leakage(task["prompt"], instance, where="single-call prompt")

    result = await llm_call(
        task["prompt"] + "\nReturn only a unified diff inside ```diff fences.",
        system="You are an expert software engineer. Produce minimal, correct patches.",
        agent="coder",
        model=cfg.model,
    )
    budget.record(result)
    budget.iterations = 1

    return {
        "instance_id": instance["instance_id"],
        "model_patch": adapter.extract_diff_block(result.text) or "",
        "stop_reason": "single_call",
        "trajectory": [{"role": "single_call", "output": result.text[:2000]}],
        "budget": budget.as_dict(),
        "config": {**vars(cfg), "agent": "single_call"},
    }
