"""
eval/ablation_runner.py
-----------------------
Runs the ablation study by disabling one agent at a time.

Ablation conditions:
    full_pipeline   — all agents enabled          (main result)
    no_critic       — skip critic review
    no_debugger     — skip auto-debug loop
    no_tester       — skip test generation
    no_planner      — single coder step, no plan
    single_agent    — one GPT-4o call, no pipeline (baseline)

Usage:
    python -m eval.ablation_runner \
        --max_tasks 100 \
        --output_dir eval/results/ablation
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from datasets import load_dataset

from eval.task_adapter import SWEBenchTaskAdapter
from eval.metrics import compute_ablation_table, print_ablation_table
from core.utils.llm import llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation-aware orchestrator patches
# ---------------------------------------------------------------------------

def make_orchestrator(condition: str):
    """
    Return an Orchestrator configured for the given ablation condition.
    Each condition monkey-patches one agent to be a no-op.
    """
    from services.orchestrator import Orchestrator
    from core.agents.agents import (
        CriticAgent, DebuggerAgent, TesterAgent, PlannerAgent,
    )

    orch = Orchestrator()

    if condition == "no_critic":
        # Critic always returns PASS — never blocks pipeline
        async def always_pass(results, task): return "PASS"
        orch.critic.review = always_pass

    elif condition == "no_debugger":
        # Debugger returns original code unchanged — no fix attempts
        async def no_fix(code, error): return code
        orch.debugger.fix = no_fix

    elif condition == "no_tester":
        # Tester returns empty string — sandbox runs with no tests
        async def no_tests(code, subtask): return ""
        orch.tester.generate_tests = no_tests

    elif condition == "no_planner":
        # Planner produces a single hardcoded coder step
        async def flat_plan(task, mem, repo):
            return {
                "explanation": "single coder step",
                "steps": [{"agent": "coder", "description": task}],
            }
        orch.planner.create_plan = flat_plan

    # "full_pipeline" and "single_agent" need no patching here
    return orch


async def run_single_agent_baseline(task_nl: str) -> str:
    """
    Baseline: one GPT-4o call, no pipeline, no tests, no critic.
    """
    prompt = (
        f"Fix the following bug by producing a unified diff:\n\n{task_nl}"
    )
    return await llm(prompt, system="You are an expert software engineer.")


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

async def run_ablation(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    tasks   = list(dataset)[: args.max_tasks]
    adapter = SWEBenchTaskAdapter()

    conditions = [
        "full_pipeline",
        "no_critic",
        "no_debugger",
        "no_tester",
        "no_planner",
        "single_agent",
    ]

    all_records: dict[str, list[dict]] = {}

    for condition in conditions:
        logger.info("Running condition: %s", condition)
        cond_dir = output_dir / condition
        cond_dir.mkdir(exist_ok=True)
        records  = []

        if condition == "single_agent":
            # Baseline — no orchestrator
            for task in tasks:
                nl     = adapter.to_natural_language(task)
                patch  = await run_single_agent_baseline(nl)
                result = adapter.evaluate(task, patch)
                record = {
                    "instance_id":   task["instance_id"],
                    "repo":          task.get("repo", ""),
                    "resolved":      result["resolved"],
                    "patch_applied": result["patch_applied"],
                    "tests_passed":  result["tests_passed"],
                    "tests_failed":  result["tests_failed"],
                    "elapsed_s":     0,
                    "error":         None,
                }
                (cond_dir / f"{task['instance_id']}.json").write_text(
                    json.dumps(record, indent=2)
                )
                records.append(record)
        else:
            orch = make_orchestrator(condition)
            for task in tasks:
                nl            = adapter.to_natural_language(task)
                context_files = adapter.get_context_files(task)
                messages: list[str] = []

                try:
                    async for chunk in orch.run_streaming(nl, context_files):
                        messages.append(chunk)
                    patch = adapter.extract_code(messages)
                except Exception as exc:
                    logger.warning("Task %s failed: %s", task["instance_id"], exc)
                    patch = None

                result = adapter.evaluate(task, patch) if patch else {
                    "resolved": False, "patch_applied": False,
                    "tests_passed": 0, "tests_failed": 0,
                }
                record = {
                    "instance_id":   task["instance_id"],
                    "repo":          task.get("repo", ""),
                    "resolved":      result["resolved"],
                    "patch_applied": result["patch_applied"],
                    "tests_passed":  result["tests_passed"],
                    "tests_failed":  result["tests_failed"],
                    "elapsed_s":     0,
                    "error":         None,
                }
                (cond_dir / f"{task['instance_id']}.json").write_text(
                    json.dumps(record, indent=2)
                )
                records.append(record)

            orch.shutdown()

        all_records[condition] = records
        logger.info(
            "Condition %s done: %d/%d resolved",
            condition,
            sum(1 for r in records if r["resolved"]),
            len(records),
        )

    # Save and display ablation table
    table = compute_ablation_table(all_records)
    (output_dir / "ablation_table.json").write_text(
        json.dumps(table, indent=2)
    )
    print_ablation_table(table)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentForge ablation study")
    p.add_argument("--max_tasks",  type=int, default=100)
    p.add_argument("--output_dir", default="eval/results/ablation")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_ablation(parse_args()))
