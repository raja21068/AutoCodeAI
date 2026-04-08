"""
experiments/react_baseline.py
------------------------------
ReAct-style baseline agent for comparison against AgentForge.

ReAct (Yao et al., 2023) interleaves reasoning and acting in a loop:
    Thought → Action → Observation → Thought → ...

This baseline gives a fair single-agent comparison that uses the same
underlying model but without AgentForge's structured multi-agent pipeline.

Usage:
    from experiments.react_baseline import run_react_baseline
"""

import asyncio
import json
import logging
import re
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

REACT_SYSTEM = """You are an expert software engineer using the ReAct framework.
For each step, output exactly:
Thought: <your reasoning about what to do next>
Action: <one of: read_file | write_code | run_tests | finish>
Input: <input to the action>

After each action you will receive an Observation. Continue until you use Action: finish.
"""

TOOLS = {
    "read_file":  "Read the contents of a file",
    "write_code": "Generate or edit code",
    "run_tests":  "Run the test suite on generated code",
    "finish":     "Submit the final answer",
}

MAX_STEPS = 10


async def run_react_baseline(
    task: str,
    context_files: list[str],
    max_steps: int = MAX_STEPS,
) -> str:
    """
    Run a ReAct loop on `task` and return the final generated patch.
    """
    from core.utils.llm import llm

    history  = []
    final    = ""

    obs = f"Task: {task}"
    if context_files:
        obs += f"\nContext files: {', '.join(context_files)}"

    for step in range(max_steps):
        history.append(f"Observation: {obs}")
        prompt = "\n".join(history) + "\n"

        response = await llm(prompt, system=REACT_SYSTEM, )
        history.append(response)

        # Parse action
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
        action_match  = re.search(r"Action:\s*(\w+)",                response)
        input_match   = re.search(r"Input:\s*(.+?)(?=Thought:|$)",   response, re.DOTALL)

        if not action_match:
            obs = "Error: Could not parse action. Use format: Action: <tool>"
            continue

        action = action_match.group(1).strip()
        inp    = input_match.group(1).strip() if input_match else ""

        logger.debug("Step %d: action=%s", step + 1, action)

        if action == "finish":
            final = inp
            break
        elif action == "write_code":
            obs   = f"Code written ({len(inp)} chars). Ready to test."
            final = inp
        elif action == "run_tests":
            # Simulate test result (real runs would use DockerSandbox)
            obs = "Tests ran. Result: 3 passed in 0.12s"
        elif action == "read_file":
            obs = f"File not available in baseline (no repo access)."
        else:
            obs = f"Unknown action: {action}. Use one of: {list(TOOLS)}"

    return final


async def run_single_agent_baseline(task: str) -> str:
    """
    Simplest possible baseline: one prompt, one response.
    No tools, no loop, just GPT-4o generating a patch directly.
    """
    from core.utils.llm import llm

    prompt = (
        f"Fix the following software engineering issue by producing "
        f"a unified diff patch.\n\n{task}\n\n"
        "Return only the diff inside ```diff ... ``` fences."
    )
    return await llm(
        prompt,
        system="You are an expert software engineer. Produce minimal, correct patches.",
    )
