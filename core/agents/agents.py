"""
core/agents/agents.py — All agent classes.

Agents:
    PlannerAgent   — converts a task into a structured JSON step plan
    CoderAgent     — generates new code or produces a unified diff
    TesterAgent    — writes pytest test cases
    DebuggerAgent  — fixes code given error output
    CriticAgent    — reviews results, returns PASS or FAIL
    MemoryAgent    — stores and retrieves past successful tasks (in-process)
"""

import json
import logging
import re
from typing import AsyncGenerator

from unidiff import PatchSet

from core.utils.llm import llm, llm_stream

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class PlannerAgent:
    SYSTEM = (
        "You are a senior software architect. "
        "Given a task, produce a JSON object with two keys:\n"
        "  'explanation': a one-sentence description of the approach\n"
        "  'steps': an array of objects, each with:\n"
        "    'agent': one of coder | tester | debugger | critic | tool\n"
        "    'description': what this step does\n"
        "    'file': (optional) path of the file to edit\n"
        "    'parallel_group': (optional) integer for parallel execution (steps with same number run in parallel)\n"
        "    'tool_name': (required if agent=tool) the tool to execute (e.g., git_clone, pip_install)\n"
        "    'tool_params': (required if agent=tool) parameters dict for the tool\n"
        "Return ONLY valid JSON — no markdown fences, no extra text."
    )

    async def create_plan(
        self,
        task: str,
        memory_context: str,
        repo_context: str,
    ) -> dict:
        prompt = (
            f"Task: {task}\n\n"
            f"Memory context:\n{memory_context}\n\n"
            f"Repo context:\n{repo_context}\n\n"
            "Produce the plan now."
        )
        raw = await llm(prompt, system=self.SYSTEM, agent="planner")
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("Planner returned non-JSON; using fallback plan.")
            return {
                "explanation": raw[:200],
                "steps": [{"agent": "coder", "description": task}],
            }


# ---------------------------------------------------------------------------
# CoderAgent
# ---------------------------------------------------------------------------

class CoderAgent:
    SYSTEM_NEW = (
        "You are an expert software engineer. "
        "Write clean, well-commented, production-ready Python code."
    )
    SYSTEM_DIFF = (
        "You are an expert at producing minimal unified diffs (git diff format). "
        "Return ONLY the diff inside ```diff ... ``` fences — no explanation."
    )

    def _build_prompt(
        self,
        subtask: str,
        context_files: list[str],
        previous_results: list[dict],
        memory: str,
        existing_code: str = "",
    ) -> str:
        ctx  = "\n".join(context_files)
        prev = "\n".join(r.get("output", "")[:300] for r in previous_results)
        if existing_code:
            return (
                f"Existing code:\n```python\n{existing_code}\n```\n\n"
                f"Task: {subtask}\n\n"
                f"Context files:\n{ctx}\n\n"
                f"Memory:\n{memory}\n\n"
                "Generate a unified diff that applies the required changes."
            )
        return (
            f"Task: {subtask}\n\n"
            f"Context files:\n{ctx}\n\n"
            f"Previous results:\n{prev}\n\n"
            f"Memory:\n{memory}\n\n"
            "Write the complete Python implementation."
        )

    async def generate_code(
        self,
        subtask: str,
        context_files: list[str],
        previous_results: list[dict],
        memory: str,
        existing_code: str = "",
    ) -> str:
        if existing_code and len(existing_code) > 100:
            diff_text = await llm(
                self._build_prompt(subtask, context_files, previous_results, memory, existing_code),
                system=self.SYSTEM_DIFF,
                agent="coder",
            )
            return self._apply_diff(existing_code, diff_text)
        prompt = self._build_prompt(subtask, context_files, previous_results, memory)
        return await llm(prompt, system=self.SYSTEM_NEW, agent="coder")

    async def stream_code(
        self,
        subtask: str,
        context_files: list[str],
        previous_results: list[dict],
        memory: str,
        existing_code: str = "",
    ) -> AsyncGenerator[str, None]:
        system = self.SYSTEM_DIFF if existing_code else self.SYSTEM_NEW
        prompt = self._build_prompt(
            subtask, context_files, previous_results, memory, existing_code
        )
        async for token in llm_stream(prompt, system=system, agent="coder"):
            yield token

    @staticmethod
    def _apply_diff(original_code: str, diff_text: str) -> str:
        """Apply a unified diff produced by the LLM to original_code."""
        match = re.search(r"```diff\s*(.*?)```", diff_text, re.DOTALL)
        raw_diff = match.group(1).strip() if match else diff_text.strip()
        if not raw_diff:
            return original_code
        try:
            patch = PatchSet(raw_diff)
            lines = original_code.splitlines(keepends=True)
            for patched_file in patch:
                for hunk in patched_file:
                    new_lines: list[str] = []
                    src_idx = 0
                    for line in hunk:
                        if line.is_context:
                            new_lines.append(lines[src_idx]); src_idx += 1
                        elif line.is_removed:
                            src_idx += 1
                        elif line.is_added:
                            new_lines.append(line.value)
                    start = hunk.source_start - 1
                    end   = start + hunk.source_length
                    lines[start:end] = new_lines
            return "".join(lines)
        except Exception as exc:
            logger.warning("Diff application failed (%s); returning original.", exc)
            return original_code


# ---------------------------------------------------------------------------
# TesterAgent
# ---------------------------------------------------------------------------

class TesterAgent:
    SYSTEM = (
        "You are an expert in pytest and test-driven development. "
        "Generate comprehensive pytest test cases including edge cases and exceptions. "
        "Return ONLY valid Python code — no explanations, no markdown fences."
    )

    async def generate_tests(self, code: str, subtask: str) -> str:
        prompt = (
            f"Code to test:\n```python\n{code}\n```\n\n"
            f"Task description: {subtask}\n\n"
            "Write pytest test cases."
        )
        return await llm(prompt, system=self.SYSTEM, agent="tester")


# ---------------------------------------------------------------------------
# DebuggerAgent
# ---------------------------------------------------------------------------

class DebuggerAgent:
    SYSTEM = (
        "You are an expert debugger. "
        "Given code and an error message, return a fixed version of the code. "
        "Return ONLY the corrected Python code — no explanation."
    )

    async def fix(self, code: str, error: str) -> str:
        prompt = (
            f"Code:\n```python\n{code}\n```\n\n"
            f"Error:\n{error}\n\n"
            "Fix the code."
        )
        return await llm(prompt, system=self.SYSTEM, agent="debugger")


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class CriticAgent:
    SYSTEM = (
        "You are a senior code reviewer. "
        "Evaluate the results of an AI coding pipeline. "
        "Reply with 'PASS' if the output is correct and complete, "
        "or 'FAIL: <reason>' if not."
    )

    async def review(self, results: list[dict], task: str) -> str:
        summary = "\n".join(
            f"[{r.get('type', '?')}] {r.get('step', '')}: "
            f"{str(r.get('output', ''))[:400]}"
            for r in results
        )
        prompt = f"Original task: {task}\n\nPipeline results:\n{summary}"
        return await llm(prompt, system=self.SYSTEM, agent="critic")


# ---------------------------------------------------------------------------
# MemoryAgent
# ---------------------------------------------------------------------------

class MemoryAgent:
    """
    In-process short-term memory store.
    Swap retrieve() for a ChromaDB vector query for persistent cross-session memory.
    """

    def __init__(self) -> None:
        self._store: list[dict] = []

    def store(self, task: str, result: str) -> None:
        self._store.append({"task": task, "result": result})
        self._store = self._store[-20:]       # keep last 20 entries

    def retrieve(self, query: str, top_k: int = 3) -> str:
        query_lower = query.lower()
        hits = [
            e for e in self._store
            if any(w in e["task"].lower() for w in query_lower.split())
        ]
        return "\n".join(e["result"][:300] for e in hits[:top_k])
