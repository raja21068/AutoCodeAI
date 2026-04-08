"""
services/orchestrator.py — Central pipeline coordinator.

Wires together: memory retrieval → planning → agent execution loop
→ sandbox → critic review → memory persistence → streaming output.

Supports:
- Tool execution (git, pip, shell commands)
- Parallel agent execution
- Streaming output
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, Awaitable, Callable
from collections import defaultdict

from core.agents.agents import (
    CoderAgent,
    CriticAgent,
    DebuggerAgent,
    MemoryAgent,
    PlannerAgent,
    TesterAgent,
)
from core.tools.sandbox import DockerSandbox
from core.tools.tool_executor import ToolExecutor
from memory.repo_indexer import RepoIndexer

logger = logging.getLogger(__name__)

Callback = Callable[[str], Awaitable[None]] | None


class Orchestrator:
    def __init__(self, repo_path: str | None = None) -> None:
        self.planner  = PlannerAgent()
        self.coder    = CoderAgent()
        self.tester   = TesterAgent()
        self.debugger = DebuggerAgent()
        self.critic   = CriticAgent()
        self.memory   = MemoryAgent()
        self.sandbox  = DockerSandbox()
        self.tool_executor = ToolExecutor(cwd=repo_path or ".")
        self.indexer  = RepoIndexer(repo_path) if repo_path else None
        if self.indexer:
            self.indexer.start_watching()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_file_content(filepath: str) -> str:
        if not filepath:
            return ""
        try:
            return Path(filepath).read_text(encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            logger.warning("File not found: %s", filepath)
            return ""

    async def _notify(self, callback: Callback, msg: str) -> None:
        if callback:
            await callback(msg)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        context_files: list[str],
        callback: Callback = None,
    ) -> list[dict]:
        await self._notify(callback, "🧠 Retrieving memory …\n")
        memory_context = self.memory.retrieve(task)

        if self.indexer:
            snippets = self.indexer.retrieve_relevant(task)
            repo_ctx = "\n".join(
                s["metadata"]["content"][:300] for s in snippets
            )
            memory_context += f"\nRepo context:\n{repo_ctx}"

        await self._notify(callback, "📝 Creating plan …\n")
        plan = await self.planner.create_plan(task, memory_context, "")
        await self._notify(callback, f"✅ Plan: {plan.get('explanation', '')}\n\n")

        # Group steps by parallel_group for concurrent execution
        steps = plan.get("steps", [])
        groups = defaultdict(list)
        for step in steps:
            group = step.get("parallel_group", 0)
            groups[group].append(step)

        results: list[dict] = []
        latest_code = ""

        # Execute each group (groups with same number run in parallel)
        for group_num in sorted(groups.keys()):
            group_steps = groups[group_num]
            
            if len(group_steps) == 1:
                # Single step - execute sequentially
                step_result = await self._execute_step(
                    group_steps[0], context_files, results, memory_context, latest_code, callback
                )
                if step_result.get("code"):
                    latest_code = step_result["code"]
                results.append(step_result)
            else:
                # Multiple steps - execute in parallel
                await self._notify(callback, f"⚡ Running {len(group_steps)} steps in parallel…\n")
                parallel_results = await asyncio.gather(
                    *[self._execute_step(s, context_files, results, memory_context, latest_code, callback) 
                      for s in group_steps],
                    return_exceptions=True
                )
                for r in parallel_results:
                    if isinstance(r, Exception):
                        logger.error(f"Parallel execution error: {r}")
                        results.append({"error": str(r), "type": "error"})
                    else:
                        if r.get("code"):
                            latest_code = r["code"]
                        results.append(r)

        # final critic pass
        final_review = await self.critic.review(results, task)
        await self._notify(callback, f"\n📋 Final review: {final_review}\n")
        if "PASS" in final_review and latest_code:
            self.memory.store(task, latest_code)

        await self._notify(callback, "\n✅ Done.\n")
        return results
    
    async def _execute_step(
        self,
        step: dict,
        context_files: list[str],
        results: list[dict],
        memory_context: str,
        latest_code: str,
        callback: Callback
    ) -> dict:
        """Execute a single step (agent or tool)."""
        agent = step.get("agent", "")
        desc = step.get("description", "")
        
        await self._notify(callback, f"⚙️  {agent.capitalize()}: {desc}\n")

        # Tool execution
        if agent == "tool":
            tool_name = step.get("tool_name")
            tool_params = step.get("tool_params", {})
            if not tool_name:
                return {"step": desc, "error": "No tool_name specified", "type": "tool"}
            
            result = self.tool_executor.execute(tool_name, tool_params)
            output = result.get("stdout", "") or result.get("stderr", "") or result.get("error", "")
            await self._notify(callback, f"🔧 Tool {tool_name}:\n{output[:800]}\n")
            
            # Store tool output in memory for future steps
            self.memory.store(f"tool_{tool_name}", output)
            return {"step": desc, "output": output, "type": "tool", "tool": tool_name}

        # Coder agent
        if agent == "coder":
            existing = self._get_file_content(step.get("file", ""))
            code = await self._stream_coder(
                desc, context_files, results, memory_context, existing, callback
            )
            return {"step": desc, "output": code, "type": "code", "code": code}

        # Tester agent
        elif agent == "tester" and latest_code:
            test_code = await self.tester.generate_tests(latest_code, desc)
            stdout, stderr = self.sandbox.run_code(latest_code, test_code)
            output = stdout + (f"\nSTDERR:\n{stderr}" if stderr else "")
            await self._notify(callback, f"🧪 Tests:\n{output[:800]}\n")

            # auto-debug on failure
            if stderr or "FAILED" in output or "ERROR" in output:
                await self._notify(callback, "🔧 Debugging …\n")
                latest_code = await self.debugger.fix(latest_code, output)
                return {
                    "step": desc, 
                    "output": output, 
                    "type": "test",
                    "code": latest_code,
                    "debug": "auto-debugged"
                }
            return {"step": desc, "output": output, "type": "test"}

        # Debugger agent
        elif agent == "debugger" and latest_code:
            error_ctx = results[-1]["output"] if results else ""
            code = await self.debugger.fix(latest_code, error_ctx)
            return {"step": desc, "output": code, "type": "code", "code": code}

        # Critic agent
        elif agent == "critic":
            review = await self.critic.review(results, desc)
            await self._notify(callback, f"📋 Review: {review}\n")
            return {"step": desc, "output": review, "type": "review"}

        return {"step": desc, "output": "Unknown agent", "type": "unknown"}

    async def _stream_coder(
        self,
        subtask: str,
        context_files: list[str],
        results: list[dict],
        memory: str,
        existing_code: str,
        callback: Callback,
    ) -> str:
        full_code = ""
        async for token in self.coder.stream_code(
            subtask, context_files, results, memory, existing_code
        ):
            full_code += token
            await self._notify(callback, token)
        return full_code

    # ------------------------------------------------------------------
    # Streaming generator (for SSE / WebSocket)
    # ------------------------------------------------------------------

    async def run_streaming(
        self,
        task: str,
        context_files: list[str],
    ) -> AsyncGenerator[str, None]:
        """Yield string chunks suitable for SSE or WebSocket streaming."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _cb(msg: str) -> None:
            await queue.put(msg)

        async def _worker() -> None:
            try:
                await self.run(task, context_files, callback=_cb)
            finally:
                await queue.put(None)           # sentinel

        worker = asyncio.create_task(_worker())
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
        await worker

    # ------------------------------------------------------------------
    # Parallel execution helper
    # ------------------------------------------------------------------

    async def run_parallel(
        self,
        steps: list[dict],
        context_files: list[str],
    ) -> list:
        """Execute multiple steps in parallel."""
        async def _exec(step: dict):
            if step["agent"] == "coder":
                return await self.coder.generate_code(
                    step["description"], context_files, [], "", ""
                )
            if step["agent"] == "tester":
                return await self.tester.generate_tests("", step["description"])
            if step["agent"] == "tool":
                tool_name = step.get("tool_name")
                tool_params = step.get("tool_params", {})
                return self.tool_executor.execute(tool_name, tool_params)
            return None

        return await asyncio.gather(
            *[_exec(s) for s in steps], return_exceptions=True
        )
    
    async def run_parallel_coders(
        self, 
        subtasks: list[str], 
        context_files: list[str], 
        memory: str
    ) -> list[str]:
        """Run multiple coder agents in parallel for independent subtasks."""
        async def _code_one(subtask: str):
            return await self.coder.generate_code(subtask, context_files, [], memory, "")
        
        return await asyncio.gather(*[_code_one(s) for s in subtasks])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self.indexer:
            self.indexer.stop()
