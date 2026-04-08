"""
eval/smoke_test.py
------------------
Quick sanity check — runs AgentForge on 3 hardcoded toy tasks
before you burn API credits on the full SWE-bench run.

Usage:
    python -m eval.smoke_test
"""

import asyncio
import logging

from services.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)

SMOKE_TASKS = [
    {
        "task": "Write a Python function `add(a, b)` that returns the sum of two numbers. Include type hints and a docstring.",
        "context_files": [],
        "expect_pass": True,
    },
    {
        "task": "Write a Python function `is_palindrome(s: str) -> bool` that returns True if the string is a palindrome, ignoring case and spaces.",
        "context_files": [],
        "expect_pass": True,
    },
    {
        "task": "Write a Python class `Stack` with methods `push(item)`, `pop()`, `peek()`, and `is_empty()`. Raise `IndexError` on pop/peek of empty stack.",
        "context_files": [],
        "expect_pass": True,
    },
]


async def main() -> None:
    orch   = Orchestrator()
    passed = 0

    for i, smoke in enumerate(SMOKE_TASKS, 1):
        print(f"\n{'='*60}")
        print(f"  Smoke test {i}: {smoke['task'][:60]}…")
        print("=" * 60)

        output = []
        try:
            async for chunk in orch.run_streaming(
                smoke["task"], smoke["context_files"]
            ):
                print(chunk, end="", flush=True)
                output.append(chunk)
            passed += 1
            print(f"\n✅ Task {i} completed")
        except Exception as exc:
            print(f"\n❌ Task {i} failed: {exc}")

    orch.shutdown()
    print(f"\n{'='*60}")
    print(f"  Smoke tests: {passed}/{len(SMOKE_TASKS)} passed")
    print("=" * 60)

    if passed == len(SMOKE_TASKS):
        print("\n✅ Ready to run full SWE-bench evaluation.")
        print("   python -m eval.swebench_runner --split lite --max_tasks 50")
    else:
        print("\n⚠️  Fix failures above before running full evaluation.")


if __name__ == "__main__":
    asyncio.run(main())
