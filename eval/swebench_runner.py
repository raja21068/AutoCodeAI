"""
eval/swebench_runner.py
-----------------------
Runs AgentForge against SWE-bench Lite and logs every result.

Usage:
    python -m eval.swebench_runner \
        --split lite \
        --max_tasks 50 \
        --model gpt-4o \
        --output_dir eval/results/run_001

SWE-bench Lite = 300 real GitHub issues from 11 popular repos.
Full SWE-bench  = 2294 issues (use --split full for ablations).

Install SWE-bench first:
    pip install swebench
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

from eval.task_adapter import SWEBenchTaskAdapter
from eval.metrics import compute_metrics
from services.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentForge SWE-bench evaluation")
    p.add_argument("--split",       default="lite",     choices=["lite", "full"],
                   help="SWE-bench split to evaluate on")
    p.add_argument("--max_tasks",   type=int, default=None,
                   help="Cap number of tasks (useful for quick runs)")
    p.add_argument("--model",       default="gpt-4o",
                   help="OpenAI model to use")
    p.add_argument("--output_dir",  default="eval/results",
                   help="Directory to write per-task logs and summary")
    p.add_argument("--resume",      action="store_true",
                   help="Skip tasks that already have a result file")
    p.add_argument("--retry",       type=int, default=3,
                   help="Max debug retries per task")
    p.add_argument("--workers",     type=int, default=1,
                   help="Parallel tasks (be careful with rate limits)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_task(
    task: dict,
    orchestrator: Orchestrator,
    adapter: "SWEBenchTaskAdapter",
    output_dir: Path,
    resume: bool,
) -> dict:
    instance_id = task["instance_id"]
    result_path = output_dir / f"{instance_id}.json"

    if resume and result_path.exists():
        logger.info("Skipping %s (already completed)", instance_id)
        return json.loads(result_path.read_text())

    logger.info("Running task: %s", instance_id)
    start = time.time()

    # Build natural language task description from SWE-bench fields
    nl_task       = adapter.to_natural_language(task)
    context_files = adapter.get_context_files(task)

    messages: list[str] = []
    result_code  = None
    error        = None

    try:
        async for chunk in orchestrator.run_streaming(nl_task, context_files):
            messages.append(chunk)

        # Extract the last code block from the message stream
        result_code = adapter.extract_code(messages)

    except Exception as exc:
        logger.exception("Task %s failed with exception", instance_id)
        error = str(exc)

    elapsed = time.time() - start

    # Evaluate the patch against the SWE-bench oracle
    evaluation = adapter.evaluate(task, result_code) if result_code else {
        "resolved": False,
        "patch_applied": False,
        "tests_passed": 0,
        "tests_failed": 0,
    }

    record = {
        "instance_id":   instance_id,
        "repo":          task.get("repo", ""),
        "task":          nl_task[:300],
        "resolved":      evaluation["resolved"],
        "patch_applied": evaluation["patch_applied"],
        "tests_passed":  evaluation["tests_passed"],
        "tests_failed":  evaluation["tests_failed"],
        "elapsed_s":     round(elapsed, 2),
        "error":         error,
        "model":         os.getenv("OPENAI_MODEL", "gpt-4o"),
        "timestamp":     datetime.utcnow().isoformat(),
    }

    result_path.write_text(json.dumps(record, indent=2))
    logger.info(
        "Task %s → resolved=%s  elapsed=%.1fs",
        instance_id, record["resolved"], elapsed,
    )
    return record


async def main() -> None:
    args = parse_args()
    os.environ["OPENAI_MODEL"] = args.model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SWE-bench dataset
    dataset_name = (
        "princeton-nlp/SWE-bench_Lite"
        if args.split == "lite"
        else "princeton-nlp/SWE-bench"
    )
    logger.info("Loading %s …", dataset_name)
    dataset = load_dataset(dataset_name, split="test")
    tasks   = list(dataset)

    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    logger.info("Running %d tasks with model=%s", len(tasks), args.model)

    orchestrator = Orchestrator()
    adapter      = SWEBenchTaskAdapter()

    # Run tasks (sequential by default, parallel if --workers > 1)
    semaphore = asyncio.Semaphore(args.workers)

    async def run_with_semaphore(task):
        async with semaphore:
            return await run_task(
                task, orchestrator, adapter, output_dir, args.resume
            )

    records = await asyncio.gather(
        *[run_with_semaphore(t) for t in tasks],
        return_exceptions=True,
    )

    # Filter out exceptions
    valid_records = [r for r in records if isinstance(r, dict)]

    # Compute and save summary metrics
    metrics      = compute_metrics(valid_records)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "run_config": vars(args),
        "metrics":    metrics,
        "timestamp":  datetime.utcnow().isoformat(),
    }, indent=2))

    # Print results table
    print("\n" + "=" * 60)
    print(f"  AgentForge — SWE-bench {args.split.upper()} Results")
    print("=" * 60)
    print(f"  Tasks evaluated : {metrics['total']}")
    print(f"  Resolved        : {metrics['resolved']} "
          f"({metrics['resolve_rate']:.1%})")
    print(f"  Patch applied   : {metrics['patch_applied']} "
          f"({metrics['patch_rate']:.1%})")
    print(f"  Avg elapsed     : {metrics['avg_elapsed_s']:.1f}s / task")
    print(f"  Results saved   : {output_dir}")
    print("=" * 60)

    orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
