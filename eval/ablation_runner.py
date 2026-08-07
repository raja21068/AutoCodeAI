"""
eval/ablation_runner.py
-----------------------
Drives the full experimental grid through the corrected prediction pipeline.

This file previously contained a second, independent path into the benchmark
that reproduced every defect of the old main runner: it derived context files
from the reference patch, graded with a local routine instead of the official
harness, and reused one orchestrator across all tasks in a condition so that
each instance conditioned on its predecessors. It also implemented role
removal by monkey-patching an agent to a no-op, which left the removed role's
calls in place and changed spend as well as structure.

Role removal is now a declared property of :class:`RepairConfig`
(``disabled_roles``), documented there, and every condition runs through the
same leakage-free runner and the same official grader as the headline result.

This module only *schedules* runs. It does not grade. Grade with:

    python -m eval.run_official_eval --run_id <run_id>

Usage:
    python -m eval.ablation_runner --run_prefix af_lite --model gpt-4o
    python -m eval.ablation_runner --run_prefix af_lite --only agentforge single_forced
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from eval.configs import GRID

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the AgentForge experimental grid")
    p.add_argument("--run_prefix", required=True,
                   help="run_id prefix; each condition becomes <prefix>_<condition>")
    p.add_argument("--model", default="deepseek/deepseek-chat")
    p.add_argument("--split", default="lite")
    p.add_argument("--only", nargs="*", default=None,
                   help="subset of conditions to run (default: all)")
    p.add_argument("--instance_file", default=None,
                   help="subset file from eval.sample_subset, forwarded to "
                        "every condition so the grid shares one instance list")
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--grade", action="store_true",
                   help="invoke the official harness after each condition")
    # Budget flags are forwarded unchanged to every condition, which is what
    # makes the grid a controlled comparison.
    p.add_argument("--max_iterations", type=int, default=8)
    p.add_argument("--max_executions", type=int, default=12)
    p.add_argument("--max_cost_usd", type=float, default=1.50)
    p.add_argument("--max_wall_clock_s", type=float, default=900.0)
    p.add_argument("--dry_run", action="store_true", help="print commands only")
    return p.parse_args()


def build_command(condition: str, run_id: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "-m", "eval.swebench_runner",
        "--split", args.split,
        "--config", condition,
        "--model", args.model,
        "--run_id", run_id,
        "--workers", str(args.workers),
        "--max_iterations", str(args.max_iterations),
        "--max_executions", str(args.max_executions),
        "--max_cost_usd", str(args.max_cost_usd),
        "--max_wall_clock_s", str(args.max_wall_clock_s),
    ]
    if args.instance_file:
        cmd += ["--instance_file", args.instance_file]
    elif args.max_tasks:
        cmd += ["--max_tasks", str(args.max_tasks)]
    if args.resume:
        cmd.append("--resume")
    return cmd


def main() -> int:
    args = parse_args()
    conditions = args.only or GRID

    unknown = set(conditions) - set(GRID)
    if unknown:
        logger.error("Unknown conditions: %s", ", ".join(sorted(unknown)))
        return 2

    logger.info("Grid: %s", ", ".join(conditions))
    logger.info("Shared budget: %d iters, %d execs, $%.2f, %.0fs",
                args.max_iterations, args.max_executions,
                args.max_cost_usd, args.max_wall_clock_s)

    completed: list[str] = []
    for condition in conditions:
        run_id = f"{args.run_prefix}_{condition}"
        cmd = build_command(condition, run_id, args)

        if args.dry_run:
            print(" ".join(cmd))
            continue

        logger.info("── %s ──", condition)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error("Condition %s failed (exit %d); continuing",
                         condition, result.returncode)
            continue
        completed.append(run_id)

        if args.grade:
            subprocess.run([
                sys.executable, "-m", "eval.run_official_eval", "--run_id", run_id
            ])

    if completed and not args.dry_run:
        print("\nGrid complete. Analyse with:\n")
        print(f"  python -m eval.analyze_results --runs {' '.join(completed)} \\")
        print(f"      --baseline {args.run_prefix}_single_optional --latex\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
