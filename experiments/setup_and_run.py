#!/usr/bin/env python3
"""
experiments/setup_and_run.py
-----------------------------
One-script setup that:
  1. Verifies your environment (API key, packages, SWE-bench)
  2. Estimates cost before spending anything
  3. Runs a 10-task pilot to confirm everything works
  4. Then runs the full benchmark on your go-ahead

Usage:
    python experiments/setup_and_run.py --check        # just verify setup
    python experiments/setup_and_run.py --pilot        # 10 tasks, ~$2
    python experiments/setup_and_run.py --full         # full SWE-bench Lite
    python experiments/setup_and_run.py --ablation     # ablation study
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ── Color output ─────────────────────────────────────────────
def green(s):  return f"\033[92m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"


# ── Step 1: Environment checks ───────────────────────────────

def check_environment() -> bool:
    print(bold("\n╔══════════════════════════════════════════════╗"))
    print(bold("║  AgentForge — Environment Check              ║"))
    print(bold("╚══════════════════════════════════════════════╝\n"))

    ok = True

    # Python version
    v = sys.version_info
    if v >= (3, 11):
        print(green(f"  ✅  Python {v.major}.{v.minor}.{v.micro}"))
    else:
        print(red(f"  ❌  Python {v.major}.{v.minor} — need 3.11+"))
        ok = False

    # OpenAI API key
    key = os.getenv("OPENAI_API_KEY", "")
    if key.startswith("sk-") and len(key) > 20:
        print(green(f"  ✅  OPENAI_API_KEY set ({key[:8]}…)"))
    else:
        print(red("  ❌  OPENAI_API_KEY not set or invalid"))
        print("      → Add it to your .env file: OPENAI_API_KEY=sk-...")
        ok = False

    # Required packages
    packages = [
        ("openai",      "openai"),
        ("fastapi",     "fastapi"),
        ("chromadb",    "chromadb"),
        ("docker",      "docker"),
        ("watchdog",    "watchdog"),
        ("unidiff",     "unidiff"),
        ("datasets",    "datasets"),
        ("swebench",    "swebench"),
    ]
    for import_name, pkg_name in packages:
        try:
            __import__(import_name)
            print(green(f"  ✅  {pkg_name}"))
        except ImportError:
            print(red(f"  ❌  {pkg_name} not installed"))
            print(f"      → pip install {pkg_name}")
            ok = False

    # Docker
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            print(green("  ✅  Docker daemon running"))
        else:
            print(red("  ❌  Docker not running"))
            print("      → Start Docker Desktop or: sudo systemctl start docker")
            ok = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(red("  ❌  Docker not found"))
        ok = False

    # ChromaDB (try to connect)
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = os.getenv("CHROMA_PORT", "8001")
    try:
        import chromadb
        client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))
        client.heartbeat()
        print(green(f"  ✅  ChromaDB at {chroma_host}:{chroma_port}"))
    except Exception:
        print(yellow(f"  ⚠️   ChromaDB not reachable at {chroma_host}:{chroma_port}"))
        print("      → Run: docker compose up chromadb -d")
        print("      (will fall back to in-process store — OK for pilot)")

    # OpenAI connectivity test
    if key.startswith("sk-"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            resp = client.models.list()
            models = [m.id for m in resp.data if "gpt" in m.id][:3]
            print(green(f"  ✅  OpenAI API reachable — models: {models}"))
        except Exception as e:
            print(red(f"  ❌  OpenAI API error: {e}"))
            ok = False

    print()
    if ok:
        print(green(bold("  ✅  All checks passed — ready to run experiments!\n")))
    else:
        print(red(bold("  ❌  Fix the issues above before running experiments.\n")))

    return ok


# ── Step 2: Cost estimation ──────────────────────────────────

def estimate_cost(n_tasks: int, model: str = "gpt-4o") -> None:
    # Rough token estimates per task (prompt + completion)
    # Based on SWE-bench Lite average problem length + agent calls
    tokens_per_task = {
        "planner":  {"in": 2_000,  "out": 500},
        "coder":    {"in": 4_000,  "out": 2_000},
        "tester":   {"in": 3_000,  "out": 1_500},
        "debugger": {"in": 4_000,  "out": 2_000},   # ~40% of tasks need debug
        "critic":   {"in": 3_000,  "out": 300},
    }

    # GPT-4o pricing (per 1M tokens, as of 2024)
    price_in  = 5.00   # $5 per 1M input tokens
    price_out = 15.00  # $15 per 1M output tokens

    total_in = total_out = 0
    for agent, t in tokens_per_task.items():
        multiplier = 0.4 if agent == "debugger" else 1.0
        total_in  += t["in"]  * multiplier
        total_out += t["out"] * multiplier

    cost_per_task = (total_in * price_in + total_out * price_out) / 1_000_000
    total_cost    = cost_per_task * n_tasks

    print(bold(f"\n  Cost estimate for {n_tasks} tasks ({model}):"))
    print(f"    Input tokens/task  : ~{int(total_in):,}")
    print(f"    Output tokens/task : ~{int(total_out):,}")
    print(f"    Cost/task          : ~${cost_per_task:.3f}")
    print(f"    Total estimate     : ~${total_cost:.2f}")
    print(f"    {'⚠️  ' if total_cost > 50 else '✅  '}This is an estimate — actual cost may vary ±30%\n")


# ── Step 3: Pilot run ────────────────────────────────────────

async def run_pilot(n_tasks: int = 10, output_dir: str = "experiments/results/pilot") -> None:
    print(bold(f"\n  Running {n_tasks}-task pilot…\n"))

    try:
        from datasets import load_dataset
    except ImportError:
        print(red("  ❌  datasets not installed: pip install datasets"))
        return

    from eval.swebench_runner import run_task
    from eval.task_adapter    import SWEBenchTaskAdapter
    from eval.metrics         import compute_metrics
    from services.orchestrator import Orchestrator

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    tasks   = list(dataset)[:n_tasks]

    orch    = Orchestrator()
    adapter = SWEBenchTaskAdapter()
    records = []

    for i, task in enumerate(tasks, 1):
        print(f"  [{i:2}/{n_tasks}] {task['instance_id']}")
        record = await run_task(task, orch, adapter, out_dir, resume=True)
        records.append(record)
        status = green("RESOLVED") if record["resolved"] else yellow("not resolved")
        print(f"         → {status}  ({record['elapsed_s']:.1f}s)")

    orch.shutdown()

    metrics = compute_metrics(records)
    print(bold(f"\n  Pilot complete — {metrics['resolved']}/{metrics['total']} resolved "
               f"({metrics['resolve_rate']:.1%})"))
    (out_dir / "summary.json").write_text(json.dumps({"metrics": metrics}, indent=2))
    print(f"  Results saved to {out_dir}/\n")


# ── CLI entry point ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--check",    action="store_true", help="Check environment only")
    p.add_argument("--pilot",    action="store_true", help="Run 10-task pilot (~$2)")
    p.add_argument("--full",     action="store_true", help="Run full SWE-bench Lite (300 tasks, ~$80)")
    p.add_argument("--ablation", action="store_true", help="Run ablation study (~$180)")
    p.add_argument("--model",    default="gpt-4o",    help="OpenAI model")
    p.add_argument("--tasks",    type=int,            help="Override task count")
    return p.parse_args()


def main():
    from dotenv import load_dotenv
    load_dotenv()

    args = parse_args()

    if args.check or not any([args.pilot, args.full, args.ablation]):
        ok = check_environment()
        if not args.check:
            estimate_cost(300, args.model)
        return

    if args.pilot:
        check_environment()
        n = args.tasks or 10
        estimate_cost(n, args.model)
        confirm = input(f"  Run {n}-task pilot? [y/N] ").strip().lower()
        if confirm == "y":
            asyncio.run(run_pilot(n))

    if args.full:
        check_environment()
        n = args.tasks or 300
        estimate_cost(n, args.model)
        confirm = input(f"  Run {n}-task full evaluation? [y/N] ").strip().lower()
        if confirm == "y":
            cmd = [sys.executable, "-m", "eval.swebench_runner",
                   "--split", "lite", "--max_tasks", str(n),
                   "--model", args.model,
                   "--output_dir", "experiments/results/full"]
            subprocess.run(cmd)

    if args.ablation:
        n = args.tasks or 100
        estimate_cost(n * 6, args.model)
        confirm = input(f"  Run ablation ({n} tasks × 6 conditions)? [y/N] ").strip().lower()
        if confirm == "y":
            cmd = [sys.executable, "-m", "eval.ablation_runner",
                   "--max_tasks", str(n),
                   "--output_dir", "experiments/results/ablation"]
            subprocess.run(cmd)


if __name__ == "__main__":
    main()
