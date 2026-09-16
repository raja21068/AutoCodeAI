#!/usr/bin/env python3
"""
experiments/setup_and_run.py
-----------------------------
One-script setup that:
  1. Verifies your environment (LLM config, packages, Docker, image cache)
  2. Estimates cost and wall-clock before spending anything
  3. Runs a small pilot to confirm everything works
  4. Then runs the full benchmark on your go-ahead

Usage:
    python experiments/setup_and_run.py --check        # verify setup
    python experiments/setup_and_run.py --pilot        # 10-instance pilot
    python experiments/setup_and_run.py --full         # full SWE-bench Lite
    python experiments/setup_and_run.py --ablation     # ablation grid

This script only *schedules* prediction runs. It never reports a resolve
rate: resolution comes from the official harness via eval.run_official_eval,
and nothing here is allowed to imply otherwise.
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Color output ─────────────────────────────────────────────
def green(s):  return f"\033[92m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"


# ── Pricing ──────────────────────────────────────────────────
# USD per 1M tokens (input, output). Provider list prices move without
# warning, so these are estimates for planning only — the authoritative
# figure for the paper is the measured cost_usd in run_summary.json, which
# comes from actual usage rather than from this table.
PRICES: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-chat":       (0.27, 1.10),
    "deepseek/deepseek-reasoner":   (0.55, 2.19),
    "deepseek-chat":                (0.27, 1.10),
    "gpt-4o":                       (2.50, 10.00),
    "gpt-4o-mini":                  (0.15, 0.60),
    "anthropic/claude-sonnet-4-5":  (3.00, 15.00),
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
}
UNKNOWN_PRICE = (1.00, 3.00)

# Measured on this benchmark: one instance image is ~4GB, and instances of
# the same repo+version share base and env layers.
GB_PER_IMAGE = 4.0

DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}


def resolved_model(cli_model: str | None) -> str:
    """
    The model that will actually be used.

    ``core.utils.llm._resolve_model`` checks ``LLM_MODEL`` before any
    per-agent routing, so an LLM_MODEL in .env silently overrides every
    per-role default. Reporting the CLI value when the env disagrees would
    make the cost estimate describe a run that is not the one about to
    happen.
    """
    return cli_model or os.getenv("LLM_MODEL") or "deepseek/deepseek-chat"


def provider_key_name(model: str) -> str | None:
    """Env var holding the credential for *model*'s provider."""
    lowered = model.lower()
    for marker, key in (
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("claude", "ANTHROPIC_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("gpt", "OPENAI_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
    ):
        if marker in lowered:
            return key
    return None


# ── Step 1: Environment checks ───────────────────────────────

def check_environment(model: str, probe_api: bool = True,
                      split: str = "lite") -> bool:
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

    # Platform. The official harness imports the POSIX-only `resource`
    # module, so grading cannot run on native Windows at all. Generation
    # would appear to work, which is the trap worth naming here.
    if sys.platform == "win32":
        print(red("  ❌  Native Windows — the official SWE-bench harness "
                  "cannot run here"))
        print("      → Use WSL: bash eval/bootstrap_wsl.sh")
        ok = False
    else:
        print(green(f"  ✅  Platform {sys.platform}"))

    # Packages the evaluation path actually needs.
    for import_name, pkg_name in [
        ("litellm",  "litellm"),
        ("docker",   "docker"),
        ("datasets", "datasets"),
        ("swebench", "swebench"),
        ("unidiff",  "unidiff"),
    ]:
        try:
            __import__(import_name)
            print(green(f"  ✅  {pkg_name}"))
        except ImportError as exc:
            print(red(f"  ❌  {pkg_name} not importable: {exc}"))
            ok = False

    # Docker, through the SDK rather than the CLI: the runner talks to the
    # daemon via the SDK, and docker-py 6.x cannot reach a healthy daemon
    # once requests>=2.32 is installed. A CLI-only check would pass while
    # every run failed.
    try:
        import docker as docker_sdk

        client = docker_sdk.from_env()
        client.ping()
        print(green(f"  ✅  Docker daemon reachable "
                    f"(engine {client.version()['Version']}, "
                    f"sdk {docker_sdk.__version__})"))
    except Exception as exc:
        print(red(f"  ❌  Docker unreachable via SDK: {exc}"))
        print("      → Start the daemon, and ensure docker>=7.1.0 is installed")
        ok = False
        client = None

    # LLM configuration
    mode = os.getenv("LLM_MODE") or os.getenv("LLM_PROVIDER", "litellm")
    print(green(f"  ✅  LLM_MODE={mode}"))
    print(green(f"  ✅  model={model}"))
    if os.getenv("LLM_MODEL") and os.getenv("LLM_MODEL") != model:
        print(yellow(f"  ⚠️   LLM_MODEL={os.getenv('LLM_MODEL')} in the "
                     f"environment overrides per-role defaults"))

    key_name = provider_key_name(model)
    if key_name:
        key = os.getenv(key_name, "")
        if key:
            print(green(f"  ✅  {key_name} set ({len(key)} chars)"))
        else:
            print(red(f"  ❌  {key_name} not set"))
            print(f"      → Add it to .env: {key_name}=...")
            ok = False
    else:
        print(yellow(f"  ⚠️   Could not infer the credential for {model}"))

    # A live one-token call. Key presence is not key validity — a rejected
    # key looks identical to a working one until the first request, and
    # finding that out after an image pull and a full preflight is the
    # expensive way to learn it.
    if probe_api and ok:
        try:
            import logging

            import litellm

            # The probe is one line of output; LiteLLM's INFO logs are four.
            logging.getLogger("LiteLLM").setLevel(logging.WARNING)
            litellm.suppress_debug_info = True

            litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            print(green("  ✅  Provider authenticated (1-token probe)"))
        except Exception as exc:
            print(red(f"  ❌  Provider rejected the request: "
                      f"{type(exc).__name__}"))
            print(f"      {str(exc)[:200]}")
            ok = False

    # Instance image cache. Pulls dominate wall-clock, so how many images are
    # already local is the single best predictor of how long a run takes.
    if client is not None:
        try:
            cached, total = image_cache_status(client, split)
            missing = total - cached
            msg = f"instance images cached: {cached}/{total}"
            if missing:
                print(yellow(f"  ⚠️   {msg} — {missing} to pull "
                             f"(~{missing * GB_PER_IMAGE:.0f} GB before "
                             f"layer sharing)"))
                print("      → python -m eval.prefetch_images "
                      f"--split {split}")
            else:
                print(green(f"  ✅  {msg}"))
        except Exception as exc:
            print(yellow(f"  ⚠️   Could not inspect image cache: {exc}"))

    print()
    if ok:
        print(green(bold("  ✅  All checks passed — ready to run experiments!\n")))
    else:
        print(red(bold("  ❌  Fix the issues above before running experiments.\n")))

    return ok


def image_cache_status(client, split: str) -> tuple[int, int]:
    """Return ``(cached, total)`` instance images for *split*."""
    from datasets import load_dataset

    from eval.instance_env import instance_image_key

    rows = [dict(r) for r in load_dataset(DATASETS[split], split="test")]
    wanted = {instance_image_key(r) for r in rows}
    local = set()
    for image in client.images.list():
        local.update(image.tags or [])
    return len(wanted & local), len(wanted)


# ── Step 2: Cost and wall-clock estimation ───────────────────

def estimate_cost(n_tasks: int, model: str) -> None:
    """
    Print a planning estimate for *n_tasks* instances under *model*.

    Deliberately not a promise. The per-role token figures below are rough
    averages over SWE-bench Lite; the number that goes in the paper is the
    measured total in run_summary.json.
    """
    tokens_per_task = {
        "planner":  {"in": 2_000, "out": 500},
        "coder":    {"in": 4_000, "out": 2_000},
        "tester":   {"in": 3_000, "out": 1_500},
        "debugger": {"in": 4_000, "out": 2_000},   # ~40% of tasks need debug
        "critic":   {"in": 3_000, "out": 300},
    }

    price_in, price_out = PRICES.get(model, UNKNOWN_PRICE)
    known = model in PRICES

    total_in = total_out = 0.0
    for agent, t in tokens_per_task.items():
        multiplier = 0.4 if agent == "debugger" else 1.0
        total_in += t["in"] * multiplier
        total_out += t["out"] * multiplier

    cost_per_task = (total_in * price_in + total_out * price_out) / 1_000_000
    total_cost = cost_per_task * n_tasks

    print(bold(f"\n  Estimate for {n_tasks} instances ({model}):"))
    print(f"    Input tokens/task  : ~{int(total_in):,}")
    print(f"    Output tokens/task : ~{int(total_out):,}")
    print(f"    Cost/task          : ~${cost_per_task:.4f}")
    print(f"    Total API cost     : ~${total_cost:.2f}")
    if not known:
        print(yellow(f"    ⚠️   No price on file for {model}; used "
                     f"${UNKNOWN_PRICE[0]}/${UNKNOWN_PRICE[1]} per 1M as a "
                     f"placeholder"))
    print(f"    Estimates move ±30%; the reported figure is the measured "
          f"cost in run_summary.json\n")


def estimate_wall_clock(n_missing_images: int, mbps: float | None = None) -> None:
    """Print the pull time that usually dominates a first run."""
    if n_missing_images <= 0:
        return
    gb = n_missing_images * GB_PER_IMAGE
    print(bold("  Image pulls (usually the dominant cost):"))
    print(f"    Images to pull     : {n_missing_images}")
    print(f"    Upper-bound volume : ~{gb:.0f} GB before layer sharing")
    if mbps:
        hours = (gb * 1000) / (mbps * 3600)
        print(f"    At {mbps:.1f} MB/s      : ~{hours:.0f} h")
    print("    → Prefetch separately so the agent loop is not "
          "blocked on the network\n")


# ── Step 3: Instance selection ───────────────────────────────

def ensure_subset(n: int, split: str, seed: int | None = None) -> Path:
    """
    Build (or reuse) a deterministic stratified subset of *n* instances.

    This replaces ``--max_tasks N``. SWE-bench splits are grouped by
    repository, so a prefix of N is not a sample: django is ~114 of Lite's
    300 and sorts early, so ``--max_tasks 50`` would be almost entirely
    django. eval.sample_subset allocates across repositories proportionally
    and records the seed, so the selection is reproducible and checkable by
    a reviewer.
    """
    out = REPO_ROOT / "eval" / "subsets" / f"{split}{n}.json"
    if out.exists():
        print(f"  Using existing subset {out}")
        return out

    cmd = [sys.executable, "-m", "eval.sample_subset",
           "--n", str(n), "--split", split, "--out", str(out)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return out


def confirm(prompt: str) -> bool:
    return input(f"  {prompt} [y/N] ").strip().lower() == "y"


# ── Step 4: Runs ─────────────────────────────────────────────

def run_prediction(run_id: str, model: str, split: str,
                   instance_file: Path | None, output_dir: str,
                   config: str = "agentforge") -> int:
    cmd = [
        sys.executable, "-m", "eval.swebench_runner",
        "--split", split, "--config", config,
        "--model", model, "--run_id", run_id,
        "--output_dir", output_dir, "--resume",
    ]
    if instance_file is not None:
        cmd += ["--instance_file", str(instance_file)]
    print(bold(f"\n  {' '.join(cmd)}\n"))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(red("\n  Prediction run failed; see the log above.\n"))
        return result.returncode

    print(bold("\n  Predictions written. NOT GRADED — resolution comes from "
               "the official harness:\n"))
    print(f"      python -m eval.run_official_eval --run_id {run_id}\n")
    return 0


# ── CLI entry point ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AgentForge setup and experiment driver")
    p.add_argument("--check",    action="store_true", help="Check environment only")
    p.add_argument("--pilot",    action="store_true", help="Run a small pilot")
    p.add_argument("--full",     action="store_true", help="Run the full split")
    p.add_argument("--ablation", action="store_true", help="Run the ablation grid")
    p.add_argument("--model",    default=None,
                   help="Model for ALL roles (default: $LLM_MODEL, else "
                        "deepseek/deepseek-chat)")
    p.add_argument("--split",    default="lite", choices=sorted(DATASETS))
    p.add_argument("--tasks",    type=int, help="Instance count (stratified subset)")
    p.add_argument("--output_dir", default="experiments/results")
    p.add_argument("--no-api-probe", action="store_true",
                   help="Skip the 1-token provider auth check")
    return p.parse_args()


def main():
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    args = parse_args()
    model = resolved_model(args.model)
    probe = not args.no_api_probe

    # Bare invocation, or --check: report and stop.
    if args.check or not any([args.pilot, args.full, args.ablation]):
        check_environment(model, probe_api=probe, split=args.split)
        if not args.check:
            estimate_cost(300, model)
        return

    if not check_environment(model, probe_api=probe, split=args.split):
        print(red("  Refusing to start a run with a failing environment.\n"))
        sys.exit(1)

    if args.pilot:
        n = args.tasks or 10
        estimate_cost(n, model)
        subset = ensure_subset(n, args.split)
        if confirm(f"Run {n}-instance pilot?"):
            run_prediction(f"pilot_{n}", model, args.split, subset,
                           args.output_dir)

    if args.full:
        n = args.tasks
        estimate_cost(n or 300, model)
        subset = ensure_subset(n, args.split) if n else None
        label = f"{n}-instance" if n else "full"
        if confirm(f"Run {label} evaluation?"):
            run_prediction(f"full_{args.split}", model, args.split, subset,
                           args.output_dir)

    if args.ablation:
        n = args.tasks or 100
        from eval.configs import GRID

        estimate_cost(n * len(GRID), model)
        subset = ensure_subset(n, args.split)
        if confirm(f"Run ablation ({n} instances × {len(GRID)} conditions)?"):
            cmd = [sys.executable, "-m", "eval.ablation_runner",
                   "--run_prefix", "ablation", "--model", model,
                   "--split", args.split,
                   "--instance_file", str(subset), "--resume", "--grade"]
            print(bold(f"\n  {' '.join(cmd)}\n"))
            subprocess.run(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
