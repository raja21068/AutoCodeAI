"""
eval/swebench_runner.py
-----------------------
Generates SWE-bench predictions with AgentForge. It does **not** grade them.

Grading is delegated in full to the official harness; see
:mod:`eval.run_official_eval`. This separation is the point: the process that
runs the agent never touches FAIL_TO_PASS, PASS_TO_PASS, ``patch`` or
``test_patch``, so it cannot leak them, and the process that grades never
runs an agent.

Isolation guarantees, one per instance:
    * a fresh container from the official instance image;
    * a fresh ``RepairOrchestrator`` — no state, memory or retrieved context
      survives from any earlier instance;
    * a fresh :class:`Budget`.

Usage:
    python -m eval.swebench_runner \
        --split lite \
        --config agentforge \
        --model gpt-4o \
        --run_id af_lite_001

Then grade:
    python -m eval.run_official_eval --run_id af_lite_001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from eval.configs import CONFIGS, agent_kind
from eval.instance_env import InstanceEnv
from services.repair_orchestrator import RepairConfig, RepairOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentForge SWE-bench prediction run")
    p.add_argument("--split", default="lite", choices=["lite", "verified", "full"])
    p.add_argument("--config", default="agentforge", choices=sorted(CONFIGS))
    p.add_argument("--model", default="deepseek/deepseek-chat",
                   help="Single model used for ALL roles, so comparisons are "
                        "not confounded by per-agent routing")
    p.add_argument("--run_id", required=True, help="Names the output directory")
    p.add_argument("--output_dir", default="eval/results")
    p.add_argument("--instance_ids", nargs="*", default=None)
    p.add_argument("--instance_file", default=None,
                   help="subset file from eval.sample_subset; preferred over "
                        "--max_tasks, which takes a biased prefix")
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    # Budget controls, held identical across compared configurations.
    p.add_argument("--max_iterations", type=int, default=8)
    p.add_argument("--max_executions", type=int, default=12)
    p.add_argument("--max_cost_usd", type=float, default=1.50)
    p.add_argument("--max_wall_clock_s", type=float, default=900.0)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> RepairConfig:
    return RepairConfig(
        model=args.model,
        max_iterations=args.max_iterations,
        max_executions=args.max_executions,
        max_cost_usd=args.max_cost_usd,
        max_wall_clock_s=args.max_wall_clock_s,
        **CONFIGS[args.config],
    )


def jsonable(value):
    """
    Coerce a config value into something ``json.dumps`` accepts.

    ``RepairConfig.disabled_roles`` is a frozenset, which json rejects. That
    mattered more than it looks: the summary is written *after* the run
    completes, so one unserializable field discarded the whole budget ledger
    — the evidence for the matched-budget comparison — once the expensive
    work had already been paid for. Sets are emitted sorted so two runs of
    the same config produce byte-identical summaries.
    """
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def make_agent(kind: str, cfg: RepairConfig):
    """Fresh agent per instance — no state survives from any earlier task."""
    if kind == "react":
        from experiments.react_baseline import ReactBaseline

        return ReactBaseline(cfg)
    return RepairOrchestrator(cfg)


async def run_instance(instance: dict, cfg: RepairConfig, kind: str,
                       out_dir: Path, resume: bool) -> dict:
    instance_id = instance["instance_id"]
    record_path = out_dir / "trajectories" / f"{instance_id}.json"

    if resume and record_path.exists():
        logger.info("Skipping %s (already complete)", instance_id)
        return json.loads(record_path.read_text(encoding="utf-8"))

    logger.info("Running %s", instance_id)
    started = time.time()

    record: dict
    try:
        if kind == "single_call":
            # No tools and no repository, so no container is provisioned.
            from experiments.react_baseline import run_single_call_baseline

            record = await run_single_call_baseline(instance, cfg)
        else:
            with InstanceEnv(instance, timeout=cfg.exec_timeout_s) as env:
                record = await make_agent(kind, cfg).run(instance, env)
                record["env_executions"] = env.exec_count
    except Exception as exc:
        logger.exception("Instance %s failed", instance_id)
        record = {
            "instance_id": instance_id,
            "model_patch": "",
            "stop_reason": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "trajectory": [],
            "budget": {},
        }

    record["repo"] = instance.get("repo", "")
    record["wall_clock_s"] = round(time.time() - started, 2)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()

    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info(
        "%s → patch=%s chars, stop=%s, %.0fs",
        instance_id, len(record.get("model_patch") or ""),
        record.get("stop_reason"), record["wall_clock_s"],
    )
    return record


async def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    out_dir = Path(args.output_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }[args.split]
    logger.info("Loading %s …", dataset_name)
    instances = list(load_dataset(dataset_name, split="test"))

    wanted: set[str] = set(args.instance_ids or [])
    if args.instance_file:
        from eval.sample_subset import load_instance_ids

        wanted |= set(load_instance_ids(args.instance_file))
    if wanted:
        instances = [i for i in instances if i["instance_id"] in wanted]
        missing = wanted - {i["instance_id"] for i in instances}
        if missing:
            logger.warning("%d requested instances not in split: %s",
                           len(missing), ", ".join(sorted(missing)[:5]))
    elif args.max_tasks:
        # A prefix over-represents whichever repositories sort first. Kept for
        # quick checks; use --instance_file for anything reported.
        logger.warning("--max_tasks takes a biased prefix; prefer "
                       "eval.sample_subset for reported runs")
        instances = instances[: args.max_tasks]

    logger.info("%d instances | config=%s | model=%s",
                len(instances), args.config, args.model)

    semaphore = asyncio.Semaphore(args.workers)

    kind = agent_kind(args.config)

    async def guarded(instance: dict) -> dict:
        async with semaphore:
            return await run_instance(instance, cfg, kind, out_dir, args.resume)

    records = await asyncio.gather(
        *[guarded(i) for i in instances], return_exceptions=True
    )
    valid = [r for r in records if isinstance(r, dict)]

    # Official prediction format.
    model_name = f"agentforge-{args.config}-{args.model}"
    predictions_path = out_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for record in valid:
            fh.write(json.dumps({
                "instance_id": record["instance_id"],
                "model_name_or_path": model_name,
                "model_patch": record.get("model_patch") or "",
            }) + "\n")

    # Resource ledger for equal-budget analysis.
    def total(key: str) -> float:
        return sum(r.get("budget", {}).get(key, 0) or 0 for r in valid)

    summary = {
        "run_id": args.run_id,
        "config": args.config,
        "model": args.model,
        "split": args.split,
        "model_name_or_path": model_name,
        "instances": len(valid),
        "nonempty_patches": sum(1 for r in valid if (r.get("model_patch") or "").strip()),
        "errors": sum(1 for r in valid if r.get("error")),
        "budget_config": jsonable(vars(cfg)),
        "totals": {
            "total_tokens": total("total_tokens"),
            "cost_usd": round(total("cost_usd"), 4),
            "llm_calls": total("llm_calls"),
            "executions": total("executions"),
            "wall_clock_s": round(sum(r.get("wall_clock_s", 0) for r in valid), 1),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 66)
    print(f"  Predictions written: {predictions_path}")
    print(f"  Instances          : {summary['instances']}")
    print(f"  Non-empty patches  : {summary['nonempty_patches']}")
    print(f"  Total cost         : ${summary['totals']['cost_usd']}")
    print(f"  Executions         : {summary['totals']['executions']}")
    print("\n  NOT GRADED. Resolution rate comes from the official harness:")
    print(f"    python -m eval.run_official_eval --run_id {args.run_id}")
    print("=" * 66)


if __name__ == "__main__":
    asyncio.run(main())
