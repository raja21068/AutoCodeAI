"""
eval/validate_generated_tests.py
--------------------------------
Measures whether the Tester agent's reproduction scripts actually discriminate
between a buggy and a fixed repository.

The paper treats agent-generated tests as diagnostic evidence rather than
correctness evidence. That is a claim about their quality, and it needs a
number. For each instance this script:

    1. builds the leakage-free agent input and asks the Tester for a
       reproduction script — exactly the prompt used during repair;
    2. runs the script on the base checkout, where the bug is present;
    3. applies the reference patch and runs it again, where the bug is fixed.

A script is **discriminating** when it fails on the buggy tree and passes on
the fixed one. Anything else is reported separately rather than folded in:
scripts that pass on both (vacuous), fail on both (usually broken imports or
a wrong reproduction), or never ran (infrastructure failure).

On the information boundary
---------------------------
This script reads the reference patch. That is legitimate and necessary — it
is measurement code, not agent code, and it applies the patch only *after*
the reproduction script has been generated and run on the buggy tree. The
Tester itself receives only ``to_agent_input()``, and every prompt passes
through ``assert_no_leakage``.

Run it on a **disjoint** split, never on the split used for headline results,
or the measurement becomes a form of evaluation-set tuning:

    python -m eval.validate_generated_tests --split verified --n 50 \\
        --exclude_split lite --model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from core.utils.llm import llm_call
from eval.instance_env import InstanceEnv
from eval.task_adapter import SWEBenchTaskAdapter, assert_no_leakage
from services.repair_orchestrator import TESTER_SYS, _strip_fences
from services.repo_retrieval import RepoRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_PATH = "repro_validation.py"

DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}

# Markers that mean the script never exercised the bug at all.
INFRA_MARKERS = (
    "ModuleNotFoundError", "ImportError", "SyntaxError",
    "IndentationError", "command not found", "No such file",
)


def classify(buggy_code: int, buggy_out: str,
             fixed_code: int, fixed_out: str) -> str:
    """Label one instance from the two runs."""
    if _is_infra(buggy_out) or _is_infra(fixed_out):
        return "infrastructure_failure"
    if buggy_code != 0 and fixed_code == 0:
        return "discriminating"
    if buggy_code == 0 and fixed_code == 0:
        return "vacuous"          # passes even with the bug present
    if buggy_code != 0 and fixed_code != 0:
        return "always_fails"     # wrong reproduction, or breaks on both
    return "inverted"             # passes buggy, fails fixed — actively misleading


def _is_infra(output: str) -> bool:
    return any(marker in (output or "") for marker in INFRA_MARKERS)


async def validate_one(instance: dict, model: str, use_retrieval: bool) -> dict:
    adapter = SWEBenchTaskAdapter()
    task = adapter.to_agent_input(instance)
    assert_no_leakage(task["prompt"], instance, where="tester validation prompt")

    record: dict = {
        "instance_id": instance["instance_id"],
        "repo": instance.get("repo", ""),
    }

    with InstanceEnv(instance) as env:
        context = ""
        if use_retrieval:
            retriever = RepoRetriever(env)
            context = retriever.format(retriever.retrieve(task["problem_statement"]))
            assert_no_leakage(context, instance, where="tester validation context")

        prompt = task["prompt"]
        if context:
            prompt += f"\n\n## Context\n{context}"

        result = await llm_call(prompt, system=TESTER_SYS, agent="tester",
                                model=model)
        script = _strip_fences(result.text)
        record["cost_usd"] = round(result.cost_usd, 6)
        record["script_chars"] = len(script)

        if not script.strip():
            record["classification"] = "no_script"
            return record

        env.write_file(SCRIPT_PATH, script)

        # Phase 1 — buggy tree.
        buggy_code, buggy_out, buggy_err = env.exec(f"python {SCRIPT_PATH}")
        record["buggy_exit"] = buggy_code

        # Phase 2 — apply the reference patch, rerun. Measurement only; this
        # happens after the script exists and after the buggy run.
        applied, message = env.apply_patch(instance["patch"])
        if not applied:
            record["classification"] = "gold_patch_failed"
            record["detail"] = message
            return record

        fixed_code, fixed_out, fixed_err = env.exec(f"python {SCRIPT_PATH}")
        record["fixed_exit"] = fixed_code

        record["classification"] = classify(
            buggy_code, buggy_out + buggy_err,
            fixed_code, fixed_out + fixed_err,
        )

    return record


async def main_async(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    instances = [dict(row) for row in load_dataset(DATASETS[args.split], split="test")]

    if args.exclude_split:
        excluded = {
            row["instance_id"]
            for row in load_dataset(DATASETS[args.exclude_split], split="test")
        }
        before = len(instances)
        instances = [i for i in instances if i["instance_id"] not in excluded]
        logger.info("Excluded %d instances present in the %s split",
                    before - len(instances), args.exclude_split)

    if args.n:
        from eval.sample_subset import stratified_sample

        keep = set(stratified_sample(instances, args.n, args.seed))
        instances = [i for i in instances if i["instance_id"] in keep]

    logger.info("Validating generated tests on %d instances", len(instances))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.workers)

    async def guarded(instance: dict) -> dict:
        async with semaphore:
            try:
                return await validate_one(instance, args.model, args.use_retrieval)
            except Exception as exc:
                logger.warning("%s failed: %s", instance["instance_id"], exc)
                return {
                    "instance_id": instance["instance_id"],
                    "repo": instance.get("repo", ""),
                    "classification": "error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }

    records = await asyncio.gather(*[guarded(i) for i in instances])

    counts = Counter(r["classification"] for r in records)
    n = len(records) or 1
    discriminating = counts["discriminating"]

    summary = {
        "split": args.split,
        "excluded_split": args.exclude_split,
        "model": args.model,
        "n": len(records),
        "seed": args.seed,
        "counts": dict(sorted(counts.items())),
        "discriminating_rate": round(discriminating / n, 4),
        "fails_on_buggy_rate": round(
            sum(1 for r in records if r.get("buggy_exit", 0) != 0) / n, 4),
        "passes_on_fixed_rate": round(
            sum(1 for r in records if r.get("fixed_exit") == 0) / n, 4),
        "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in records), 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    (out_dir / "generated_test_validation.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 62)
    print("  Generated-test validation")
    print("=" * 62)
    print(f"  split              : {args.split}"
          + (f" (minus {args.exclude_split})" if args.exclude_split else ""))
    print(f"  instances          : {summary['n']}")
    print(f"  fails on buggy     : {summary['fails_on_buggy_rate']:.1%}")
    print(f"  passes on fixed    : {summary['passes_on_fixed_rate']:.1%}")
    print(f"  DISCRIMINATING     : {summary['discriminating_rate']:.1%}")
    print("  " + "-" * 58)
    for label, count in sorted(counts.items()):
        print(f"    {label:<24}: {count:>4}  ({count / n:.1%})")
    print(f"\n  Written to {out_dir / 'generated_test_validation.json'}")
    print("=" * 62 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate agent-generated reproduction scripts")
    parser.add_argument("--split", default="verified", choices=sorted(DATASETS))
    parser.add_argument("--exclude_split", default="lite",
                        choices=sorted(DATASETS),
                        help="hold out the evaluation split so this measurement "
                             "stays disjoint from reported results")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--use_retrieval", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="give the Tester the same retrieved context it "
                             "sees during repair (--no-use_retrieval to omit)")
    parser.add_argument("--output_dir", default="eval/results/test_validation")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
