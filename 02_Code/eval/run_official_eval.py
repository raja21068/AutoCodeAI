"""
eval/run_official_eval.py
-------------------------
Grades a predictions file with the official SWE-bench harness.

The previous evaluator cloned each repository from GitHub, ran a bare
``pytest``, and decided resolution by regexing ``"N passed"`` out of stdout.
It never applied ``test_patch``, so the FAIL_TO_PASS tests it claimed to
check did not exist in the tree being graded, and it could not run the
repositories whose suites are not pytest-driven.

Nothing in this module reimplements grading. It shells out to
``swebench.harness.run_evaluation``, which builds the instance-specific
environment, applies the test patch, runs the declared FAIL_TO_PASS and
PASS_TO_PASS tests with the correct per-repository log parser, and writes a
report. We read that report and nothing else.

Usage:
    python -m eval.run_official_eval --run_id af_lite_001
    python -m eval.run_official_eval --run_id af_lite_001 --max_workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grade predictions with the official harness")
    p.add_argument("--run_id", required=True)
    p.add_argument("--output_dir", default="eval/results")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--cache_level", default="env",
                   choices=["none", "base", "env", "instance"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.output_dir) / args.run_id
    predictions = run_dir / "predictions.jsonl"

    if not predictions.exists():
        logger.error("No predictions at %s — run eval.swebench_runner first.", predictions)
        return 1

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", args.dataset,
        "--split", args.split,
        "--predictions_path", str(predictions),
        "--run_id", args.run_id,
        "--max_workers", str(args.max_workers),
        "--timeout", str(args.timeout),
        "--cache_level", args.cache_level,
    ]
    logger.info("Running official harness:\n  %s", " ".join(cmd))

    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        logger.error("Harness exited with %d", completed.returncode)
        return completed.returncode

    # The harness writes <model_name_or_path>.<run_id>.json in the CWD.
    reports = sorted(Path.cwd().glob(f"*.{args.run_id}.json"))
    if not reports:
        logger.warning("Harness finished but no report matched *.%s.json", args.run_id)
        return 0

    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    destination = run_dir / "official_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    resolved = report.get("resolved_instances", 0)
    submitted = report.get("submitted_instances", 0) or 1
    print("\n" + "=" * 66)
    print(f"  OFFICIAL SWE-bench report — run_id={args.run_id}")
    print("=" * 66)
    for key in (
        "total_instances", "submitted_instances", "completed_instances",
        "resolved_instances", "unresolved_instances",
        "empty_patch_instances", "error_instances",
    ):
        if key in report:
            print(f"  {key:<24}: {report[key]}")
    print(f"  {'resolve rate':<24}: {resolved / submitted:.1%}")
    print(f"\n  Saved to {destination}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
