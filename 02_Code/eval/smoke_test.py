"""
eval/smoke_test.py
------------------
Preflight for the benchmark pipeline. Run this before spending on 300 tasks.

The previous smoke test asked the agent to write ``add(a, b)`` and a ``Stack``
class. Those exercise nothing the benchmark depends on: no instance image, no
repository checkout, no patch application, no official harness. Passing them
told you the LLM key worked and little else.

This version checks the parts that actually break, in the order they break,
and stops at the first failure:

    1. dependencies importable
    2. Docker reachable
    3. leakage and invariant test suites pass
    4. instance image resolvable and container starts
    5. repository is at the base commit and the project environment works
    6. edits are captured by ``git diff`` and ``reset`` restores a clean tree
    7. optionally, one real end-to-end instance

Usage:
    python -m eval.smoke_test                  # steps 1-6, no API spend
    python -m eval.smoke_test --live           # adds step 7 (one instance)
"""

from __future__ import annotations

import argparse
import subprocess
import sys

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
SAMPLE_INSTANCE = "astropy__astropy-12907"


def ok(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET}  {msg}")


def fail(msg: str, hint: str = "") -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")
    if hint:
        print(f"        -> {hint}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_imports() -> bool:
    missing = []
    for module, package in [
        ("docker", "docker"), ("datasets", "datasets"),
        ("swebench", "swebench"), ("litellm", "litellm"),
    ]:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        fail(f"missing packages: {', '.join(missing)}",
             f"pip install {' '.join(missing)}")
        return False
    ok("dependencies importable")
    return True


def check_docker() -> bool:
    try:
        import docker

        docker.from_env().ping()
    except Exception as exc:
        fail(f"Docker unreachable: {exc}", "start Docker and retry")
        return False
    ok("Docker reachable")
    return True


def check_test_suites() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/test_no_leakage.py",
         "tests/test_execution_invariant.py",
         "tests/test_ablation_semantics.py"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        fail("leakage / invariant tests failed", result.stdout[-600:])
        return False
    ok("leakage and invariant test suites pass")
    return True


def load_instance() -> dict | None:
    from datasets import load_dataset

    for row in load_dataset("princeton-nlp/SWE-bench_Lite", split="test"):
        if row["instance_id"] == SAMPLE_INSTANCE:
            return dict(row)
    return None


def check_environment(instance: dict) -> bool:
    from eval.instance_env import InstanceEnv, instance_image_key

    ok(f"instance image resolved: {instance_image_key(instance)}")
    print("        (first run pulls ~1GB; this may take a few minutes)")

    try:
        with InstanceEnv(instance) as env:
            _, out, _ = env.exec("git rev-parse HEAD")
            head = out.strip()
            if not head.startswith(instance["base_commit"][:8]):
                fail(f"repo at {head[:8]}, expected {instance['base_commit'][:8]}")
                return False
            ok(f"repository checked out at base commit {head[:8]}")

            code, out, err = env.exec("python -c \"import sys; print(sys.version)\"")
            if code != 0:
                fail(f"project environment unusable: {err.strip()}")
                return False
            ok(f"project environment works: python {out.strip().split()[0]}")

            env.exec("echo '# smoke' >> setup.py")
            if "setup.py" not in env.diff():
                fail("git diff did not capture a working-tree edit",
                     "patch extraction would silently return empty patches")
                return False
            ok("edits are captured by git diff")

            env.reset()
            _, status, _ = env.exec("git status --porcelain")
            if status.strip():
                fail(f"reset left the tree dirty: {status.strip()[:60]}")
                return False
            ok("reset restores a clean tree")
        return True
    except Exception as exc:
        fail(f"instance environment failed: {type(exc).__name__}: {exc}")
        return False


def check_live(model: str) -> bool:
    print(f"\n  Running one real instance ({SAMPLE_INSTANCE}) with {model}…")
    result = subprocess.run([
        sys.executable, "-m", "eval.swebench_runner",
        "--split", "lite", "--config", "agentforge", "--model", model,
        "--run_id", "smoke", "--instance_ids", SAMPLE_INSTANCE,
    ])
    if result.returncode != 0:
        fail("end-to-end run failed")
        return False
    ok("end-to-end prediction run completed")
    print("\n  Grade it:  python -m eval.run_official_eval --run_id smoke\n")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark pipeline preflight")
    parser.add_argument("--live", action="store_true",
                        help="also run one real instance (costs API credit)")
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    args = parser.parse_args()

    print("\n  AgentForge pipeline preflight\n  " + "-" * 40)

    if not check_imports():
        return 1
    if not check_docker():
        return 1
    if not check_test_suites():
        return 1

    instance = load_instance()
    if instance is None:
        fail(f"{SAMPLE_INSTANCE} not found in SWE-bench Lite")
        return 1
    if not check_environment(instance):
        return 1

    if args.live and not check_live(args.model):
        return 1

    print("  " + "-" * 40)
    print(f"  {GREEN}Preflight passed.{RESET}")
    if not args.live:
        print("  Add --live to run one real instance end to end.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
