"""
eval/task_adapter.py
--------------------
Converts SWE-bench task dicts into AgentForge inputs and evaluates
generated patches against the oracle test suite.

SWE-bench task fields used:
    instance_id     — unique identifier  e.g. "django__django-11099"
    repo            — GitHub repo slug   e.g. "django/django"
    base_commit     — commit SHA of the repo state before the fix
    problem_statement — natural language description of the bug
    hints_text      — optional hints from the issue thread
    patch           — gold standard unified diff (for reference only)
    test_patch      — test file diff that passes only after the fix
    PASS_TO_PASS    — tests that should keep passing
    FAIL_TO_PASS    — tests that should flip from fail to pass
"""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class SWEBenchTaskAdapter:

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    def to_natural_language(self, task: dict) -> str:
        """
        Convert a SWE-bench task dict into a clear natural language
        task description suitable for AgentForge's Planner.
        """
        repo     = task.get("repo", "unknown/repo")
        problem  = task.get("problem_statement", "").strip()
        hints    = task.get("hints_text", "").strip()
        commit   = task.get("base_commit", "")[:8]

        nl = (
            f"Fix the following bug in the GitHub repository `{repo}` "
            f"(base commit: {commit}).\n\n"
            f"## Problem\n{problem}\n"
        )
        if hints:
            nl += f"\n## Hints\n{hints}\n"

        nl += (
            "\n## Instructions\n"
            "- Identify the root cause of the bug.\n"
            "- Produce a minimal code fix — change only what is necessary.\n"
            "- Do not modify test files.\n"
            "- Output a unified diff that applies cleanly to the repository.\n"
        )
        return nl

    def get_context_files(self, task: dict) -> list[str]:
        """
        Extract file paths mentioned in the problem statement or patch
        to pass as context hints to the Coder agent.
        """
        text  = task.get("problem_statement", "") + task.get("patch", "")
        paths = re.findall(r"[a-zA-Z0-9_/\-]+\.py", text)
        # deduplicate, keep order
        seen, unique = set(), []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique[:5]     # cap at 5 files

    # ------------------------------------------------------------------
    # Output extraction
    # ------------------------------------------------------------------

    def extract_code(self, messages: list[str]) -> str | None:
        """
        Extract the generated patch from the agent's message stream.
        Looks for a unified diff block first, then any code block.
        """
        full_text = "".join(messages)

        # 1. Explicit diff fence
        diff_match = re.search(
            r"```diff\s*(.*?)```", full_text, re.DOTALL
        )
        if diff_match:
            return diff_match.group(1).strip()

        # 2. Diff-like content without fences
        if re.search(r"^--- .+\n\+\+\+ .+\n@@", full_text, re.MULTILINE):
            return full_text.strip()

        # 3. Any fenced code block
        code_match = re.search(
            r"```(?:python)?\s*(.*?)```", full_text, re.DOTALL
        )
        if code_match:
            return code_match.group(1).strip()

        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, task: dict, generated_patch: str) -> dict:
        """
        Apply the generated patch to a clean checkout of the repo and
        run the SWE-bench test suite against it.

        Returns a dict with:
            resolved      — bool: FAIL_TO_PASS tests all pass
            patch_applied — bool: patch applied without conflicts
            tests_passed  — int: number of passing tests
            tests_failed  — int: number of failing tests
        """
        repo        = task.get("repo", "")
        base_commit = task.get("base_commit", "")
        fail_to_pass = task.get("FAIL_TO_PASS", [])
        pass_to_pass = task.get("PASS_TO_PASS", [])

        result = {
            "resolved":      False,
            "patch_applied": False,
            "tests_passed":  0,
            "tests_failed":  0,
        }

        if not generated_patch:
            return result

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Clone repo at the base commit
                repo_url = f"https://github.com/{repo}.git"
                subprocess.run(
                    ["git", "clone", "--quiet", repo_url, tmpdir],
                    check=True, capture_output=True, timeout=120,
                )
                subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=tmpdir, check=True, capture_output=True, timeout=30,
                )

                # Write and apply the patch
                patch_file = Path(tmpdir) / "agent.patch"
                patch_file.write_text(generated_patch)
                apply = subprocess.run(
                    ["git", "apply", "--check", str(patch_file)],
                    cwd=tmpdir, capture_output=True, timeout=30,
                )
                if apply.returncode != 0:
                    logger.warning("Patch does not apply cleanly.")
                    return result

                subprocess.run(
                    ["git", "apply", str(patch_file)],
                    cwd=tmpdir, check=True, capture_output=True, timeout=30,
                )
                result["patch_applied"] = True

                # Install the project
                subprocess.run(
                    ["pip", "install", "-e", ".", "--quiet"],
                    cwd=tmpdir, capture_output=True, timeout=180,
                )

                # Run FAIL_TO_PASS tests (these must now pass)
                all_tests  = (fail_to_pass or []) + (pass_to_pass or [])
                if not all_tests:
                    return result

                test_run = subprocess.run(
                    ["python", "-m", "pytest", "--tb=no", "-q"] + all_tests,
                    cwd=tmpdir, capture_output=True,
                    timeout=300, text=True,
                )
                stdout = test_run.stdout

                passed = len(re.findall(r"\d+ passed", stdout))
                failed = len(re.findall(r"\d+ failed", stdout))

                # Parse actual counts
                m_pass = re.search(r"(\d+) passed", stdout)
                m_fail = re.search(r"(\d+) failed", stdout)
                result["tests_passed"] = int(m_pass.group(1)) if m_pass else 0
                result["tests_failed"] = int(m_fail.group(1)) if m_fail else 0

                # Resolved = all FAIL_TO_PASS tests now pass
                # and no PASS_TO_PASS tests regressed
                result["resolved"] = (
                    result["tests_failed"] == 0
                    and result["tests_passed"] >= len(fail_to_pass or [])
                )

            except subprocess.TimeoutExpired:
                logger.warning("Evaluation timed out for %s", task.get("instance_id"))
            except subprocess.CalledProcessError as exc:
                logger.warning("Subprocess error: %s", exc)
            except Exception as exc:
                logger.exception("Evaluation error: %s", exc)

        return result
