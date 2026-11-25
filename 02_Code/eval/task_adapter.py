"""
eval/task_adapter.py
--------------------
Builds the agent-visible projection of a SWE-bench instance and extracts
the model patch from an agent run.

Information boundary
--------------------
Repository-level issue resolution includes *localizing* the defect. Any
signal derived from the reference patch — including the set of files it
touches — is oracle information and inflates measured performance.

This module therefore defines a hard allowlist. ``to_agent_input()`` reads
only ``ALLOWED_FIELDS`` and raises if a forbidden field is ever consulted.
The previous implementation mined ``.py`` paths out of
``problem_statement + patch`` and passed them to the Coder as context; that
path is removed, and ``tests/test_no_leakage.py`` guards against its return.

Grading is *not* performed here. Correctness is decided exclusively by the
official harness (see :mod:`eval.run_official_eval`).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Kept as a module constant so the prompt and the environment cannot drift.
WORKDIR_HINT = "/testbed"

# Fields an agent may see.
ALLOWED_FIELDS = frozenset({"instance_id", "repo", "base_commit", "problem_statement"})

# Fields that must never reach an agent, a prompt, or a retrieval index.
FORBIDDEN_FIELDS = frozenset(
    {
        "patch",           # the reference fix
        "test_patch",      # introduces the graded tests
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "environment_setup_commit",
    }
)


class LeakageError(RuntimeError):
    """Raised when oracle information would enter the agent-visible input."""


class SWEBenchTaskAdapter:
    """Converts an instance into agent input and reads a patch back out."""

    # ------------------------------------------------------------------
    # Input projection
    # ------------------------------------------------------------------

    def to_agent_input(self, instance: dict, *, include_hints: bool = False) -> dict:
        """
        Return the only dict an agent is permitted to see.

        ``hints_text`` is off by default: on many instances it contains
        maintainer replies naming the offending file or the fix itself, which
        is a softer form of the same localization leak. Enable it only for an
        explicitly labelled ablation.
        """
        projection = {k: instance.get(k, "") for k in ALLOWED_FIELDS}

        problem = (projection["problem_statement"] or "").strip()
        if not problem:
            raise LeakageError(
                f"{projection['instance_id']}: empty problem statement — refusing to "
                "fall back to any patch-derived signal"
            )

        prompt = (
            f"Resolve the following issue in the repository `{projection['repo']}`.\n"
            f"The repository is checked out at `{WORKDIR_HINT}` at commit "
            f"{projection['base_commit'][:8]}.\n\n"
            f"## Issue\n{problem}\n"
        )

        if include_hints:
            hints = (instance.get("hints_text") or "").strip()
            if hints:
                prompt += f"\n## Issue thread\n{hints}\n"

        prompt += (
            "\n## Instructions\n"
            "- Locate the responsible code yourself by reading and searching the repository.\n"
            "- Make the smallest change that fixes the issue.\n"
            "- Do not modify or add test files.\n"
            "- Run commands to check your work; the exit status and output are your feedback.\n"
        )

        projection["prompt"] = prompt
        return projection

    # ------------------------------------------------------------------
    # Output extraction
    # ------------------------------------------------------------------

    def extract_patch(self, env) -> str:
        """
        Return the model patch as a real ``git diff`` against the base commit.

        The patch is read from the repository state, not parsed out of model
        prose. An agent that edited files has, by construction, produced a
        well-formed diff; an agent that only talked about editing produces an
        empty one.
        """
        return env.diff()

    @staticmethod
    def extract_diff_block(text: str) -> str | None:
        """
        Pull a unified diff out of a model message, for the edit step only.

        This is how a proposed edit is read *before* it is applied to the
        repository. It plays no part in grading.
        """
        fenced = re.search(r"```(?:diff|patch)\s*\n(.*?)```", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        if re.search(r"^(?:---|diff --git) ", text, re.MULTILINE):
            return text.strip()

        return None


def _diff_body_lines(diff: str) -> list[str]:
    """Substantive added/removed lines of a diff, ignoring headers and noise."""
    lines = []
    for raw in (diff or "").splitlines():
        if raw.startswith(("+++", "---", "@@", "diff --git", "index ")):
            continue
        if raw[:1] in "+-":
            body = raw[1:].strip()
            # Short or boilerplate lines collide by chance; require substance.
            if len(body) >= 24:
                lines.append(body)
    return lines


def assert_no_leakage(payload, instance: dict, *, where: str = "agent input") -> None:
    """
    Fail loudly if oracle content from *instance* appears in *payload*.

    Checks forbidden **values**, not field names — a prompt is allowed to use
    the word "patch", but it must not contain a line from the reference fix,
    a graded test identifier, or any part of the test patch.

    Called on every prompt before it is sent. Cheap, and it turns a silent
    validity bug into a crash.
    """
    blob = payload if isinstance(payload, str) else repr(payload)

    for field in ("patch", "test_patch"):
        for line in _diff_body_lines(instance.get(field, "")):
            if line in blob:
                raise LeakageError(
                    f"{instance.get('instance_id')}: line from {field!r} found in "
                    f"{where}: {line[:80]!r}"
                )

    for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        tests = instance.get(field) or []
        if isinstance(tests, str):
            try:
                import json

                tests = json.loads(tests)
            except Exception:
                tests = [tests]
        for test_id in tests:
            if test_id and str(test_id) in blob:
                raise LeakageError(
                    f"{instance.get('instance_id')}: graded test id from {field!r} "
                    f"found in {where}: {test_id!r}"
                )
