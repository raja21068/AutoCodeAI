"""
tests/test_no_leakage.py — Executable proof of the information boundary.

Reviewer 1's central objection was that the public implementation derived
agent context from the reference patch. These tests are the standing answer:
they fail if that behaviour, or anything equivalent to it, comes back.

Run:
    python -m pytest tests/test_no_leakage.py -v
"""

from __future__ import annotations

import inspect

import pytest

from eval import task_adapter
from eval.task_adapter import (
    ALLOWED_FIELDS,
    FORBIDDEN_FIELDS,
    LeakageError,
    SWEBenchTaskAdapter,
    assert_no_leakage,
)


@pytest.fixture
def instance() -> dict:
    """A SWE-bench-shaped instance whose oracle fields are distinctive."""
    return {
        "instance_id": "acme__widget-4242",
        "repo": "acme/widget",
        "base_commit": "a1b2c3d4e5f6a7b8c9d0",
        "problem_statement": (
            "Calling Widget.resize() with a negative width raises IndexError "
            "instead of ValueError. Expected a clear validation error."
        ),
        "hints_text": "Probably in src/acme/widget/geometry.py, the resize helper.",
        "patch": (
            "diff --git a/src/acme/widget/geometry.py b/src/acme/widget/geometry.py\n"
            "--- a/src/acme/widget/geometry.py\n"
            "+++ b/src/acme/widget/geometry.py\n"
            "@@ -10,6 +10,8 @@\n"
            " def resize(self, width, height):\n"
            "+        if width < 0:\n"
            "+            raise ValueError('width must be non-negative')\n"
            "         return self._apply_dimensions(width, height)\n"
        ),
        "test_patch": (
            "diff --git a/tests/test_geometry.py b/tests/test_geometry.py\n"
            "--- a/tests/test_geometry.py\n"
            "+++ b/tests/test_geometry.py\n"
            "@@ -1,3 +1,7 @@\n"
            "+def test_negative_width_raises_value_error():\n"
            "+    with pytest.raises(ValueError):\n"
            "+        Widget().resize(-1, 10)\n"
        ),
        "FAIL_TO_PASS": ["tests/test_geometry.py::test_negative_width_raises_value_error"],
        "PASS_TO_PASS": ["tests/test_geometry.py::test_basic_resize"],
    }


# ---------------------------------------------------------------------------
# The projection itself
# ---------------------------------------------------------------------------


def test_agent_input_contains_no_oracle_content(instance):
    payload = SWEBenchTaskAdapter().to_agent_input(instance)
    blob = repr(payload)

    assert "raise ValueError('width must be non-negative')" not in blob
    assert "test_negative_width_raises_value_error" not in blob
    assert "tests/test_geometry.py" not in blob
    assert "geometry.py" not in blob, (
        "the reference patch's file path reached the agent — this is the "
        "oracle localization leak"
    )


def test_projection_keys_are_allowlisted(instance):
    payload = SWEBenchTaskAdapter().to_agent_input(instance)
    extra = set(payload) - ALLOWED_FIELDS - {"prompt"}
    assert not extra, f"unexpected fields in agent input: {extra}"


def test_hints_are_off_by_default(instance):
    """hints_text often names the offending file — a softer form of the leak."""
    default = SWEBenchTaskAdapter().to_agent_input(instance)
    assert "geometry.py" not in default["prompt"]

    opted_in = SWEBenchTaskAdapter().to_agent_input(instance, include_hints=True)
    assert "geometry.py" in opted_in["prompt"], (
        "the opt-in path should be the only way hint text is ever included"
    )


# ---------------------------------------------------------------------------
# The runtime assertion
# ---------------------------------------------------------------------------


def test_assert_no_leakage_catches_patch_content(instance):
    poisoned = "Here is some context: raise ValueError('width must be non-negative')"
    with pytest.raises(LeakageError, match="patch"):
        assert_no_leakage(poisoned, instance)


def test_assert_no_leakage_catches_graded_test_ids(instance):
    poisoned = "Make sure tests/test_geometry.py::test_negative_width_raises_value_error passes"
    with pytest.raises(LeakageError):
        assert_no_leakage(poisoned, instance)


def test_assert_no_leakage_allows_the_word_patch(instance):
    """The boundary is about values, not vocabulary."""
    benign = "Return a unified diff. Do not modify test files. Apply the patch cleanly."
    assert_no_leakage(benign, instance)


def test_assert_no_leakage_passes_on_the_real_projection(instance):
    payload = SWEBenchTaskAdapter().to_agent_input(instance)
    assert_no_leakage(payload["prompt"], instance)


# ---------------------------------------------------------------------------
# Regressions of the specific defects the reviewers found
# ---------------------------------------------------------------------------


def test_get_context_files_is_gone():
    """
    The removed method built context by regexing paths out of
    ``problem_statement + patch``. It must not return under any name.
    """
    assert not hasattr(SWEBenchTaskAdapter, "get_context_files")

    source = inspect.getsource(task_adapter)
    assert 'task.get("patch"' not in source
    assert "instance['patch']" not in source
    assert 'instance.get("patch")' not in source or "assert_no_leakage" in source


def test_adapter_does_not_grade():
    """
    Correctness is the official harness's job. A local ``evaluate`` was how
    the unofficial grader crept in, so the adapter must not expose one.
    """
    assert not hasattr(SWEBenchTaskAdapter, "evaluate")


def test_forbidden_fields_cover_the_oracle_surface():
    for field in ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"):
        assert field in FORBIDDEN_FIELDS
    assert not (ALLOWED_FIELDS & FORBIDDEN_FIELDS)
