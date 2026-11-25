"""
tests/test_instance_image_naming.py — Pin the instance image name.

A wrong image name is the most expensive silent failure in the pipeline. It
does not raise anywhere near the mistake: the run proceeds, every instance
fails at pull time, and the official harness records the lot as
``error_instances``. Read quickly, that looks like a model that solved
nothing rather than a run that never started.

Two things are pinned here:

  1. The fallback naming matches the convention in
     ``TestSpec.instance_image_key`` exactly — namespaced, lowercased, with
     the ``__`` separator swapped across the whole string.

  2. When swebench is importable, its own resolution agrees with the
     fallback. This is the check that would have caught the original bug,
     where the preferred path returned an unpullable local name because
     ``namespace`` was left at its ``None`` default.

Run:
    python -m pytest tests/test_instance_image_naming.py -v
"""

from __future__ import annotations

import pytest

from eval.instance_env import IMAGE_NAMESPACE, instance_image_key

# Minimal shape: only the fields image resolution is allowed to read.
LITE_INSTANCE = {
    "instance_id": "astropy__astropy-12907",
    "repo": "astropy/astropy",
    "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
    "version": "4.3",
    "test_patch": "",
    "problem_statement": "",
}

EXPECTED = "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"


def test_fallback_matches_documented_convention(monkeypatch):
    """With swebench unavailable, the fallback still yields a pullable name."""
    import builtins

    real_import = builtins.__import__

    def no_swebench(name, *args, **kwargs):
        if name.startswith("swebench"):
            raise ImportError("swebench unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_swebench)
    assert instance_image_key(LITE_INSTANCE) == EXPECTED


def test_name_is_namespaced_and_pullable():
    """A local (non-namespaced) name would fail at pull time on every instance."""
    key = instance_image_key(LITE_INSTANCE)
    assert key.startswith(f"{IMAGE_NAMESPACE}/"), (
        f"{key!r} has no registry namespace; images.pull() cannot resolve it"
    )
    assert "__" not in key, f"{key!r} keeps a '__' that Docker tags disallow"
    assert key.split(":")[-1] == "latest"


def test_instance_id_is_lowercased():
    """Mixed-case ids appear outside Lite; Docker tags are case-sensitive."""
    key = instance_image_key({**LITE_INSTANCE, "instance_id": "PyCQA__flake8-1234"})
    assert key == "swebench/sweb.eval.x86_64.pycqa_1776_flake8-1234:latest"


def test_agrees_with_swebench_when_available():
    """The harness and the fallback must not disagree about the image."""
    pytest.importorskip("swebench", reason="swebench not importable on this platform")
    from swebench.harness.test_spec.test_spec import make_test_spec

    harness_key = make_test_spec(LITE_INSTANCE, IMAGE_NAMESPACE).instance_image_key
    assert instance_image_key(LITE_INSTANCE) == harness_key == EXPECTED
