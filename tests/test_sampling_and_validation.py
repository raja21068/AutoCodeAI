"""
tests/test_sampling_and_validation.py

Covers the two pieces of measurement machinery whose logic is easy to get
subtly wrong and impossible to eyeball after the fact:

* stratified subset allocation — must sum to N, respect per-repository
  capacity, and be reproducible from the seed alone;
* generated-test classification — the label that decides whether a
  reproduction script counts as discriminating.

Both are pure functions, so neither test needs Docker, a dataset download, or
an API key.

Run:
    python -m pytest tests/test_sampling_and_validation.py -v
"""

from __future__ import annotations

import pytest

from eval.sample_subset import allocate, stratified_sample
from eval.validate_generated_tests import classify

# Roughly the shape of SWE-bench Lite: a few dominant repositories and a tail.
LITE_SHAPE = {
    "django/django": 114, "sympy/sympy": 77, "scikit-learn/scikit-learn": 23,
    "matplotlib/matplotlib": 23, "sphinx-doc/sphinx": 16, "astropy/astropy": 6,
    "pytest-dev/pytest": 17, "pydata/xarray": 5, "pallets/flask": 3,
    "psf/requests": 6, "mwaskom/seaborn": 4,
}


def make_instances(shape: dict[str, int]) -> list[dict]:
    return [
        {"instance_id": f"{repo.replace('/', '__')}-{i}", "repo": repo}
        for repo, count in shape.items()
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 10, 50, 100, 150, 294])
def test_allocation_sums_to_n(n):
    quota = allocate(LITE_SHAPE, n)
    assert sum(quota.values()) == n


@pytest.mark.parametrize("n", [10, 100, 200])
def test_allocation_never_exceeds_capacity(n):
    quota = allocate(LITE_SHAPE, n)
    for repo, count in quota.items():
        assert count <= LITE_SHAPE[repo], f"{repo} over-allocated"


def test_allocation_is_proportional():
    """django is ~39% of the split, so it should get roughly 39 of 100."""
    quota = allocate(LITE_SHAPE, 100)
    total = sum(LITE_SHAPE.values())
    expected = 100 * LITE_SHAPE["django/django"] / total
    assert abs(quota["django/django"] - expected) <= 1


def test_allocation_requesting_everything_returns_everything():
    assert allocate(LITE_SHAPE, sum(LITE_SHAPE.values())) == LITE_SHAPE
    assert allocate(LITE_SHAPE, 10_000) == LITE_SHAPE


def test_allocation_of_zero_is_empty():
    assert sum(allocate(LITE_SHAPE, 0).values()) == 0


def test_allocation_handles_capacity_pressure():
    """Small repositories cannot absorb their proportional share of a large N."""
    shape = {"big": 100, "tiny": 2}
    quota = allocate(shape, 101)
    assert sum(quota.values()) == 101
    assert quota["tiny"] <= 2


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_is_reproducible_from_the_seed():
    instances = make_instances(LITE_SHAPE)
    a = stratified_sample(instances, 100, seed=20260801)
    b = stratified_sample(instances, 100, seed=20260801)
    assert a == b


def test_different_seeds_give_different_samples():
    instances = make_instances(LITE_SHAPE)
    a = stratified_sample(instances, 100, seed=1)
    b = stratified_sample(instances, 100, seed=2)
    assert a != b


def test_sample_is_independent_of_dataset_ordering():
    """Row order is not guaranteed; the subset must not depend on it."""
    instances = make_instances(LITE_SHAPE)
    shuffled = list(reversed(instances))
    assert stratified_sample(instances, 100) == stratified_sample(shuffled, 100)


def test_sample_covers_every_repository_at_reasonable_n():
    instances = make_instances(LITE_SHAPE)
    chosen = set(stratified_sample(instances, 100))
    repos = {i["repo"] for i in instances if i["instance_id"] in chosen}
    assert repos == set(LITE_SHAPE), (
        "a stratified sample of 100 should touch every repository; a prefix "
        "of the split would not"
    )


def test_sample_has_no_duplicates():
    instances = make_instances(LITE_SHAPE)
    chosen = stratified_sample(instances, 100)
    assert len(chosen) == len(set(chosen)) == 100


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_fails_buggy_passes_fixed_is_discriminating():
    assert classify(1, "AssertionError", 0, "") == "discriminating"


def test_passes_both_is_vacuous():
    assert classify(0, "", 0, "") == "vacuous"


def test_fails_both_is_always_fails():
    assert classify(1, "AssertionError", 1, "AssertionError") == "always_fails"


def test_passes_buggy_fails_fixed_is_inverted():
    assert classify(0, "", 1, "AssertionError") == "inverted"


@pytest.mark.parametrize("marker", [
    "ModuleNotFoundError: No module named 'foo'",
    "ImportError: cannot import name 'bar'",
    "SyntaxError: invalid syntax",
])
def test_import_and_syntax_errors_are_not_counted_as_bug_detection(marker):
    """
    A script that fails on the buggy tree because it cannot import anything
    has not detected the bug, and must not inflate the discriminating rate.
    """
    assert classify(1, marker, 0, "") == "infrastructure_failure"


def test_infrastructure_failure_on_either_side_is_caught():
    assert classify(1, "AssertionError", 1, "ModuleNotFoundError: x") == \
        "infrastructure_failure"
