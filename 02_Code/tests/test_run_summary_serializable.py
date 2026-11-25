"""
tests/test_run_summary_serializable.py — The run summary must survive json.

``run_summary.json`` holds the token, cost and execution ledger that backs
the matched-budget comparison. It is written at the very end of a run, so
anything that makes it unserializable destroys the evidence *after* the
expensive work has already been paid for — the predictions survive, the
accounting does not, and the process exits non-zero as though nothing worked.

The original failure was ``RepairConfig.disabled_roles``, a frozenset, dumped
straight through ``vars(cfg)``. Every config carries that field, including
the default one where it is merely empty, so every run crashed.

Run:
    python -m pytest tests/test_run_summary_serializable.py -v
"""

from __future__ import annotations

import json

import pytest

from eval.configs import CONFIGS
from eval.swebench_runner import jsonable
from services.repair_orchestrator import RepairConfig


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_config_serializes(name):
    """No entry in the experimental grid may break the summary write."""
    cfg = RepairConfig(**CONFIGS[name])
    encoded = json.dumps(jsonable(vars(cfg)))
    assert json.loads(encoded) is not None


def test_frozenset_becomes_sorted_list():
    """Sets are ordered so re-running a config yields an identical summary."""
    assert jsonable(frozenset({"tester", "critic", "planner"})) == [
        "critic", "planner", "tester",
    ]


def test_disabled_roles_survive_the_round_trip():
    """The ablation must stay legible in the summary, not vanish into a string."""
    cfg = RepairConfig(**CONFIGS["no_critic"])
    decoded = json.loads(json.dumps(jsonable(vars(cfg))))
    assert decoded["disabled_roles"] == ["critic"]


def test_unknown_types_degrade_instead_of_raising():
    """A future config field must not be able to crash a completed run."""
    class Exotic:
        def __repr__(self):
            return "<exotic>"

    assert json.dumps(jsonable({"x": Exotic()})) == '{"x": "<exotic>"}'
