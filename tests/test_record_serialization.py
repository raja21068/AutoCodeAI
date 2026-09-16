"""
tests/test_record_serialization.py — A finished instance must survive being written.

The per-instance record is written *after* the agent has run. Anything that
raises at that point discards work that has already been paid for in API
spend and wall-clock, and the loss is silent: run_instance's own try/except
wraps only the agent call, so a write failure propagates to
``asyncio.gather(..., return_exceptions=True)``, which turns it into a value
that is then filtered out of `valid`.

The observed consequence was a run that executed the full agent loop, wrote
an empty predictions.jsonl, printed "Instances: 0", and exited 0 — which the
preflight reported as PASS. On a 300-instance sweep that is a complete loss
of the run presented as a success.

The cause was ``vars(self.config)`` embedding ``disabled_roles``, a frozenset.

Run:
    python -m pytest tests/test_record_serialization.py -v
"""

from __future__ import annotations

import json

import pytest

from eval.configs import CONFIGS
from eval.swebench_runner import jsonable
from services.repair_orchestrator import RepairConfig


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_config_as_dict_is_json_safe(name):
    """Every grid config must serialize, including the ablations."""
    cfg = RepairConfig(**CONFIGS[name])
    json.dumps(cfg.as_dict())


def test_as_dict_sorts_sets():
    cfg = RepairConfig(disabled_roles=frozenset({"tester", "critic"}))
    assert cfg.as_dict()["disabled_roles"] == ["critic", "tester"]


def test_as_dict_keeps_every_field():
    """A JSON-safe view must not quietly drop configuration."""
    cfg = RepairConfig(**CONFIGS["no_critic"])
    assert set(cfg.as_dict()) == set(vars(cfg))


def test_full_record_shape_serializes():
    """The record the orchestrator returns, written the way the runner writes it."""
    cfg = RepairConfig(**CONFIGS["no_planner"])
    record = {
        "instance_id": "astropy__astropy-12907",
        "model_patch": "diff --git a/x b/x\n",
        "stop_reason": "solved",
        "trajectory": [{"role": "planner", "text": "..."}],
        "budget": {"total_tokens": 10, "cost_usd": 0.01},
        "config": cfg.as_dict(),
        "repo": "astropy/astropy",
    }
    decoded = json.loads(json.dumps(jsonable(record), indent=2))
    assert decoded["config"]["disabled_roles"] == ["planner"]
    assert decoded["model_patch"].startswith("diff --git")


def test_record_write_survives_an_unexpected_object():
    """
    A future field must not be able to destroy a completed instance.

    This is the property that matters more than any particular type: the
    write happens after the money is spent, so it must degrade rather than
    raise.
    """
    class Exotic:
        def __repr__(self):
            return "<exotic>"

    record = {"instance_id": "x", "surprise": Exotic(), "nested": {"s": {1, 2}}}
    decoded = json.loads(json.dumps(jsonable(record)))
    assert decoded["surprise"] == "<exotic>"
    assert decoded["nested"]["s"] == [1, 2]
