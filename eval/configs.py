"""
eval/configs.py
---------------
The experimental grid, in one place.

Kept free of heavy imports so that the runner, the ablation driver, the
analysis code and the tests can all read the same table without pulling in
``datasets`` or the Docker SDK.

Each entry is a set of overrides for :class:`services.repair_orchestrator.RepairConfig`.
Budget fields are deliberately absent: they are supplied identically to every
condition from the command line, which is what makes the grid controlled.

The first four entries form the primary 2x2 factorial, separating role
decomposition from execution scheduling:

                        optional execution      forced execution
    single role         single_optional         single_forced
    five roles          multi_optional          agentforge

The remainder are component ablations. See ``RepairConfig.disabled_roles``
for the precise meaning of each role removal.
"""

from __future__ import annotations

FACTORIAL: dict[str, dict] = {
    "single_optional": {"role_structure": "single", "force_execution": False},
    "single_forced":   {"role_structure": "single", "force_execution": True},
    "multi_optional":  {"role_structure": "multi",  "force_execution": False},
    "agentforge":      {"role_structure": "multi",  "force_execution": True},
}

ABLATIONS: dict[str, dict] = {
    "no_retrieval": {
        "role_structure": "multi", "force_execution": True, "use_retrieval": False,
    },
    "no_generated_tests": {
        "role_structure": "multi", "force_execution": True,
        "use_generated_tests": False,
    },
    "no_planner": {
        "role_structure": "multi", "force_execution": True,
        "disabled_roles": frozenset({"planner"}),
    },
    "no_tester": {
        "role_structure": "multi", "force_execution": True,
        "disabled_roles": frozenset({"tester"}),
    },
    "no_debugger": {
        "role_structure": "multi", "force_execution": True,
        "disabled_roles": frozenset({"debugger"}),
    },
    "no_critic": {
        "role_structure": "multi", "force_execution": True,
        "disabled_roles": frozenset({"critic"}),
    },
}

BASELINES: dict[str, dict] = {
    # Single-agent ReAct with the same repository, tools and budget. This is
    # the meaningful single-agent comparison.
    "react": {"role_structure": "single", "force_execution": False},
    # One prompt, one response, no tools, no repository. Reported as a floor
    # only; it differs from AgentForge in every respect at once.
    "single_call": {"role_structure": "single", "force_execution": False,
                    "use_retrieval": False, "use_generated_tests": False},
}

CONFIGS: dict[str, dict] = {**FACTORIAL, **ABLATIONS, **BASELINES}

# Which loop implements each condition. Everything not listed here runs on
# the AgentForge orchestrator.
AGENT_KIND: dict[str, str] = {
    "react": "react",
    "single_call": "single_call",
}


def agent_kind(config_name: str) -> str:
    return AGENT_KIND.get(config_name, "agentforge")


# Order used by the ablation driver and by the results tables.
GRID: list[str] = list(FACTORIAL) + list(ABLATIONS) + list(BASELINES)
