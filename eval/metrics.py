"""
eval/metrics.py
---------------
Computes evaluation metrics from a list of per-task result dicts.

Metrics reported:
    resolve_rate     — % of tasks fully resolved (primary SWE-bench metric)
    patch_rate       — % of tasks where patch applied cleanly
    avg_elapsed_s    — average wall-clock time per task
    per_repo         — resolve rate broken down by repository
    error_rate       — % of tasks that threw an exception
"""

from collections import defaultdict
from typing import Any


def compute_metrics(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {}

    total         = len(records)
    resolved      = sum(1 for r in records if r.get("resolved"))
    patch_applied = sum(1 for r in records if r.get("patch_applied"))
    errors        = sum(1 for r in records if r.get("error"))
    elapsed_vals  = [r["elapsed_s"] for r in records if "elapsed_s" in r]

    # Per-repo breakdown
    repo_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "resolved": 0})
    for r in records:
        repo = r.get("repo", "unknown")
        repo_stats[repo]["total"]    += 1
        repo_stats[repo]["resolved"] += int(bool(r.get("resolved")))

    per_repo = {
        repo: {
            "total":        s["total"],
            "resolved":     s["resolved"],
            "resolve_rate": round(s["resolved"] / s["total"], 4),
        }
        for repo, s in sorted(repo_stats.items())
    }

    return {
        "total":          total,
        "resolved":       resolved,
        "patch_applied":  patch_applied,
        "errors":         errors,
        "resolve_rate":   round(resolved / total, 4),
        "patch_rate":     round(patch_applied / total, 4),
        "error_rate":     round(errors / total, 4),
        "avg_elapsed_s":  round(sum(elapsed_vals) / len(elapsed_vals), 2)
                          if elapsed_vals else 0,
        "per_repo":       per_repo,
    }


def compute_ablation_table(runs: dict[str, list[dict]]) -> list[dict]:
    """
    Compute a comparison table across multiple named runs.

    Args:
        runs: dict mapping run_name → list of result records
              e.g. {
                  "full_pipeline":   [...],
                  "no_critic":       [...],
                  "no_debugger":     [...],
                  "single_agent":    [...],
              }

    Returns:
        List of dicts, one per run, sorted by resolve_rate descending.
    """
    rows = []
    for name, records in runs.items():
        m = compute_metrics(records)
        rows.append({
            "run":          name,
            "total":        m.get("total", 0),
            "resolved":     m.get("resolved", 0),
            "resolve_rate": m.get("resolve_rate", 0.0),
            "patch_rate":   m.get("patch_rate", 0.0),
            "avg_elapsed_s": m.get("avg_elapsed_s", 0.0),
        })
    return sorted(rows, key=lambda r: r["resolve_rate"], reverse=True)


def print_ablation_table(rows: list[dict]) -> None:
    """Pretty-print the ablation table to stdout."""
    header = f"{'Run':<25} {'Resolved':>9} {'Rate':>7} {'Patch%':>7} {'Avg(s)':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['run']:<25} "
            f"{r['resolved']:>9} "
            f"{r['resolve_rate']:>6.1%} "
            f"{r['patch_rate']:>6.1%} "
            f"{r['avg_elapsed_s']:>7.1f}"
        )
    print("=" * len(header) + "\n")
