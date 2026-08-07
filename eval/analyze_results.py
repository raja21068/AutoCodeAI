"""
eval/analyze_results.py
-----------------------
Paired, budget-aware analysis over official harness reports.

Reviewer-driven requirements this module enforces:

* **McNemar needs pairs.** ``mcnemar()`` refuses to run on two aggregate
  rates. It takes two runs, intersects their instance sets, builds the
  discordant pairs (b, c) and reports an exact binomial test. Comparing
  against a published number from another paper is not possible here, by
  construction — that is the point.
* **Patch rate cannot be below resolve rate.** ``summarize()`` asserts it.
  A non-empty patch is a precondition for resolution, so the inversion in
  the rejected draft was a reporting error, and the assertion makes the same
  error impossible to repeat.
* **Intervals, not point estimates.** Resolution is reported with an exact
  Clopper-Pearson interval.
* **Cost is skewed.** Median and IQR are reported alongside the mean.

Usage:
    python -m eval.analyze_results --runs af_lite_001 single_forced_001
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from math import comb
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class Run:
    run_id: str
    config: str
    model: str
    resolved: set[str]
    submitted: set[str]
    nonempty: set[str]
    per_instance_cost: dict[str, float]
    per_instance_tokens: dict[str, int]
    per_instance_execs: dict[str, int]

    @property
    def resolve_rate(self) -> float:
        return len(self.resolved) / len(self.submitted) if self.submitted else 0.0

    @property
    def patch_rate(self) -> float:
        return len(self.nonempty) / len(self.submitted) if self.submitted else 0.0


def load_run(run_id: str, results_dir: Path) -> Run:
    run_dir = results_dir / run_id
    report = json.loads((run_dir / "official_report.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))

    resolved = set(report.get("resolved_ids", []))
    submitted = set(report.get("submitted_ids") or report.get("completed_ids", []))
    submitted |= resolved

    nonempty: set[str] = set()
    costs: dict[str, float] = {}
    tokens: dict[str, int] = {}
    execs: dict[str, int] = {}

    for path in sorted((run_dir / "trajectories").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        iid = record["instance_id"]
        if (record.get("model_patch") or "").strip():
            nonempty.add(iid)
        budget = record.get("budget", {})
        costs[iid] = budget.get("cost_usd", 0.0)
        tokens[iid] = budget.get("total_tokens", 0)
        execs[iid] = budget.get("executions", 0)

    return Run(
        run_id=run_id,
        config=summary.get("config", "?"),
        model=summary.get("model", "?"),
        resolved=resolved,
        submitted=submitted or set(costs),
        nonempty=nonempty,
        per_instance_cost=costs,
        per_instance_tokens=tokens,
        per_instance_execs=execs,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval; no normal approximation on small counts."""
    if n == 0:
        return (0.0, 0.0)

    def _cdf(p: float, k: int) -> float:
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

    def _bisect(target: float, k: int) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if _cdf(mid, k) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if successes == 0 else _bisect(1 - alpha / 2, successes - 1)
    upper = 1.0 if successes == n else _bisect(alpha / 2, successes)
    return (lower, upper)


def mcnemar(run_a: Run, run_b: Run) -> dict:
    """
    Exact McNemar test on paired per-instance outcomes.

    Requires both runs to have been executed locally on the same instances.
    There is deliberately no code path that accepts two aggregate rates.
    """
    shared = sorted(run_a.submitted & run_b.submitted)
    if not shared:
        raise ValueError(
            f"{run_a.run_id} and {run_b.run_id} share no instances — a paired "
            "test is not defined. Rerun both systems on the same instance list."
        )

    b = sum(1 for i in shared if i in run_a.resolved and i not in run_b.resolved)
    c = sum(1 for i in shared if i not in run_a.resolved and i in run_b.resolved)
    n = b + c

    if n == 0:
        p_value = 1.0
    else:
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
        p_value = min(1.0, 2 * tail)

    return {
        "n_paired": len(shared),
        "only_a": b,
        "only_b": c,
        "discordant": n,
        "p_value": p_value,
        "note": f"exact binomial McNemar over {len(shared)} paired instances",
    }


def summarize(run: Run) -> dict:
    n = len(run.submitted)
    k = len(run.resolved)
    low, high = clopper_pearson(k, n)

    assert run.patch_rate >= run.resolve_rate - 1e-9, (
        f"{run.run_id}: patch rate {run.patch_rate:.3f} < resolve rate "
        f"{run.resolve_rate:.3f}. A non-empty patch is necessary for "
        "resolution, so this indicates a bookkeeping error."
    )

    costs = [run.per_instance_cost.get(i, 0.0) for i in run.submitted]
    ordered = sorted(costs)
    half = len(ordered) // 2
    q1 = statistics.median(ordered[:half]) if half else 0.0
    q3 = statistics.median(ordered[-half:]) if half else 0.0

    return {
        "run_id": run.run_id,
        "config": run.config,
        "model": run.model,
        "n": n,
        "resolved": k,
        "resolve_rate": round(run.resolve_rate, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "patch_rate": round(run.patch_rate, 4),
        "mean_cost_usd": round(statistics.fmean(costs), 4) if costs else 0.0,
        "median_cost_usd": round(statistics.median(costs), 4) if costs else 0.0,
        "iqr_cost_usd": [round(q1, 4), round(q3, 4)],
        "cost_per_resolved_usd": round(sum(costs) / k, 4) if k else None,
        "mean_tokens": round(statistics.fmean(
            [run.per_instance_tokens.get(i, 0) for i in run.submitted]), 1) if n else 0,
        "mean_executions": round(statistics.fmean(
            [run.per_instance_execs.get(i, 0) for i in run.submitted]), 2) if n else 0,
    }


def latex_main_table(summaries: dict[str, dict]) -> str:
    """Emit the main results table with intervals — paste-ready."""
    rows = []
    for s in summaries.values():
        rows.append(
            f"{s['config'].replace('_', ' ')} & {s['resolved']}/{s['n']} & "
            f"{s['resolve_rate']:.1%} [{s['ci95'][0]:.1%}, {s['ci95'][1]:.1%}] & "
            f"{s['patch_rate']:.1%} & \\${s['mean_cost_usd']:.2f} & "
            f"{s['mean_executions']:.1f} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Official \\swebench{} Lite results under matched budgets. "
        "Intervals are exact Clopper--Pearson.}\n"
        "\\label{tab:main}\n\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "Configuration & Resolved & Resolve rate [95\\% CI] & Patch rate & "
        "Mean cost & Execs \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired analysis of official reports")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--results_dir", default="eval/results")
    parser.add_argument("--baseline", default=None,
                        help="run_id to compare every other run against")
    parser.add_argument("--latex", action="store_true", help="print a LaTeX table")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = {r: load_run(r, results_dir) for r in args.runs}

    print("\n" + "=" * 96)
    print(f"{'run':<22}{'config':<20}{'n':>5}{'resolved':>10}{'rate':>9}"
          f"{'95% CI':>18}{'patch':>8}{'$/task':>9}")
    print("-" * 96)
    summaries = {}
    for run_id, run in runs.items():
        s = summarize(run)
        summaries[run_id] = s
        ci = f"[{s['ci95'][0]:.1%},{s['ci95'][1]:.1%}]"
        print(f"{run_id:<22}{s['config']:<20}{s['n']:>5}{s['resolved']:>10}"
              f"{s['resolve_rate']:>8.1%}{ci:>18}{s['patch_rate']:>7.1%}"
              f"{s['mean_cost_usd']:>9.3f}")
    print("=" * 96)

    baseline = args.baseline or args.runs[0]
    for run_id in args.runs:
        if run_id == baseline:
            continue
        test = mcnemar(runs[run_id], runs[baseline])
        delta = summaries[run_id]["resolve_rate"] - summaries[baseline]["resolve_rate"]
        print(f"\n{run_id} vs {baseline}")
        print(f"  paired instances   : {test['n_paired']}")
        print(f"  {run_id} only      : {test['only_a']}")
        print(f"  {baseline} only    : {test['only_b']}")
        print(f"  difference         : {delta:+.1%}")
        print(f"  exact McNemar p    : {test['p_value']:.4g}")

    if args.latex:
        print("\n" + latex_main_table(summaries))

    out = results_dir / "analysis.json"
    out.write_text(json.dumps(
        {"summaries": summaries, "baseline": baseline}, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}\n")


if __name__ == "__main__":
    main()
