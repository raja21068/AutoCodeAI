"""
eval/analyze_results.py
-----------------------
Loads evaluation results and produces:
  1. LaTeX tables ready to paste into the paper
  2. Matplotlib figures (resolve rate bar chart, per-repo breakdown)
  3. Console summary

Usage:
    python -m eval.analyze_results \
        --results_dir eval/results/run_001 \
        --ablation_dir eval/results/ablation \
        --output_dir eval/figures
"""

import argparse
import json
from pathlib import Path

from eval.metrics import compute_metrics, compute_ablation_table, print_ablation_table


# ---------------------------------------------------------------------------
# LaTeX table generators
# ---------------------------------------------------------------------------

def latex_main_results(metrics: dict, model: str = "GPT-4o") -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{AgentForge results on SWE-bench Lite (300 tasks).}",
        r"\label{tab:main}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Resolve Rate & Patch Rate \\",
        r"\midrule",
        f"Single-agent ({model}) & -- & -- \\\\",
        f"ReAct ({model})        & -- & -- \\\\",
        f"\\textbf{{AgentForge ({model})}} & "
        f"\\textbf{{{metrics['resolve_rate']:.1%}}} & "
        f"{metrics['patch_rate']:.1%} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_ablation_table(rows: list[dict]) -> str:
    name_map = {
        "full_pipeline": r"\textbf{Full pipeline (ours)}",
        "no_critic":     r"w/o Critic agent",
        "no_debugger":   r"w/o Debugger agent",
        "no_tester":     r"w/o Tester agent",
        "no_planner":    r"w/o Planner agent",
        "single_agent":  r"Single-agent baseline",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study. Each row removes one agent from the pipeline.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Configuration & Resolved & Resolve Rate & Patch Rate \\",
        r"\midrule",
    ]
    for r in rows:
        name = name_map.get(r["run"], r["run"])
        lines.append(
            f"{name} & {r['resolved']} & "
            f"{r['resolve_rate']:.1%} & "
            f"{r['patch_rate']:.1%} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_per_repo_table(per_repo: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-repository resolve rates on SWE-bench Lite.}",
        r"\label{tab:per_repo}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Repository & Tasks & Resolve Rate \\",
        r"\midrule",
    ]
    for repo, s in sorted(
        per_repo.items(), key=lambda x: x[1]["resolve_rate"], reverse=True
    ):
        short = repo.split("/")[-1]
        lines.append(
            f"{short} & {s['total']} & {s['resolve_rate']:.1%} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_resolve_rates(ablation_rows: list[dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({
            "font.family": "serif",
            "font.size":   11,
            "axes.spines.top":   False,
            "axes.spines.right": False,
        })

        labels = [r["run"].replace("_", "\n") for r in ablation_rows]
        rates  = [r["resolve_rate"] * 100 for r in ablation_rows]
        colors = ["#2563EB"] + ["#93C5FD"] * (len(rows) - 1)

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels, rates, color=colors, width=0.55, zorder=3)
        ax.set_ylabel("Resolve Rate (%)")
        ax.set_ylim(0, max(rates) * 1.25)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)

        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{rate:.1f}%",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_title("AgentForge ablation study — SWE-bench Lite", pad=12)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {output_path}")
    except ImportError:
        print("matplotlib not installed — skipping figure generation.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir",  default="eval/results/run_001")
    p.add_argument("--ablation_dir", default="eval/results/ablation")
    p.add_argument("--output_dir",   default="eval/figures")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load main run summary
    summary_path = Path(args.results_dir) / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        metrics = summary["metrics"]
        model   = summary.get("run_config", {}).get("model", "GPT-4o")

        print("\n=== Main results ===")
        print(f"  Resolve rate : {metrics['resolve_rate']:.1%}")
        print(f"  Patch rate   : {metrics['patch_rate']:.1%}")
        print(f"  Tasks        : {metrics['total']}")

        latex = latex_main_results(metrics, model)
        (output_dir / "table_main.tex").write_text(latex)
        print(f"\nSaved: {output_dir}/table_main.tex")

        if "per_repo" in metrics:
            latex_repo = latex_per_repo_table(metrics["per_repo"])
            (output_dir / "table_per_repo.tex").write_text(latex_repo)

    # Load ablation results
    ablation_path = Path(args.ablation_dir) / "ablation_table.json"
    if ablation_path.exists():
        rows = json.loads(ablation_path.read_text())
        print("\n=== Ablation table ===")
        print_ablation_table(rows)

        latex_abl = latex_ablation_table(rows)
        (output_dir / "table_ablation.tex").write_text(latex_abl)
        print(f"Saved: {output_dir}/table_ablation.tex")

        plot_resolve_rates(rows, output_dir / "ablation_bar.pdf")


if __name__ == "__main__":
    main()
