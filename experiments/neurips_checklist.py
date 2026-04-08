# NeurIPS 2026 — Submission Checklist & Timeline
# Run: python experiments/neurips_checklist.py

"""
experiments/neurips_checklist.py
---------------------------------
Tracks your progress toward NeurIPS 2026 submission.
Run this anytime to see what's done and what's next.

Usage:
    python experiments/neurips_checklist.py
"""

from datetime import date
from pathlib import Path


TODAY = date.today()

MILESTONES = [
    # (deadline, label, check_fn)
    (date(2025, 7,  1),  "Pilot run complete (10 tasks)",
     lambda: (Path("experiments/results/pilot/summary.json").exists(), "experiments/results/pilot/")),

    (date(2025, 7, 15),  "Full SWE-bench Lite run (300 tasks)",
     lambda: (Path("experiments/results/full_run/summary.json").exists(), "experiments/results/full_run/")),

    (date(2025, 8,  1),  "Ablation study complete (6 conditions × 100 tasks)",
     lambda: (Path("experiments/results/ablation/ablation_table.json").exists(), "experiments/results/ablation/")),

    (date(2025, 8, 15),  "Baselines run (single-agent + ReAct)",
     lambda: (Path("experiments/results/baselines").exists(), "experiments/results/baselines/")),

    (date(2025, 9,  1),  "First full paper draft",
     lambda: (Path("paper/main.tex").exists(), "paper/main.tex")),

    (date(2025, 9, 15),  "All figures and tables generated",
     lambda: (Path("experiments/figures/ablation_bar.pdf").exists(), "experiments/figures/")),

    (date(2025, 10, 1),  "Related work section complete",
     lambda: (Path("paper/sections/related_work.tex").exists(), "paper/sections/related_work.tex")),

    (date(2025, 10, 15), "Co-author / advisor review round 1",
     lambda: (False, "manual — mark done when complete")),

    (date(2025, 11, 1),  "Revision complete after review",
     lambda: (False, "manual")),

    (date(2025, 12, 1),  "Camera-ready polish + appendix",
     lambda: (False, "manual")),

    (date(2026, 1,  1),  "Abstract submitted to NeurIPS",
     lambda: (False, "manual")),

    (date(2026, 5, 15),  "Full paper submitted to NeurIPS 2026",
     lambda: (False, "manual — estimated deadline")),
]

PAPER_SECTIONS = {
    "Abstract":       "paper/sections/abstract.tex",
    "Introduction":   "paper/sections/introduction.tex",
    "Related Work":   "paper/sections/related_work.tex",
    "Method":         "paper/sections/method.tex",        # ✅ done
    "Experiments":    "paper/sections/experiments.tex",
    "Results":        "paper/sections/results.tex",
    "Ablation":       "paper/sections/ablation.tex",
    "Conclusion":     "paper/sections/conclusion.tex",
    "Appendix":       "paper/sections/appendix.tex",
}

EXPERIMENT_FILES = {
    "Pilot results":    "experiments/results/pilot/summary.json",
    "Full run results": "experiments/results/full_run/summary.json",
    "Ablation table":   "experiments/results/ablation/ablation_table.json",
    "Cost log":         "experiments/results/cost_log.json",
    "Main result table":"experiments/tables/table_main.tex",
    "Ablation table tex":"experiments/tables/table_ablation.tex",
    "Figure: ablation": "experiments/figures/ablation_bar.pdf",
}


def colored(text, color):
    codes = {"green": "\033[92m", "red": "\033[91m",
             "yellow": "\033[93m", "gray": "\033[90m", "bold": "\033[1m"}
    return f"{codes.get(color,'')}{text}\033[0m"


def days_until(d: date) -> int:
    return (d - TODAY).days


def main():
    print(colored("\n╔══════════════════════════════════════════════════════╗", "bold"))
    print(colored("║  AgentForge — NeurIPS 2026 Submission Tracker       ║", "bold"))
    print(colored("╚══════════════════════════════════════════════════════╝", "bold"))
    print(f"\n  Today: {TODAY}  |  NeurIPS deadline: ~2026-05-15\n")

    # Timeline
    print(colored("  ── Milestones ─────────────────────────────────────────", "bold"))
    completed = 0
    for deadline, label, check_fn in MILESTONES:
        done, hint = check_fn()
        days = days_until(deadline)
        if done:
            status = colored("✅  DONE", "green")
            completed += 1
        elif days < 0:
            status = colored(f"⚠️   OVERDUE ({-days}d ago)", "red")
        elif days <= 14:
            status = colored(f"🔥  DUE IN {days}d", "yellow")
        else:
            status = colored(f"○   {deadline}  ({days}d)", "gray")
        print(f"  {status:35}  {label}")
        if not done:
            print(colored(f"                                         → {hint}", "gray"))

    print(f"\n  Progress: {completed}/{len(MILESTONES)} milestones complete\n")

    # Paper sections
    print(colored("  ── Paper sections ──────────────────────────────────────", "bold"))
    for section, path in PAPER_SECTIONS.items():
        exists = Path(path).exists()
        icon   = colored("✅", "green") if exists else colored("○ ", "gray")
        print(f"  {icon}  {section:<20} {path}")

    # Experiment files
    print(colored("\n  ── Experiment outputs ──────────────────────────────────", "bold"))
    for label, path in EXPERIMENT_FILES.items():
        exists = Path(path).exists()
        icon   = colored("✅", "green") if exists else colored("○ ", "gray")
        size   = f"({Path(path).stat().st_size:,} bytes)" if exists else ""
        print(f"  {icon}  {label:<25} {size}")

    # Next action
    print(colored("\n  ── Next action ─────────────────────────────────────────", "bold"))
    next_undone = next(
        ((d, l) for d, l, cf in MILESTONES if not cf()[0]), None
    )
    if next_undone:
        deadline, label = next_undone
        days = days_until(deadline)
        print(colored(f"  👉  {label}", "bold"))
        print(f"      Deadline: {deadline} ({days} days from today)")
        print(f"      Run: python experiments/setup_and_run.py --pilot\n")
    else:
        print(colored("  🎉  All milestones complete — submit!\n", "green"))


if __name__ == "__main__":
    main()
