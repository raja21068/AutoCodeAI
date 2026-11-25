# AgentForge paper

The paper for the AgentForge framework implemented in this repository (`../02_Code/core`, `../02_Code/services`, `../02_Code/eval`).
Folder names below are relative to the repository root.

| Folder | Contents |
|---|---|
| `01_Paper/` (this folder) | **Current version**: "AgentForge: A Role-Structured Multi-Agent Framework for Repository-Level Software Repair with Execution Feedback" (IEEEtran, 10 pages). `AgentForge_Polished_main.tex`/`.pdf`, `ref.bib`, `Figure1.png`, `tables/` (main results, published context, budget, ablation) |
| `03_Evidence/table_fix_2026-08-17/` | Material behind Tables II-V: `TABLE_FIX_NOTES.md` (only AgentForge has an official SWE-bench harness report: 1/50 resolved, 2.0%, 95% CI [0.05%, 10.65%]; other rows marked "Not graded"), `eval/analyze_results.py`, `eval/export_tables.py`, `lite50.json` |
| `06_Reconstruction/v1_execution_grounded/` | Earlier version: "AgentForge: Execution-Grounded Multi-Agent LLM Framework for Autonomous Software Engineering" (`main.tex`, `sections/`, figures, NeurIPS style and checklist). Has its own copies of `Figure1.png` and `ref.bib` so it builds on its own (11 pages) |
| `06_Reconstruction/` | `paper_build_older_build_of_Polished_main.*` (an earlier build of the current version), `AgentForge_Polished_Revision_Draft.pdf`, a duplicate PDF of the current version, and the tables-source zip |

## Compile
From inside `01_Paper/`: `latexmk -pdf AgentForge_Polished_main.tex`.

## Notes (2026-11-25)
- Reorganized with `git mv`; file contents unchanged. The current version rebuilds at 10 pages, as before.
- `ref.bib` has a repeated entry (BibTeX: "Repeated entry---line 438"); the generated bibliography is unaffected.
- `sections/results.tex` in the earlier version refers to `sections/ablation_bar.pdf`, which does not exist; `main.tex` does not load `sections/`.
