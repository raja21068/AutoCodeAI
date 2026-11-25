# AutoCodeAI / AgentForge

AgentForge: a role-structured multi-agent framework for repository-level software repair with execution feedback.
This repository holds the implementation and the paper.

| Folder | Contents |
|---|---|
| `01_Paper/` | Current paper (IEEEtran, 10 pages): `AgentForge_Polished_main.tex`/`.pdf`, `ref.bib`, `Figure1.png`, `tables/`. See `01_Paper/README.md` |
| `02_Code/` | The application, layout unchanged: `api/`, `core/`, `services/`, `memory/`, `packages/`, `eval/` (SWE-bench pipeline), `experiments/`, `tests/`, `main.py`, Dockerfiles, `docker-compose*.yml`, `requirements.txt`. **Full documentation: [`02_Code/README.md`](02_Code/README.md)** and [`02_Code/QUICKSTART.md`](02_Code/QUICKSTART.md) |
| `03_Evidence/table_fix_2026-08-17/` | Notes and scripts behind the paper's Tables II-V |
| `04_Outputs/diagnostics/` | AgentForge DeepSeek-chat smoke and diagnostic run summaries (JSON) |
| `06_Reconstruction/` | Earlier "Execution-Grounded" paper version (`v1_execution_grounded/`), an older build of the current paper, revision draft, tables-source zip |

## Run
All commands run from inside `02_Code/` (as they did from the repository root before 2026-11-25):
```
cd 02_Code
pip install -r requirements.txt
python main.py
python -m pytest tests
docker compose up
```
The eval launchers (`eval/overnight_run.ps1`, `eval/wait_and_run.ps1`, `eval/bootstrap_wsl.sh`) point at `.../AutoCodeAI/02_Code`.

## Paper
From inside `01_Paper/`: `latexmk -pdf AgentForge_Polished_main.tex`.

## License
MIT, see `LICENSE`.
