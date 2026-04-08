# Contributing

Thank you for your interest in contributing to AI Coder.

## Getting started

```bash
git clone https://github.com/your-username/ai-coder.git
cd ai-coder
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # add your OPENAI_API_KEY
```

## Workflow

1. Fork the repo and create a branch from `main`

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes — keep commits atomic and well-described

3. Add or update tests for anything you change

```bash
pytest tests/ -v
```

4. Make sure lint passes

```bash
pip install ruff
ruff check . --select E,F,W --ignore E501
```

5. Push and open a pull request against `main`

## Code style

- **Async-first** — all I/O must be `async`/`await`; no blocking calls in the hot path
- **Type annotations** — all function signatures must be fully annotated
- **No bare `except`** — always catch a specific exception type
- **Docstrings** — module-level and class-level docstrings required; function docstrings for anything non-obvious
- Line length is not enforced (E501 is ignored) but be reasonable

## Adding a new agent

1. Add a class to `core/agents/agents.py` following the same pattern as existing agents
2. Wire it into `services/orchestrator.py` — add an `elif agent == "your_agent":` branch
3. Update the `PlannerAgent.SYSTEM` prompt to include the new agent name
4. Add tests in `tests/test_agents.py`

## Reporting bugs

Open a GitHub issue with:
- Python version and OS
- Minimal reproduction steps
- Full traceback
