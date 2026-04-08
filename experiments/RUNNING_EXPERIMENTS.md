# Running Real Experiments — Step by Step

## Before you run anything

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Open .env and set:
# OPENAI_API_KEY=sk-your-key-here
# OPENAI_MODEL=gpt-4o
```

### 3. Start ChromaDB

```bash
docker compose up chromadb -d
# Verify it's running:
curl http://localhost:8001/api/v1/heartbeat
```

### 4. Pull the Docker sandbox image

```bash
docker pull python:3.10-slim
```

---

## Check everything is ready

```bash
python experiments/setup_and_run.py --check
```

You should see all green checkmarks. Fix anything that's red before continuing.

---

## Step 1 — Pilot run (10 tasks, ~$2, ~20 minutes)

This is your first real run. It verifies the full pipeline works end-to-end
with real API calls and real SWE-bench tasks before spending more.

```bash
python experiments/setup_and_run.py --pilot
```

Check results:
```bash
cat experiments/results/pilot/summary.json
```

Expected: 2–5 tasks resolved out of 10 (20–50% resolve rate on pilot).
If 0 resolve, check the per-task JSON files for errors before continuing.

---

## Step 2 — Full SWE-bench Lite run (300 tasks, ~$80, ~6 hours)

```bash
python experiments/setup_and_run.py --full
```

This will ask for confirmation before starting. The run saves a JSON file
per task so you can resume if it crashes:

```bash
# Resume a partial run:
python -m eval.swebench_runner \
    --split lite \
    --output_dir experiments/results/full_run \
    --resume
```

---

## Step 3 — Ablation study (600 tasks total, ~$180, ~12 hours)

Run one condition at a time to manage cost:

```bash
# Full pipeline (this is your main result — run this first)
python -m eval.ablation_runner \
    --max_tasks 100 \
    --output_dir experiments/results/ablation

# Or run all 6 conditions at once:
python experiments/setup_and_run.py --ablation
```

---

## Step 4 — Generate paper tables and figures

```bash
python -m eval.analyze_results \
    --results_dir experiments/results/full_run \
    --ablation_dir experiments/results/ablation \
    --output_dir experiments/figures

# Tables are now in experiments/figures/:
#   table_main.tex       ← paste into paper Section 4
#   table_ablation.tex   ← paste into paper Section 4.2
#   ablation_bar.pdf     ← Figure 3 in the paper
```

---

## Track your progress toward NeurIPS

```bash
python experiments/neurips_checklist.py
```

---

## Recommended model settings for reproducibility

Always use these settings in experiments — reviewers will ask:

```
model:        gpt-4o
temperature:  0.0        # deterministic outputs
seed:         42         # OpenAI best-effort reproducibility
```

Log the exact model version returned by the API in your paper
(e.g., `gpt-4o-2024-08-06`) — OpenAI updates models silently.

---

## Cost control tips

- Use `--max_tasks 50` for a cheap validation run before committing to 300
- The `--resume` flag skips completed tasks — safe to Ctrl+C and restart
- Watch live cost: `tail -f experiments/results/full_run/*.json | grep cost`
- Set a spend limit in your OpenAI account settings before starting
