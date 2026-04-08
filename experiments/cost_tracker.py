"""
experiments/cost_tracker.py
----------------------------
Wraps every LLM call to track token usage and estimated cost.
Import this INSTEAD of core.utils.llm when running experiments.

Usage:
    from experiments.cost_tracker import llm, llm_stream, print_cost_summary
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from openai import AsyncOpenAI

_client: AsyncOpenAI | None = None
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# GPT-4o pricing per 1M tokens
PRICE_IN  = 5.00
PRICE_OUT = 15.00

# Global usage log
_usage_log: list[dict] = []


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _record(agent_hint: str, in_tokens: int, out_tokens: int, elapsed: float):
    cost = (in_tokens * PRICE_IN + out_tokens * PRICE_OUT) / 1_000_000
    _usage_log.append({
        "timestamp":  datetime.utcnow().isoformat(),
        "agent":      agent_hint,
        "in_tokens":  in_tokens,
        "out_tokens": out_tokens,
        "cost_usd":   round(cost, 6),
        "elapsed_s":  round(elapsed, 3),
    })


async def llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    agent: str = "unknown",
) -> str:
    client = _get_client()
    t0 = time.time()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        seed=42,
        temperature=0.0,
    )
    elapsed   = time.time() - t0
    usage     = response.usage
    _record(agent, usage.prompt_tokens, usage.completion_tokens, elapsed)
    return response.choices[0].message.content or ""


async def llm_stream(
    prompt: str,
    system: str = "You are a helpful assistant.",
    agent: str = "unknown",
) -> AsyncGenerator[str, None]:
    client     = _get_client()
    t0         = time.time()
    in_tokens  = 0
    out_tokens = 0

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        seed=42,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        if chunk.usage:
            in_tokens  = chunk.usage.prompt_tokens
            out_tokens = chunk.usage.completion_tokens
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta

    _record(agent, in_tokens, out_tokens, time.time() - t0)


# ── Summary helpers ───────────────────────────────────────────

def get_usage_summary() -> dict:
    if not _usage_log:
        return {}
    total_in   = sum(e["in_tokens"]  for e in _usage_log)
    total_out  = sum(e["out_tokens"] for e in _usage_log)
    total_cost = sum(e["cost_usd"]   for e in _usage_log)
    by_agent: dict[str, dict] = {}
    for e in _usage_log:
        a = e["agent"]
        if a not in by_agent:
            by_agent[a] = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "cost_usd": 0}
        by_agent[a]["calls"]      += 1
        by_agent[a]["in_tokens"]  += e["in_tokens"]
        by_agent[a]["out_tokens"] += e["out_tokens"]
        by_agent[a]["cost_usd"]   += e["cost_usd"]
    return {
        "total_calls":      len(_usage_log),
        "total_in_tokens":  total_in,
        "total_out_tokens": total_out,
        "total_cost_usd":   round(total_cost, 4),
        "by_agent":         by_agent,
    }


def print_cost_summary() -> None:
    s = get_usage_summary()
    if not s:
        print("No API calls recorded.")
        return
    print("\n── Token Usage Summary ─────────────────────────────────")
    print(f"  Total calls      : {s['total_calls']}")
    print(f"  Input tokens     : {s['total_in_tokens']:,}")
    print(f"  Output tokens    : {s['total_out_tokens']:,}")
    print(f"  Estimated cost   : ${s['total_cost_usd']:.4f}")
    print("\n  By agent:")
    for agent, u in sorted(s["by_agent"].items(), key=lambda x: -x[1]["cost_usd"]):
        print(f"    {agent:12} {u['calls']:4} calls  "
              f"in={u['in_tokens']:7,}  out={u['out_tokens']:6,}  "
              f"${u['cost_usd']:.4f}")
    print("───────────────────────────────────────────────────────\n")


def save_usage_log(path: str) -> None:
    summary = get_usage_summary()
    summary["calls"] = _usage_log
    Path(path).write_text(json.dumps(summary, indent=2))
    print(f"Usage log saved → {path}")


def reset() -> None:
    global _usage_log
    _usage_log = []
