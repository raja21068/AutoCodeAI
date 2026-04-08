"""
core/utils/llm.py — Single source of truth for all LLM calls.

Provides:
    llm()        — async, returns full response string
    llm_stream() — async generator, yields tokens as they arrive

Supports multiple modes:
    - LiteLLM (default): unified interface to 100+ providers
    - OpenAI-compatible: DeepSeek, OpenAI, Azure OpenAI
    - Local models: Ollama and other local endpoints

Configuration via environment variables:
    LLM_MODE          — litellm (default) | openai | deepseek | local
    LLM_PROVIDER      — legacy alias for LLM_MODE
    OPENAI_API_KEY    — for OpenAI mode
    DEEPSEEK_API_KEY  — for DeepSeek mode
    LOCAL_LLM_URL     — for local mode (default: http://localhost:11434/v1)
    LLM_MODEL         — model name override
    LOCAL_MODEL       — model for local mode

Per-agent routing (env overrides):
    PLANNER_MODEL   default: gpt-4o
    CODER_MODEL     default: deepseek/deepseek-chat
    TESTER_MODEL    default: groq/llama-3.3-70b-versatile
    DEBUGGER_MODEL  default: anthropic/claude-sonnet-4-5
    CRITIC_MODEL    default: anthropic/claude-sonnet-4-5
    DEFAULT_MODEL   fallback when agent is unrecognised
"""

from __future__ import annotations

import os
from typing import AsyncGenerator
import logging

import litellm
import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Silence LiteLLM's verbose success logs; keep warnings/errors.
litellm.set_verbose = False

# ── Mode Detection ─────────────────────────────────────────────────────────
LLM_MODE = os.getenv("LLM_MODE") or os.getenv("LLM_PROVIDER", "litellm")

# Initialize clients for non-LiteLLM modes
_openai_client: AsyncOpenAI | None = None
_deepseek_client: AsyncOpenAI | None = None
_local_client: AsyncOpenAI | None = None

if LLM_MODE == "openai":
    _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif LLM_MODE == "deepseek":
    _deepseek_client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
elif LLM_MODE == "local":
    _local_client = AsyncOpenAI(
        api_key="none",  # local models don't need a key
        base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
    )

# ── Model routing ──────────────────────────────────────────────────────────

_ROUTING: dict[str, str] = {
    "planner":  "PLANNER_MODEL",
    "coder":    "CODER_MODEL",
    "tester":   "TESTER_MODEL",
    "debugger": "DEBUGGER_MODEL",
    "critic":   "CRITIC_MODEL",
}

_DEFAULTS: dict[str, str] = {
    "planner":  "gpt-4o",
    "coder":    "deepseek/deepseek-chat" if LLM_MODE == "litellm" else "deepseek-chat",
    "tester":   "groq/llama-3.3-70b-versatile" if LLM_MODE == "litellm" else "llama-3.3-70b-versatile",
    "debugger": "anthropic/claude-sonnet-4-5" if LLM_MODE == "litellm" else "claude-sonnet-4",
    "critic":   "anthropic/claude-sonnet-4-5" if LLM_MODE == "litellm" else "claude-sonnet-4",
}


def _resolve_model(agent: str) -> str:
    """Return the model string for *agent*, respecting env-var overrides."""
    agent = agent.lower()
    env_key = _ROUTING.get(agent)
    
    # Check for explicit model override
    if os.getenv("LLM_MODEL"):
        return os.getenv("LLM_MODEL")
    
    if env_key:
        return os.getenv(env_key, _DEFAULTS.get(agent, "gpt-4o"))
    
    # Local mode special handling
    if LLM_MODE == "local":
        return os.getenv("LOCAL_MODEL", "deepseek-coder")
    
    return os.getenv("DEFAULT_MODEL", "gpt-4o")


def _get_client() -> AsyncOpenAI | None:
    """Return the appropriate client based on LLM_MODE."""
    if LLM_MODE == "openai":
        return _openai_client
    elif LLM_MODE == "deepseek":
        return _deepseek_client
    elif LLM_MODE == "local":
        return _local_client
    return None


# ── Public API ─────────────────────────────────────────────────────────────

async def llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    agent: str = "",
) -> str:
    """Non-streaming call. Returns the full response string.

    Args:
        prompt: The user message.
        system: System prompt. Defaults to a generic helpful-assistant prompt.
        agent:  Optional agent name for per-agent model routing
                ("planner", "coder", "tester", "debugger", "critic").
                If omitted the DEFAULT_MODEL env var (or gpt-4o) is used.
    """
    model = _resolve_model(agent)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    
    # Use LiteLLM mode (default)
    if LLM_MODE == "litellm":
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=0.0,
            seed=42,
        )
        return response.choices[0].message.content or ""
    
    # Use OpenAI-compatible client (openai, deepseek, local)
    client = _get_client()
    if client:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(f"LLM call failed in {LLM_MODE} mode: {exc}")
            return ""
    
    logger.error(f"Invalid LLM_MODE: {LLM_MODE}")
    return ""


async def llm_stream(
    prompt: str,
    system: str = "You are a helpful assistant.",
    agent: str = "",
) -> AsyncGenerator[str, None]:
    """Streaming call. Yields tokens as they arrive from the API.

    Args:
        prompt: The user message.
        system: System prompt.
        agent:  Optional agent name for per-agent model routing.
    """
    model = _resolve_model(agent)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    
    # Use LiteLLM mode (default)
    if LLM_MODE == "litellm":
        stream = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=0.0,
            seed=42,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        return
    
    # Use OpenAI-compatible client (openai, deepseek, local)
    client = _get_client()
    if client:
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.error(f"LLM streaming failed in {LLM_MODE} mode: {exc}")
            yield f"Error: {exc}"
        return
    
    logger.error(f"Invalid LLM_MODE: {LLM_MODE}")
    yield "Error: Invalid LLM configuration"
