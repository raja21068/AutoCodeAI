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

import asyncio
import logging
import os
import random
from pathlib import Path
from typing import AsyncGenerator

import litellm
import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Load .env here, before anything below reads os.getenv.
#
# Only main.py and experiments/setup_and_run.py used to do this, so running
# an evaluation entrypoint directly — `python -m eval.swebench_runner`,
# `python -m eval.smoke_test --live` — left DEEPSEEK_API_KEY unset. LiteLLM
# then sent no credential and the provider answered 401, which reads as "your
# key is invalid" rather than "your key was never loaded". A preflight that
# does call load_dotenv passes moments before the run that does not, so the
# failure looks intermittent and gets blamed on the provider.
#
# override=False so a real environment variable still wins over the file,
# which is what CI and per-run overrides depend on.
try:
    from dotenv import load_dotenv

    _ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(_ENV_PATH, override=False)
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    logger.debug("python-dotenv not installed; relying on the ambient environment")

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

class ContextTooLong(RuntimeError):
    """
    The prompt exceeded the model's context window.

    Raised as its own type because retrying is pointless — the caller must
    shrink the prompt. Conflating it with transient errors burns the retry
    budget on a request that cannot succeed.
    """


def _exception(name: str):
    """Look up a litellm exception defensively; names move between versions."""
    return getattr(litellm.exceptions, name, None)


def _classify(exc: Exception) -> str:
    """Return 'retry', 'context', or 'fatal' for *exc*."""
    for name in ("ContextWindowExceededError",):
        cls = _exception(name)
        if cls and isinstance(exc, cls):
            return "context"

    for name in ("AuthenticationError", "PermissionDeniedError",
                 "NotFoundError", "ContentPolicyViolationError"):
        cls = _exception(name)
        if cls and isinstance(exc, cls):
            return "fatal"

    for name in ("RateLimitError", "APIConnectionError", "Timeout",
                 "APIError", "InternalServerError", "ServiceUnavailableError"):
        cls = _exception(name)
        if cls and isinstance(exc, cls):
            return "retry"

    # Unknown failures are treated as transient once or twice rather than
    # losing a whole instance to a hiccup.
    message = str(exc).lower()
    if any(token in message for token in
           ("rate limit", "timeout", "timed out", "overloaded",
            "temporarily unavailable", "connection", "502", "503", "529")):
        return "retry"
    if "context length" in message or "maximum context" in message:
        return "context"
    return "retry"


async def _with_retries(call, *, what: str):
    """
    Run *call* with exponential backoff and jitter.

    A benchmark sweep issues on the order of tens of thousands of requests, so
    transient rate limits are certain rather than unlikely. Without this, one
    429 loses an entire instance and silently biases the run toward whichever
    configuration happened to hit a quieter moment.
    """
    max_attempts = int(os.getenv("LLM_MAX_RETRIES", "5"))
    base = float(os.getenv("LLM_BACKOFF_BASE_S", "2.0"))
    cap = float(os.getenv("LLM_BACKOFF_CAP_S", "60.0"))

    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await call()
        except Exception as exc:
            kind = _classify(exc)
            last = exc

            if kind == "context":
                raise ContextTooLong(str(exc)) from exc
            if kind == "fatal" or attempt == max_attempts:
                logger.error("%s failed (%s): %s", what, kind, exc)
                raise

            # Jitter first, then cap — capping first lets the jitter multiplier
            # push the delay back above the ceiling.
            delay = base * (2 ** (attempt - 1)) * (0.5 + random.random())
            delay = min(cap, delay)
            logger.warning("%s attempt %d/%d failed (%s); retrying in %.1fs",
                           what, attempt, max_attempts, type(exc).__name__, delay)
            await asyncio.sleep(delay)

    raise last if last else RuntimeError(f"{what} failed")


class LLMResult:
    """Text plus the accounting needed for budget-matched comparisons."""

    __slots__ = ("text", "model", "prompt_tokens", "completion_tokens", "cost_usd")

    def __init__(
        self,
        text: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.text = text
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost_usd = cost_usd

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


async def llm_call(
    prompt: str,
    system: str = "You are a helpful assistant.",
    agent: str = "",
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> LLMResult:
    """
    Non-streaming call that reports token usage and cost.

    Every controlled comparison in the paper needs per-call accounting, so
    benchmark code should call this rather than :func:`llm`. ``model``
    overrides per-agent routing, which is how a single-model configuration is
    enforced across all five roles.
    """
    resolved = model or _resolve_model(agent)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    timeout = float(os.getenv("LLM_TIMEOUT_S", "180"))

    if LLM_MODE == "litellm":
        async def _call():
            return await litellm.acompletion(
                model=resolved, messages=messages, temperature=temperature,
                seed=42, timeout=timeout,
            )
    else:
        client = _get_client()
        if client is None:
            logger.error("Invalid LLM_MODE: %s", LLM_MODE)
            return LLMResult("", resolved)

        async def _call():
            return await client.chat.completions.create(
                model=resolved, messages=messages, temperature=temperature,
                timeout=timeout,
            )

    response = await _with_retries(_call, what=f"llm_call({resolved}, {agent})")

    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        # Unknown/local model pricing — tokens are still recorded.
        cost = 0.0

    return LLMResult(text, resolved, prompt_tokens, completion_tokens, cost)


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
