"""
tests/test_llm_retry.py — Retry, backoff, and error classification.

A full sweep issues on the order of tens of thousands of requests, so
transient rate limits are a certainty. Before this, a single 429 propagated
out of ``llm_call``, the runner caught it at instance level, and that instance
was recorded as an error — which does not merely lose data, it biases the
comparison toward whichever configuration happened to run at a quieter time.

These tests cover the classification boundaries that matter: what is retried,
what is not, and that a context-window overflow is surfaced as its own type
rather than consuming the retry budget on a request that cannot succeed.

Run:
    python -m pytest tests/test_llm_retry.py -v
"""

from __future__ import annotations

import asyncio

import pytest

from core.utils import llm as llm_module
from core.utils.llm import ContextTooLong, _classify, _with_retries


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    """Keep the suite quick; the delay arithmetic is asserted separately."""
    monkeypatch.setenv("LLM_BACKOFF_BASE_S", "0.001")
    monkeypatch.setenv("LLM_BACKOFF_CAP_S", "0.002")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("message", [
    "Rate limit reached for gpt-4o",
    "Request timed out",
    "upstream connection error",
    "Error code: 503 Service Unavailable",
    "Overloaded, please retry",
])
def test_transient_failures_are_retried(message):
    assert _classify(RuntimeError(message)) == "retry"


@pytest.mark.parametrize("message", [
    "This model's maximum context length is 128000 tokens",
    "context length exceeded",
])
def test_context_overflow_is_its_own_category(message):
    assert _classify(RuntimeError(message)) == "context"


def test_unknown_errors_default_to_retry():
    """Losing an instance to an unrecognised hiccup is worse than one retry."""
    assert _classify(RuntimeError("something odd happened")) == "retry"


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_succeeds_without_retry_when_the_call_works():
    calls = []

    async def call():
        calls.append(1)
        return "ok"

    assert run(_with_retries(call, what="t")) == "ok"
    assert len(calls) == 1


def test_retries_then_succeeds():
    calls = []

    async def call():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Rate limit reached")
        return "ok"

    assert run(_with_retries(call, what="t")) == "ok"
    assert len(calls) == 3, "should have retried twice before succeeding"


def test_gives_up_after_max_attempts():
    calls = []

    async def call():
        calls.append(1)
        raise RuntimeError("Rate limit reached")

    with pytest.raises(RuntimeError):
        run(_with_retries(call, what="t"))
    assert len(calls) == 5, "should stop at LLM_MAX_RETRIES"


def test_context_overflow_raises_immediately_without_burning_retries():
    calls = []

    async def call():
        calls.append(1)
        raise RuntimeError("maximum context length is 128000 tokens")

    with pytest.raises(ContextTooLong):
        run(_with_retries(call, what="t"))
    assert len(calls) == 1, (
        "a prompt that is too long will still be too long on retry; "
        "retrying it wastes the budget for genuinely transient failures"
    )


def test_backoff_grows_and_is_capped(monkeypatch):
    delays: list[float] = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(llm_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("LLM_BACKOFF_BASE_S", "2.0")
    monkeypatch.setenv("LLM_BACKOFF_CAP_S", "10.0")

    async def call():
        raise RuntimeError("Rate limit reached")

    with pytest.raises(RuntimeError):
        run(_with_retries(call, what="t"))

    assert len(delays) == 4, "four waits between five attempts"
    assert all(d <= 10.0 for d in delays), "cap not honoured"
    assert all(d > 0 for d in delays)
    # Jitter is multiplicative in [0.5, 1.5), so compare against the floor of
    # each step rather than an exact value.
    assert delays[1] > delays[0] * 0.5, "backoff should grow"
