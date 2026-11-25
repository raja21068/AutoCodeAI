"""
eval/check_provider.py
----------------------
Exit 0 if the configured provider will accept a request, non-zero otherwise.

Key presence is not the same as key usability. A key can be valid and still be
refused for an exhausted balance, a rate limit, or a suspended account, and
each of those looks identical to a working setup right up until the first
call. The grid run that this guards spent an hour producing empty patches for
eleven configurations after the balance ran out mid-run, because nothing
checked.

Costs one token.

Usage:
    python -m eval.check_provider          # quiet; exit code is the answer
    python -m eval.check_provider -v       # print the provider's reason
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe provider availability")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Probe through the project's own wrapper rather than calling litellm
    # directly. The wrapper is what a run actually uses, and it dispatches on
    # LLM_MODE: a direct litellm call ignores that and would report a local
    # Ollama deployment as unavailable while the runner used it happily -- or,
    # worse, report a hosted provider as reachable when the mode in .env
    # points somewhere else entirely. Importing it also loads .env.
    import asyncio

    from core.utils.llm import LLM_MODE, _resolve_model, llm_call

    import litellm

    litellm.suppress_debug_info = True
    for noisy in ("LiteLLM", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    model = os.getenv("LLM_MODEL") or _resolve_model("planner")
    try:
        result = asyncio.run(llm_call("hi", agent="planner", model=model))
    except Exception as exc:
        if args.verbose:
            print(f"UNAVAILABLE mode={LLM_MODE} model={model}: "
                  f"{type(exc).__name__}: {str(exc)[:200]}")
        return 1

    # llm_call swallows some failures and returns empty text rather than
    # raising, so an empty response counts as unavailable: a run against it
    # would produce nothing and still look healthy.
    if not (result.text or "").strip():
        if args.verbose:
            print(f"UNAVAILABLE mode={LLM_MODE} model={model}: empty response")
        return 1

    if args.verbose:
        print(f"AVAILABLE mode={LLM_MODE} model={model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
