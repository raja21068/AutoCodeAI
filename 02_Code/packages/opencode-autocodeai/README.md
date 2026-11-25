# @opencode-ai/autocodeai

OpenCode plugin that proxies coding tasks to an [AutoCodeAI](https://github.com/raja21068/AutoCodeAI) multi-agent backend over HTTP/SSE.

This plugin does **not** modify OpenCode's core, and it does **not** modify AutoCodeAI. Both processes stay in their native language — TypeScript/Bun for OpenCode, Python/FastAPI for AutoCodeAI — and communicate over localhost HTTP. That means you can update either side independently without breaking the integration.

## What it adds

Five tools become available to any OpenCode agent (primary or subagent):

| Tool                       | Endpoint                      | When to use                                                                           |
| -------------------------- | ----------------------------- | ------------------------------------------------------------------------------------- |
| `autocodeai_run`           | `POST /api/agent/run`         | Run a task end-to-end through the full pipeline and return the transcript.            |
| `autocodeai_run_parallel`  | `POST /api/agent/run_parallel`| Run multiple independent steps concurrently.                                          |
| `autocodeai_stream`        | `POST /api/agent/stream` (SSE)| Long-running task with live progress via tool metadata.                               |
| `autocodeai_plan`          | planner-only                  | Decompose a task into a step plan without executing.                                  |
| `autocodeai_critic`        | critic-only                   | Review code and get PASS/FAIL feedback without modifying files.                       |

## Prerequisites

- AutoCodeAI backend reachable at `http://localhost:8000` (or whatever `baseUrl` you configure). The repo ships `docker-compose.yml` that brings up the FastAPI server plus ChromaDB. See the top-level `INTEGRATION.md` for a one-command stack launcher.
- OpenCode installed and configured to load plugins.

## Enabling the plugin

In your OpenCode config (`~/.config/opencode/opencode.json` or project-local `.opencode/opencode.json`):

```json
{
  "plugin": ["@opencode-ai/autocodeai"]
}
```

With custom options:

```json
{
  "plugin": [
    [
      "@opencode-ai/autocodeai",
      { "baseUrl": "http://localhost:8000", "timeoutMs": 600000 }
    ]
  ]
}
```

Options fall back to environment variables: `AUTOCODEAI_URL`, `AUTOCODEAI_TIMEOUT`.

## Cancellation

All tools honor OpenCode's `AbortSignal`. Pressing Esc in the TUI, or any internal cancellation, aborts the in-flight fetch. A plugin-level timeout (default 5 minutes) aborts stuck requests on its own.

## Errors

Non-2xx responses from AutoCodeAI surface as tool errors with the status code and the first 500 characters of the response body, so the calling model can decide whether to retry, re-plan, or escalate. Connection refused (backend not running) surfaces as a standard `fetch` error — the plain remedy there is `docker compose up` for AutoCodeAI.
