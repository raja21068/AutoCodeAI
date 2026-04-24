/**
 * Fixture for opencode-autocodeai tests.
 *
 * Provides a real HTTP server standing in for AutoCodeAI (via Bun.serve)
 * and a minimal ToolContext builder. We use a real server rather than
 * mocking fetch so the tests exercise the actual network path, matching
 * the repo guidance: "Avoid mocks as much as possible. Test actual
 * implementation, do not duplicate logic into tests."
 */

import type { Server } from "bun"

export type Route = (req: Request, body: unknown) => Response | Promise<Response>

export type MockBackend = {
  url: string
  requests: Array<{ method: string; path: string; body: unknown }>
  stop: () => Promise<void>
}

/**
 * Spin up a Bun.serve instance on a random port with a per-path route map.
 * Each request is recorded in `requests` so tests can assert on what the
 * plugin sent. The server must be `.stop()`-ed at the end of the test.
 *
 * Note: we consume the request body once (to record it) and pass the parsed
 * value to handlers as the second argument, because Request bodies are
 * single-use streams — re-reading via req.clone() isn't reliable.
 */
export async function startBackend(routes: Record<string, Route>): Promise<MockBackend> {
  const requests: MockBackend["requests"] = []

  const server: Server = Bun.serve({
    port: 0,
    async fetch(req) {
      const url = new URL(req.url)
      const text = req.method === "POST" ? await req.text() : ""
      const body = text ? safeJsonParse(text) : null
      requests.push({ method: req.method, path: url.pathname, body })

      const handler = routes[url.pathname]
      if (!handler) return new Response("not found", { status: 404 })
      return handler(req, body)
    },
  })

  return {
    url: `http://${server.hostname}:${server.port}`,
    requests,
    async stop() {
      await server.stop(true)
    },
  }
}

function safeJsonParse(s: string): unknown {
  try {
    return JSON.parse(s)
  } catch {
    return s
  }
}

/**
 * Build a minimal ToolContext that satisfies the type and captures
 * metadata calls for assertions. The returned `metadata` array is
 * mutated in place as the tool emits updates.
 */
export function makeCtx(abort?: AbortSignal): {
  ctx: any
  metadata: Array<{ title?: string; metadata?: Record<string, unknown> }>
} {
  const metadata: Array<{ title?: string; metadata?: Record<string, unknown> }> = []
  const ctx = {
    sessionID: "test-session",
    messageID: "test-message",
    agent: "test-agent",
    directory: "/tmp/test",
    worktree: "/tmp/test",
    abort: abort ?? new AbortController().signal,
    metadata: (input: { title?: string; metadata?: Record<string, unknown> }) => {
      metadata.push(input)
    },
    ask: () => ({ _tag: "Sync", i0: () => undefined }),
  }
  return { ctx, metadata }
}

/** Helper for SSE responses — joins data frames with the required terminator. */
export function sseResponse(frames: string[]): Response {
  return new Response(frames.map((f) => `data: ${f}`).join("\n\n") + "\n\n", {
    headers: { "Content-Type": "text/event-stream" },
  })
}
