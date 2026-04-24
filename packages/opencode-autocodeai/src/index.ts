/**
 * @opencode-ai/autocodeai
 *
 * OpenCode plugin that proxies coding tasks to an AutoCodeAI multi-agent
 * backend over HTTP/SSE. Designed to be additive: does not modify OpenCode
 * core and does not modify AutoCodeAI. Both processes stay in their native
 * language (TypeScript/Bun on the OpenCode side, Python/FastAPI on the
 * AutoCodeAI side) and talk over localhost HTTP.
 *
 * Registered tools:
 *   - autocodeai_run           POST /api/agent/run          sync full pipeline
 *   - autocodeai_run_parallel  POST /api/agent/run_parallel concurrent steps
 *   - autocodeai_stream        POST /api/agent/stream       SSE streaming
 *   - autocodeai_plan          planner-only convenience wrapper
 *   - autocodeai_critic        critic-only convenience wrapper
 *
 * Configuration (plugin options, or env var fallbacks):
 *   - baseUrl   / AUTOCODEAI_URL       default "http://localhost:8000"
 *   - timeoutMs / AUTOCODEAI_TIMEOUT   default 300000  (5 min)
 */

import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"

type Options = {
  baseUrl?: string
  timeoutMs?: number
}

// Shape the AutoCodeAI orchestrator returns in its `results` array.
// From services/orchestrator.py: items are either strings or
// { step, output, type } objects. We treat it permissively.
type AgentResultItem =
  | string
  | {
      step?: string
      output?: unknown
      type?: string
      [k: string]: unknown
    }

const AutoCodeAIPlugin: Plugin = async (_input, options) => {
  const opts = (options ?? {}) as Options
  const baseUrl = (opts.baseUrl ?? process.env.AUTOCODEAI_URL ?? "http://localhost:8000").replace(
    /\/+$/,
    "",
  )
  const envTimeout = process.env.AUTOCODEAI_TIMEOUT ? Number(process.env.AUTOCODEAI_TIMEOUT) : undefined
  const timeoutMs = opts.timeoutMs ?? (envTimeout && !Number.isNaN(envTimeout) ? envTimeout : undefined) ?? 300_000

  // Combine the caller-provided AbortSignal with a plugin-level timeout.
  // Cancellation propagates both ways: opencode abort → fetch abort,
  // and timeout → fetch abort.
  const withTimeout = (abort: AbortSignal): { signal: AbortSignal; cleanup: () => void } => {
    const ctrl = new AbortController()
    const onAbort = () => ctrl.abort(abort.reason)
    if (abort.aborted) ctrl.abort(abort.reason)
    else abort.addEventListener("abort", onAbort, { once: true })
    const id = setTimeout(() => ctrl.abort(new Error(`AutoCodeAI timeout after ${timeoutMs}ms`)), timeoutMs)
    return {
      signal: ctrl.signal,
      cleanup: () => {
        clearTimeout(id)
        abort.removeEventListener("abort", onAbort)
      },
    }
  }

  const postJson = async <T>(path: string, body: unknown, abort: AbortSignal): Promise<T> => {
    const { signal, cleanup } = withTimeout(abort)
    const res = await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    }).finally(cleanup)
    if (!res.ok) {
      const text = await res.text().catch(() => "")
      throw new Error(`AutoCodeAI ${path} → ${res.status} ${res.statusText}: ${text.slice(0, 500)}`)
    }
    return (await res.json()) as T
  }

  // Render the orchestrator's `results` array into a readable transcript
  // for the calling model. Each entry becomes a fenced section so the
  // model can cite individual steps without re-parsing JSON.
  const formatResults = (results: unknown): string => {
    if (typeof results === "string") return results
    if (!Array.isArray(results)) return JSON.stringify(results, null, 2)
    return results
      .map((r: AgentResultItem, i) => {
        if (typeof r === "string") return `## Step ${i + 1}\n\n${r}`
        const header = r.step ? r.step : `Step ${i + 1}`
        const typeTag = r.type ? ` _[${r.type}]_` : ""
        const body = typeof r.output === "string" ? r.output : JSON.stringify(r.output ?? r, null, 2)
        return `## ${header}${typeTag}\n\n${body}`
      })
      .join("\n\n---\n\n")
  }

  // Parse AutoCodeAI's SSE stream. routes.py encodes embedded newlines
  // in data fields as the "↵" character because sse-starlette can't
  // transmit literal \n inside a data: line; we decode them back.
  const readSseStream = async (
    res: Response,
    onChunk: (text: string, totalBytes: number) => void,
  ): Promise<{ chunks: string[]; bytes: number }> => {
    if (!res.body) throw new Error("AutoCodeAI stream returned no body")
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    const chunks: string[] = []
    let buffer = ""
    let bytes = 0

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      bytes += value.byteLength
      buffer += decoder.decode(value, { stream: true })
      const frames = buffer.split("\n\n")
      buffer = frames.pop() ?? ""
      for (const frame of frames) {
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue
          const decoded = line.slice(5).trimStart().replace(/↵/g, "\n")
          chunks.push(decoded)
          onChunk(decoded, bytes)
        }
      }
    }
    // Flush trailing partial frame if it is a full data: line.
    if (buffer.startsWith("data:")) {
      const decoded = buffer.slice(5).trimStart().replace(/↵/g, "\n")
      chunks.push(decoded)
      onChunk(decoded, bytes)
    }
    return { chunks, bytes }
  }

  return {
    tool: {
      autocodeai_run: tool({
        description:
          "Run a coding task through the full AutoCodeAI multi-agent pipeline (Planner → Coder → Tester → Debugger → Critic) and return the complete transcript. Use this when you want a task completed end-to-end with validation.",
        args: {
          task: tool.schema.string().describe("Natural-language coding task to execute"),
          context_files: tool.schema
            .array(tool.schema.string())
            .optional()
            .describe("Paths (relative to project root) to include as context"),
        },
        async execute(args, ctx) {
          ctx.metadata({ title: `AutoCodeAI: ${args.task.slice(0, 80)}` })
          const { results } = await postJson<{ results: AgentResultItem[] }>(
            "/api/agent/run",
            { task: args.task, context_files: args.context_files ?? [] },
            ctx.abort,
          )
          return {
            output: formatResults(results),
            metadata: {
              endpoint: "/api/agent/run",
              steps: Array.isArray(results) ? results.length : 1,
            },
          }
        },
      }),

      autocodeai_run_parallel: tool({
        description:
          "Run multiple independent AutoCodeAI steps concurrently (bounded by the backend's MAX_PARALLEL_WORKERS, default 3). Each step targets a specific agent (coder/tester/debugger/critic/planner) or an allowlisted tool (git_clone, pip_install, shell). Use this when you have several orthogonal changes to make and want them parallelized.",
        args: {
          steps: tool.schema
            .array(
              tool.schema.object({
                agent: tool.schema.enum(["planner", "coder", "tester", "debugger", "critic", "tool"] as const),
                description: tool.schema.string().optional(),
                file: tool.schema.string().optional(),
                tool_name: tool.schema.string().optional(),
                tool_params: tool.schema.record(tool.schema.string(), tool.schema.any()).optional(),
              }),
            )
            .describe("Steps to execute concurrently"),
          context_files: tool.schema.array(tool.schema.string()).optional(),
        },
        async execute(args, ctx) {
          ctx.metadata({ title: `AutoCodeAI parallel: ${args.steps.length} steps` })
          const { results } = await postJson<{ results: AgentResultItem[] }>(
            "/api/agent/run_parallel",
            { steps: args.steps, context_files: args.context_files ?? [] },
            ctx.abort,
          )
          return {
            output: formatResults(results),
            metadata: { endpoint: "/api/agent/run_parallel", steps: args.steps.length },
          }
        },
      }),

      autocodeai_stream: tool({
        description:
          "Run a coding task through the AutoCodeAI pipeline with Server-Sent Events streaming. Progress is surfaced via tool metadata while it runs; the full transcript is returned on completion. Use this for long-running tasks where you want live visibility.",
        args: {
          task: tool.schema.string().describe("Natural-language coding task to execute"),
          context_files: tool.schema.array(tool.schema.string()).optional(),
        },
        async execute(args, ctx) {
          ctx.metadata({ title: `AutoCodeAI (stream): ${args.task.slice(0, 80)}` })
          const { signal, cleanup } = withTimeout(ctx.abort)
          const res = await fetch(`${baseUrl}/api/agent/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
            body: JSON.stringify({ task: args.task, context_files: args.context_files ?? [] }),
            signal,
          }).catch((err: Error) => {
            cleanup()
            throw err
          })
          if (!res.ok) {
            cleanup()
            const text = await res.text().catch(() => "")
            throw new Error(`AutoCodeAI /api/agent/stream → ${res.status}: ${text.slice(0, 500)}`)
          }
          const result = await readSseStream(res, (_chunk, totalBytes) =>
            ctx.metadata({ metadata: { bytes: totalBytes } }),
          ).finally(cleanup)
          return {
            output: result.chunks.join(""),
            metadata: {
              endpoint: "/api/agent/stream",
              chunks: result.chunks.length,
              bytes: result.bytes,
            },
          }
        },
      }),

      autocodeai_plan: tool({
        description:
          "Planner-only: ask AutoCodeAI to decompose a task into a JSON step list without executing it. Useful for previewing what the pipeline would do before committing to a full run.",
        args: {
          task: tool.schema.string().describe("Task to plan"),
          context_files: tool.schema.array(tool.schema.string()).optional(),
        },
        async execute(args, ctx) {
          const { results } = await postJson<{ results: AgentResultItem[] }>(
            "/api/agent/run_parallel",
            {
              steps: [{ agent: "planner", description: args.task }],
              context_files: args.context_files ?? [],
            },
            ctx.abort,
          )
          return { output: formatResults(results), metadata: { agent: "planner" } }
        },
      }),

      autocodeai_critic: tool({
        description:
          "Critic-only: ask AutoCodeAI's Critic agent to review code and return PASS/FAIL with suggestions. Does not modify files.",
        args: {
          description: tool.schema.string().describe("What to review and on what criteria"),
          file: tool.schema.string().optional().describe("File to review, relative to project root"),
        },
        async execute(args, ctx) {
          const { results } = await postJson<{ results: AgentResultItem[] }>(
            "/api/agent/run_parallel",
            {
              steps: [{ agent: "critic", description: args.description, file: args.file }],
              context_files: args.file ? [args.file] : [],
            },
            ctx.abort,
          )
          return { output: formatResults(results), metadata: { agent: "critic" } }
        },
      }),
    },
  }
}

export default AutoCodeAIPlugin
export { AutoCodeAIPlugin }
