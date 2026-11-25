import { afterEach, beforeEach, describe, expect, test } from "bun:test"
import Plugin from "../src/index"
import { makeCtx, startBackend, type MockBackend } from "./fixture/fixture"

let backend: MockBackend

beforeEach(async () => {
  backend = await startBackend({
    "/api/agent/run": (_req, body: any) => {
      // Echo the requested task back as a structured result so tests
      // can verify both the request body and the response formatting.
      return Response.json({
        results: [
          { step: "Plan", output: `planning: ${body.task}`, type: "plan" },
          { step: "Code", output: "def foo(): pass", type: "code" },
          "a plain string step",
        ],
      })
    },
    "/api/agent/run_parallel": (_req, body: any) => {
      return Response.json({
        results: body.steps.map((s: any, i: number) => ({
          step: `#${i}`,
          output: `${s.agent} done`,
          type: s.agent,
        })),
      })
    },
    "/api/agent/fail": () => new Response("backend exploded", { status: 500 }),
  })
})

afterEach(async () => {
  await backend.stop()
})

describe("autocodeai_run", () => {
  test("POSTs to /api/agent/run with correct body and formats results", async () => {
    const hooks = await Plugin({} as any, { baseUrl: backend.url })
    const { ctx, metadata } = makeCtx()

    const result = await hooks.tool!.autocodeai_run.execute(
      { task: "write a hello world", context_files: ["src/app.ts"] },
      ctx,
    )

    // Request shape
    expect(backend.requests).toHaveLength(1)
    expect(backend.requests[0]!.method).toBe("POST")
    expect(backend.requests[0]!.path).toBe("/api/agent/run")
    expect(backend.requests[0]!.body).toEqual({
      task: "write a hello world",
      context_files: ["src/app.ts"],
    })

    // Result formatting: headers, type tags, plain strings all rendered
    const output = typeof result === "string" ? result : result.output
    expect(output).toContain("## Plan _[plan]_")
    expect(output).toContain("planning: write a hello world")
    expect(output).toContain("## Code _[code]_")
    expect(output).toContain("def foo(): pass")
    expect(output).toContain("a plain string step")

    // Metadata: title set, step count reported
    expect(metadata[0]!.title).toContain("write a hello world")
    const last = typeof result === "string" ? undefined : result.metadata
    expect(last).toEqual({ endpoint: "/api/agent/run", steps: 3 })
  })

  test("defaults context_files to empty array when omitted", async () => {
    const hooks = await Plugin({} as any, { baseUrl: backend.url })
    const { ctx } = makeCtx()
    await hooks.tool!.autocodeai_run.execute({ task: "t", context_files: undefined }, ctx)
    expect((backend.requests[0]!.body as any).context_files).toEqual([])
  })
})

describe("autocodeai_run_parallel", () => {
  test("forwards steps array unchanged", async () => {
    const hooks = await Plugin({} as any, { baseUrl: backend.url })
    const { ctx } = makeCtx()
    const steps = [
      { agent: "coder" as const, description: "build model" },
      { agent: "tester" as const, description: "write tests" },
    ]
    const result = await hooks.tool!.autocodeai_run_parallel.execute(
      { steps, context_files: ["main.py"] },
      ctx,
    )
    expect(backend.requests[0]!.path).toBe("/api/agent/run_parallel")
    expect((backend.requests[0]!.body as any).steps).toEqual(steps)
    const output = typeof result === "string" ? result : result.output
    expect(output).toContain("coder done")
    expect(output).toContain("tester done")
  })
})

describe("autocodeai_plan", () => {
  test("delegates to run_parallel with a single planner step", async () => {
    const hooks = await Plugin({} as any, { baseUrl: backend.url })
    const { ctx } = makeCtx()
    await hooks.tool!.autocodeai_plan.execute({ task: "refactor auth", context_files: undefined }, ctx)
    const body = backend.requests[0]!.body as any
    expect(backend.requests[0]!.path).toBe("/api/agent/run_parallel")
    expect(body.steps).toHaveLength(1)
    expect(body.steps[0].agent).toBe("planner")
    expect(body.steps[0].description).toBe("refactor auth")
  })
})

describe("autocodeai_critic", () => {
  test("targets critic agent and includes file in context", async () => {
    const hooks = await Plugin({} as any, { baseUrl: backend.url })
    const { ctx } = makeCtx()
    await hooks.tool!.autocodeai_critic.execute(
      { description: "check security", file: "src/auth.py" },
      ctx,
    )
    const body = backend.requests[0]!.body as any
    expect(body.steps[0].agent).toBe("critic")
    expect(body.steps[0].file).toBe("src/auth.py")
    expect(body.context_files).toEqual(["src/auth.py"])
  })
})

describe("error handling", () => {
  test("non-2xx response surfaces as a thrown Error with status and body", async () => {
    const failingBackend = await startBackend({
      "/api/agent/run": () => new Response("backend exploded", { status: 500 }),
    })
    const hooks = await Plugin({} as any, { baseUrl: failingBackend.url })
    const { ctx } = makeCtx()
    const err = await hooks
      .tool!.autocodeai_run.execute({ task: "x", context_files: undefined }, ctx)
      .catch((e: Error) => e)
    expect(err).toBeInstanceOf(Error)
    expect(String(err)).toContain("500")
    expect(String(err)).toContain("backend exploded")
    await failingBackend.stop()
  })

  test("caller cancellation propagates through AbortSignal", async () => {
    // Use a shorter hang (1s) so the test fits comfortably inside bun's
    // 5s default timeout even if the abort somehow fails to fire.
    const slowBackend = await startBackend({
      "/api/agent/run": async () => {
        await new Promise((r) => setTimeout(r, 1000))
        return Response.json({ results: [] })
      },
    })
    const hooks = await Plugin({} as any, { baseUrl: slowBackend.url })
    const ctrl = new AbortController()
    const { ctx } = makeCtx(ctrl.signal)

    const started = Date.now()
    const promise = hooks.tool!.autocodeai_run.execute({ task: "slow", context_files: undefined }, ctx)
    setTimeout(() => ctrl.abort(new Error("user-cancelled")), 20)
    const err = await promise.catch((e: Error) => e)
    const elapsed = Date.now() - started
    expect(err).toBeInstanceOf(Error)
    // Must return well before the 1000ms hang — i.e. abort actually cancelled
    expect(elapsed).toBeLessThan(500)
    await slowBackend.stop()
  })

  test("plugin-level timeout fires when backend hangs", async () => {
    const hangBackend = await startBackend({
      "/api/agent/run": async () => {
        await new Promise((r) => setTimeout(r, 1000))
        return Response.json({ results: [] })
      },
    })
    const hooks = await Plugin({} as any, { baseUrl: hangBackend.url, timeoutMs: 50 })
    const { ctx } = makeCtx()
    const started = Date.now()
    const err = await hooks
      .tool!.autocodeai_run.execute({ task: "slow", context_files: undefined }, ctx)
      .catch((e: Error) => e)
    const elapsed = Date.now() - started
    expect(err).toBeInstanceOf(Error)
    // Timeout fired in ~50ms, not the 1000ms the backend would take
    expect(elapsed).toBeLessThan(500)
    await hangBackend.stop()
  })
})
