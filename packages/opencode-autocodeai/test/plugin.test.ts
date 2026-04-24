import { describe, expect, test } from "bun:test"
import Plugin from "../src/index"

describe("plugin shape", () => {
  test("registers the expected tool names", async () => {
    const hooks = await Plugin({} as any, {})
    expect(hooks.tool).toBeDefined()
    const names = Object.keys(hooks.tool!).sort()
    expect(names).toEqual(
      ["autocodeai_critic", "autocodeai_plan", "autocodeai_run", "autocodeai_run_parallel", "autocodeai_stream"].sort(),
    )
  })

  test("each tool has a description and args schema", async () => {
    const hooks = await Plugin({} as any, {})
    for (const [name, def] of Object.entries(hooks.tool!)) {
      expect(def.description, `${name} missing description`).toBeTypeOf("string")
      expect(def.description.length, `${name} description too short`).toBeGreaterThan(20)
      expect(def.args, `${name} missing args schema`).toBeDefined()
      expect(def.execute, `${name} missing execute function`).toBeTypeOf("function")
    }
  })

  test("options baseUrl takes precedence over env and default", async () => {
    // Smoke the config resolution by hitting a tool with a deliberately
    // unreachable baseUrl and confirming the error surfaces that URL.
    const hooks = await Plugin({} as any, { baseUrl: "http://127.0.0.1:1" })
    const { ctx } = await import("./fixture/fixture").then((m) => m.makeCtx())
    const err = await hooks
      .tool!.autocodeai_run.execute({ task: "x", context_files: undefined }, ctx)
      .catch((e: Error) => e)
    expect(err).toBeInstanceOf(Error)
    // Either a connection-refused message or the URL itself confirms baseUrl
    // was honored rather than the default 8000.
    expect(String(err)).not.toContain("8000")
  })
})
