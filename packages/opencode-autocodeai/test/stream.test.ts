import { afterEach, beforeEach, describe, expect, test } from "bun:test"
import Plugin from "../src/index"
import { makeCtx, startBackend, type MockBackend } from "./fixture/fixture"

let backend: MockBackend

afterEach(async () => {
  if (backend) await backend.stop()
})

describe("autocodeai_stream SSE parsing", () => {
  test("concatenates data frames and decodes ↵ back to newlines", async () => {
    // AutoCodeAI's /api/agent/stream encodes embedded \n as ↵ because
    // SSE data: lines can't contain literal newlines. The plugin must
    // reverse that encoding.
    backend = await startBackend({
      "/api/agent/stream": () =>
        new Response(
          "data: Planning step 1...↵\n\n" +
            "data: def binary_search(arr, target):↵    low, high = 0, len(arr) - 1↵\n\n" +
            "data: ✅ Done.\n\n",
          { headers: { "Content-Type": "text/event-stream" } },
        ),
    })

    const hooks = await Plugin({} as any, { baseUrl: backend.url })
    const { ctx, metadata } = makeCtx()
    const result = await hooks.tool!.autocodeai_stream.execute(
      { task: "binary search", context_files: undefined },
      ctx,
    )
    const output = typeof result === "string" ? result : result.output

    // Each ↵ decoded to \n. Chunks concatenated.
    expect(output).toContain("Planning step 1...\n")
    expect(output).toContain("def binary_search(arr, target):\n")
    expect(output).toContain("    low, high = 0, len(arr) - 1\n")
    expect(output).toContain("✅ Done.")

    // Metadata should include chunk count and bytes
    const finalMeta = typeof result === "string" ? undefined : result.metadata
    expect((finalMeta as any).endpoint).toBe("/api/agent/stream")
    expect((finalMeta as any).chunks).toBeGreaterThan(0)
    expect((finalMeta as any).bytes).toBeGreaterThan(0)

    // Progress metadata should fire at least once during streaming
    expect(metadata.some((m) => m.metadata && "bytes" in m.metadata)).toBe(true)
  })

  test("handles frame boundaries that split across chunks", async () => {
    // Stream bytes in pieces smaller than a frame to verify the buffering
    // logic correctly reassembles across reads.
    backend = await startBackend({
      "/api/agent/stream": () => {
        const frames = ["data: alpha\n\n", "data: beta\n\n", "data: gamma\n\n"]
        const full = frames.join("")
        const bytes = new TextEncoder().encode(full)
        const stream = new ReadableStream({
          async start(controller) {
            // Emit in odd-sized slices that cut across frame boundaries
            const sizes = [3, 7, 11, 4, 5, 8, bytes.length]
            let pos = 0
            for (const size of sizes) {
              const end = Math.min(pos + size, bytes.length)
              if (pos >= end) break
              controller.enqueue(bytes.slice(pos, end))
              pos = end
              await new Promise((r) => setTimeout(r, 2))
            }
            controller.close()
          },
        })
        return new Response(stream, { headers: { "Content-Type": "text/event-stream" } })
      },
    })

    const hooks = await Plugin({} as any, { baseUrl: backend.url })
    const { ctx } = makeCtx()
    const result = await hooks.tool!.autocodeai_stream.execute(
      { task: "test", context_files: undefined },
      ctx,
    )
    const output = typeof result === "string" ? result : result.output
    expect(output).toBe("alphabetagamma")
  })

  test("surfaces non-2xx stream response as an error", async () => {
    backend = await startBackend({
      "/api/agent/stream": () => new Response("unavailable", { status: 503 }),
    })
    const hooks = await Plugin({} as any, { baseUrl: backend.url })
    const { ctx } = makeCtx()
    const err = await hooks
      .tool!.autocodeai_stream.execute({ task: "x", context_files: undefined }, ctx)
      .catch((e: Error) => e)
    expect(err).toBeInstanceOf(Error)
    expect(String(err)).toContain("503")
  })
})
