/**
 * client.js — Browser helpers for the AI Coder backend.
 *
 * streamTask()  — streams output via SSE (POST-based fetch + ReadableStream)
 * AgentSocket   — bidirectional WebSocket client
 */

const BASE_URL = "http://localhost:8000/api";

// ---------------------------------------------------------------------------
// SSE streaming (POST body, not GET query param)
// ---------------------------------------------------------------------------

/**
 * @param {string}   task
 * @param {string[]} contextFiles
 * @param {(chunk: string) => void} onChunk
 * @param {() => void}              onDone
 */
export async function streamTask(task, contextFiles = [], onChunk, onDone) {
  const response = await fetch(`${BASE_URL}/agent/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task, context_files: contextFiles }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  const reader  = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const raw = decoder.decode(value);
    for (const line of raw.split("\n")) {
      if (line.startsWith("data: ")) {
        onChunk(line.slice(6).replace(/↵/g, "\n"));
      }
    }
  }
  onDone?.();
}

// ---------------------------------------------------------------------------
// WebSocket — bidirectional
// ---------------------------------------------------------------------------

export class AgentSocket {
  /**
   * @param {(msg: string) => void} onMessage
   * @param {() => void}            onDone
   * @param {(err: Event) => void}  onError
   */
  constructor(onMessage, onDone, onError) {
    this.ws        = new WebSocket(`ws://localhost:8000/api/ws`);
    this.onMessage = onMessage;
    this.onDone    = onDone;
    this.onError   = onError;

    this.ws.onmessage = (event) => {
      if (event.data === "__DONE__") {
        this.onDone?.();
      } else {
        this.onMessage(event.data);
      }
    };
    this.ws.onerror = (e) => this.onError?.(e);
  }

  /**
   * @param {string}   task
   * @param {string[]} contextFiles
   */
  send(task, contextFiles = []) {
    this.ws.send(JSON.stringify({ task, context_files: contextFiles }));
  }

  close() {
    this.ws.close();
  }
}
