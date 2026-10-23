// LIFESPAN: permanent
import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { ModelError, OllamaClient } from "../src/model.ts";

type Handler = (body: Record<string, unknown>, res: ServerResponse) => void;

async function withServer(handler: Handler, fn: (client: OllamaClient, seen: Record<string, unknown>[]) => Promise<void>) {
  const seen: Record<string, unknown>[] = [];
  const server: Server = createServer((req: IncomingMessage, res: ServerResponse) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      const body = JSON.parse(raw) as Record<string, unknown>;
      seen.push(body);
      handler(body, res);
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const port = (server.address() as AddressInfo).port;
  try {
    await fn(new OllamaClient("test-model", { baseUrl: `http://127.0.0.1:${port}` }), seen);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((r) => server.close(() => r()));
  }
}

const ok = (res: ServerResponse, payload: unknown) => {
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(payload));
};

test("request: stream false, num_ctx always sent, temperature only when set, 0 is sent as 0", async () => {
  await withServer((_b, res) => ok(res, { message: { content: "hi" }, prompt_eval_count: 7, eval_count: 2, done_reason: "stop" }), async (client, seen) => {
    await client.chat([{ role: "user", content: "x" }], [], { numCtx: 4096, timeoutMs: 2000 });
    await client.chat([{ role: "user", content: "x" }], [], { numCtx: 4096, temperature: 0, timeoutMs: 2000 });
    const unset = seen[0]?.["options"] as Record<string, unknown>;
    const zero = seen[1]?.["options"] as Record<string, unknown>;
    assert.equal(seen[0]?.["stream"], false);
    assert.equal(unset["num_ctx"], 4096);
    assert.ok(!("temperature" in unset), "unset sends no temperature at all");
    assert.equal(zero["temperature"], 0);
  });
});

test("response: content, usage and finish reason are mapped; a native tool call is parsed", async () => {
  const payload = { message: { content: "", tool_calls: [{ function: { name: "calculator", arguments: { expression: "1+1" } } }] }, prompt_eval_count: 50, eval_count: 9, done_reason: "stop" };
  await withServer((_b, res) => ok(res, payload), async (client, seen) => {
    const r = await client.chat([{ role: "user", content: "x" }], [{ name: "calculator", description: "d", parameters: {} }], { numCtx: 2048, timeoutMs: 2000 });
    assert.deepEqual(r.toolCalls.map((c) => [c.name, c.arguments]), [["calculator", { expression: "1+1" }]]);
    assert.deepEqual(r.usage, { promptTokens: 50, completionTokens: 9 });
    assert.equal(r.finishReason, "tool_calls");
    assert.equal((seen[0]?.["tools"] as unknown[]).length, 1);
  });
});

test("response: done_reason length comes through as the finish reason", async () => {
  await withServer((_b, res) => ok(res, { message: { content: "" }, done_reason: "length" }), async (client) => {
    const r = await client.chat([{ role: "user", content: "x" }], [], { numCtx: 2048, timeoutMs: 2000 });
    assert.equal(r.finishReason, "length");
  });
});

test("errors: 500 is retryable, 400 is not, a non-JSON body is malformed, a slow server times out", async () => {
  const cases: [Handler, (e: ModelError) => void][] = [
    [(_b, res) => { res.statusCode = 500; res.end("boom"); }, (e) => assert.ok(e.kind === "http" && e.status === 500 && e.retryable)],
    [(_b, res) => { res.statusCode = 400; res.end("bad"); }, (e) => assert.ok(e.kind === "http" && !e.retryable)],
    [(_b, res) => { res.end("<html>"); }, (e) => assert.ok(e.kind === "malformed" && !e.retryable)],
    [() => {}, (e) => assert.ok(e.kind === "timeout" && e.retryable)],
  ];
  for (const [handler, verify] of cases) {
    await withServer(handler, async (client) => {
      await assert.rejects(client.chat([{ role: "user", content: "x" }], [], { numCtx: 2048, timeoutMs: 150 }), (e: unknown) => {
        assert.ok(e instanceof ModelError);
        verify(e);
        return true;
      });
    });
  }
});

test("errors: a refused connection is a retryable transport error", async () => {
  const client = new OllamaClient("m", { baseUrl: "http://127.0.0.1:1" });
  await assert.rejects(client.chat([{ role: "user", content: "x" }], [], { numCtx: 2048, timeoutMs: 1000 }), (e: unknown) => e instanceof ModelError && e.kind === "transport" && e.retryable);
});
