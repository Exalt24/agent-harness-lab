import assert from "node:assert/strict";
import { test } from "node:test";
import { capToolResult, estimateTokens, fitMessages, readResult, ResultStore, STUB_MARKER, TRIM_NOTE_PREFIX } from "../src/context.ts";
import type { Message } from "../src/model.ts";

test("token estimate is ceil(chars / 4)", () => {
  assert.equal(estimateTokens(""), 0);
  assert.equal(estimateTokens("a"), 1);
  assert.equal(estimateTokens("abcd"), 1);
  assert.equal(estimateTokens("abcde"), 2);
});

test("byte cap: a result at the cap passes through untouched, and nothing is stored (positive control)", () => {
  const store = new ResultStore();
  const text = "x".repeat(500);
  const out = capToolResult(text, store, 500);
  assert.equal(out.text, text);
  assert.equal(out.stubbed, false);
  assert.equal(store.size, 0);
});

test("byte cap: one byte over becomes head + visible marker + tail, and the full text is kept (negative)", () => {
  const store = new ResultStore();
  const text = `HEAD${"m".repeat(2000)}TAIL`;
  const out = capToolResult(text, store, 400);
  assert.equal(out.stubbed, true);
  assert.ok(out.text.startsWith("HEAD"));
  assert.ok(out.text.endsWith("TAIL"));
  assert.ok(out.text.includes(STUB_MARKER), "the cut is visible, never silent");
  assert.ok(out.text.includes(`id=${out.id}`));
  assert.ok(Buffer.byteLength(out.text) < text.length);
  assert.equal(store.get(out.id!), text, "the stub is never the only copy");
});

test("byte cap: the omitted count in the marker is the real count", () => {
  const store = new ResultStore();
  const text = "a".repeat(1000);
  const out = capToolResult(text, store, 200);
  const m = /(\d+) of (\d+) bytes omitted/.exec(out.text);
  assert.equal(Number(m?.[2]), 1000);
  assert.equal(Number(m?.[1]), 1000 - 100 - 100);
});

test("byte cap: multibyte text is cut on a character boundary, never mid character", () => {
  const store = new ResultStore();
  const out = capToolResult("é".repeat(600), store, 101);
  assert.ok(!out.text.includes("�"));
});

test("read_result: pages through a stored result and says when it ends", () => {
  const store = new ResultStore();
  const id = store.put("0123456789".repeat(100));
  const first = readResult(store, { id, offset: 0, length: 50 }, 200);
  assert.match(first, /characters 0 to 50 of 1000\. Next offset 50/);
  const last = readResult(store, { id, offset: 950, length: 50 }, 200);
  assert.match(last, /End of result/);
});

test("read_result: an unknown id and an offset past the end are errors, not empty answers", () => {
  const store = new ResultStore();
  const id = store.put("short");
  assert.match(readResult(store, { id: "nope" }, 200), /^error: no stored result/);
  assert.match(readResult(store, { id, offset: 99 }, 200), /^error: offset 99 is past the end/);
});

test("read_result: a slice never exceeds the byte cap", () => {
  const store = new ResultStore();
  const id = store.put("é".repeat(5000));
  const out = readResult(store, { id, length: 999999 }, 400);
  assert.ok(Buffer.byteLength(out.split("\n").slice(1).join("\n")) <= 400);
});

function convo(rounds: number, size: number): Message[] {
  const m: Message[] = [
    { role: "system", content: "SYSTEM PROMPT" },
    { role: "user", content: "FIRST QUESTION" },
  ];
  for (let i = 0; i < rounds; i++) {
    m.push({ role: "assistant", content: "", toolCalls: [{ id: `c${i}`, name: "lookup", arguments: { query: `q${i}` } }] });
    m.push({ role: "tool", content: `result ${i} ${"z".repeat(size)}`, toolName: "lookup" });
  }
  return m;
}

test("fit: under budget changes nothing (positive control)", () => {
  const m = convo(2, 40);
  const out = fitMessages(m, 100_000);
  assert.equal(out.dropped, 0);
  assert.deepEqual(out.messages, m);
});

test("fit: over budget drops the oldest exchanges but keeps the system prompt, the first question and the newest result", () => {
  const out = fitMessages(convo(10, 400), 600);
  assert.ok(out.dropped > 0);
  assert.equal(out.messages[0]?.content, "SYSTEM PROMPT");
  assert.equal(out.messages[1]?.content, "FIRST QUESTION");
  assert.ok(out.messages[2]?.content.startsWith(TRIM_NOTE_PREFIX), "the cut is announced");
  assert.ok(out.messages.at(-1)?.content.startsWith("result 9"));
});

test("fit: an assistant tool call is never separated from its result", () => {
  const out = fitMessages(convo(10, 400), 900);
  const body = out.messages.slice(3);
  for (let i = 0; i < body.length; i += 2) {
    assert.equal(body[i]?.role, "assistant");
    assert.equal(body[i + 1]?.role, "tool");
  }
});

test("fit: trimming twice keeps one note, and its count includes the first trim", () => {
  const first = fitMessages(convo(10, 400), 600);
  const grown = [...first.messages, ...convo(3, 400).slice(2)];
  const second = fitMessages(grown, 600);
  const notes = second.messages.filter((m) => m.content.startsWith(TRIM_NOTE_PREFIX));
  assert.equal(notes.length, 1);
  assert.ok(Number(/trimmed: (\d+)/.exec(notes[0]!.content)?.[1]) > first.dropped);
});

test("fit: a later call under budget keeps the earlier note (a trim is never forgotten)", () => {
  const once = fitMessages(convo(10, 400), 600).messages;
  const again = fitMessages(once, 600);
  assert.equal(again.dropped, 0);
  assert.equal(again.messages.filter((m) => m.content.startsWith(TRIM_NOTE_PREFIX)).length, 1);
});
