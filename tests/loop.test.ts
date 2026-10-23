// LIFESPAN: permanent
import assert from "node:assert/strict";
import { test } from "node:test";
import { backoffDelay } from "../src/retry.ts";
import { script } from "../src/model.ts";
import type { ScriptStep } from "../src/model.ts";
import { standardRegistry } from "../src/tools.ts";
import { failedReason, honestTokens, run, spyRegistry, transitions } from "./helpers.ts";

const slow = (i: number): ScriptStep => ({
  kind: "reply",
  reply: { message: "", toolCalls: [{ id: "c", name: "calculator", arguments: { expression: `${i}+1` } }], latencyMs: 40 },
});
const calc = (expr: string, id = "c") => script.toolCall("calculator", { expression: expr }, id);

test("happy path: tool call then answer, with every transition logged in order", async () => {
  const { result } = await run([calc("17*23"), script.text("It is 391.")]);
  assert.equal(result.state.kind, "done");
  assert.deepEqual(transitions(result), ["running>tool_pending", "tool_pending>running", "running>done"]);
  assert.equal(result.toolTrace[0]?.output, "391");
  assert.equal(result.steps, 2);
});

test("step limit: endless tool calls fail with step_limit (negative)", async () => {
  const steps = [calc("1+1"), calc("2+2"), calc("3+3"), calc("4+4"), calc("5+5")];
  const { result, model } = await run(steps, { maxSteps: 3 });
  assert.equal(failedReason(result), "step_limit");
  assert.equal(model.calls.length, 3);
});

test("step limit: an answer on exactly the last allowed step still succeeds (positive control)", async () => {
  const { result } = await run([calc("1+1"), calc("2+2"), script.text("done")], { maxSteps: 3 });
  assert.equal(result.state.kind, "done");
});

test("repeat guard: the third identical call fails the run, and it ran only twice (negative)", async () => {
  const { registry, executed } = spyRegistry();
  const same = script.toolCall("echo", { x: "a" });
  const { result } = await run([same, same, same, script.text("never")], {}, { registry });
  assert.equal(failedReason(result), "repeated_tool_call");
  assert.deepEqual(executed, ["echo", "echo"]);
});

test("repeat guard: argument order does not hide a repeat (negative)", async () => {
  const { registry } = spyRegistry();
  const a = script.toolCall("echo", { x: "a", y: 1 });
  const b = script.toolCall("echo", { y: 1, x: "a" });
  const { result } = await run([a, b, a, script.text("never")], { maxRepeatedCalls: 2 }, { registry });
  assert.equal(failedReason(result), "repeated_tool_call");
});

test("repeat guard: two identical calls, or the same tool with new arguments, are fine (positive control)", async () => {
  const { registry } = spyRegistry();
  const echo = (x: string) => script.toolCall("echo", { x });
  const { result } = await run([echo("a"), echo("a"), echo("b"), echo("c"), script.text("ok")], {}, { registry });
  assert.equal(result.state.kind, "done");
});

test("timeout: a hang then a good reply recovers through a retry (positive control)", async () => {
  const { result, model } = await run([script.hang(), script.text("fine")]);
  assert.equal(result.state.kind, "done");
  assert.equal(model.calls.length, 2);
});

test("timeout: a model that always hangs ends as model_timeout after the retries (negative)", async () => {
  const { result, model } = await run([script.hang(), script.hang(), script.hang(), script.text("late")], { maxRetries: 2 });
  assert.equal(failedReason(result), "model_timeout");
  assert.equal(model.calls.length, 3);
});

test("timeout: every call is given the per-call limit", async () => {
  const { model } = await run([script.text("fine")], { callTimeoutMs: 25 });
  assert.equal(model.calls[0]?.options.timeoutMs, 25);
});

test("time budget: slow replies use up the total budget and the run fails with time_budget (negative)", async () => {
  const { result } = await run([slow(1), slow(2), slow(3), slow(4)], { callTimeoutMs: 1000, totalTimeoutMs: 100 });
  assert.equal(failedReason(result), "time_budget");
});

test("time budget: the same replies finish when the budget is big enough (positive control)", async () => {
  const { result } = await run([slow(1), slow(2), script.text("ok")], { callTimeoutMs: 1000, totalTimeoutMs: 1_000_000 });
  assert.equal(result.state.kind, "done");
});

test("retry: a 500 is retried once and the run recovers, with the wait logged (positive)", async () => {
  const { result, model, clock } = await run([script.http(500), script.text("ok")]);
  assert.equal(result.state.kind, "done");
  assert.equal(model.calls.length, 2);
  assert.equal(clock.sleeps.length, 1);
});

test("retry: a 503, and a dropped connection, are retried too", async () => {
  for (const fault of [script.http(503), script.transport()]) {
    const { result, model } = await run([fault, script.text("ok")]);
    assert.equal(result.state.kind, "done");
    assert.equal(model.calls.length, 2);
  }
});

test("retry: a 400 is never retried (negative)", async () => {
  for (const status of [400, 401, 404, 429]) {
    const { result, model } = await run([script.http(status), script.text("never")]);
    assert.equal(failedReason(result), "model_http_4xx", `status ${status}`);
    assert.equal(model.calls.length, 1, `status ${status} must not be retried`);
  }
});

test("retry: a malformed response body is not retried", async () => {
  const { result, model } = await run([script.malformedBody(), script.text("never")]);
  assert.equal(failedReason(result), "model_malformed");
  assert.equal(model.calls.length, 1);
});

test("retry: a server that keeps failing ends as model_unavailable after maxRetries + 1 calls", async () => {
  const { result, model } = await run([script.http(500), script.http(500), script.http(500), script.text("never")], { maxRetries: 2 });
  assert.equal(failedReason(result), "model_unavailable");
  assert.equal(model.calls.length, 3);
});

test("backoff: waits grow exponentially, stay inside [exp/2, exp], and respect the cap", () => {
  const policy = (j: number) => ({ maxRetries: 5, baseMs: 100, capMs: 700, jitter: () => j });
  assert.deepEqual([0, 1, 2, 3, 4].map((n) => backoffDelay(n, policy(0))), [50, 100, 200, 350, 350]);
  assert.deepEqual([0, 1, 2, 3, 4].map((n) => backoffDelay(n, policy(0.999))), [100, 200, 400, 700, 700]);
  for (let n = 0; n < 6; n++) {
    for (const j of [0, 0.25, 0.5, 0.75, 0.999]) {
      const d = backoffDelay(n, policy(j));
      const exp = Math.min(700, 100 * 2 ** n);
      assert.ok(d >= exp / 2 - 1 && d <= exp, `attempt ${n} jitter ${j} gave ${d}`);
    }
  }
});

test("backoff: the loop waits what backoffDelay says, in order", async () => {
  const { clock } = await run([script.http(500), script.http(500), script.text("ok")], {}, { jitter: () => 0 });
  assert.deepEqual(clock.sleeps, [50, 100]);
});

test("truncation: a server reading half the prompt is reported in the events and the warnings (negative)", async () => {
  const long = "word ".repeat(600);
  const { result } = await run([script.truncated("answer", 120)], {}, { prompt: long });
  assert.equal(result.warnings.length, 1);
  assert.ok(result.events.some((e) => e.type === "truncation_suspected"));
  assert.equal(result.state.kind, "done");
});

test("truncation: with onTruncation=fail the run fails closed with prompt_truncated", async () => {
  const long = "word ".repeat(600);
  const { result } = await run([script.truncated("answer", 120)], { onTruncation: "fail" }, { prompt: long });
  assert.equal(failedReason(result), "prompt_truncated");
});

test("truncation: an honest token count raises nothing (positive control)", async () => {
  const long = "word ".repeat(600);
  const { result } = await run([script.text("answer")], {}, { prompt: long, model: { reportedPromptTokens: honestTokens } });
  assert.deepEqual(result.warnings, []);
  assert.ok(!result.events.some((e) => e.type === "truncation_suspected"));
});

test("truncation: a server that reports no count (0) is not accused", async () => {
  const long = "word ".repeat(600);
  const { result } = await run([script.truncated("answer", 0)], {}, { prompt: long });
  assert.deepEqual(result.warnings, []);
});

test("empty reply with finish_reason length: one retry with the same messages and fewer tools", async () => {
  const { result, model } = await run([script.emptyLength(), script.text("ok")], {}, { registry: standardRegistry(24), prompt: "Calculate 3 plus 4 with the calculator" });
  assert.equal(result.state.kind, "done");
  const [first, second] = model.calls;
  assert.equal(first?.tools.length, 24);
  assert.ok((second?.tools.length ?? 99) <= 3, "the retry offers at most narrowTo tools");
  assert.equal(second?.tools[0]?.name, "calculator");
  assert.deepEqual(second?.messages, first?.messages, "no longer prompt, no nudge");
});

test("empty reply with finish_reason length: a second empty reply fails closed (negative)", async () => {
  const { result, model } = await run([script.emptyLength(), script.emptyLength(), script.text("never")]);
  assert.equal(failedReason(result), "empty_reply");
  assert.equal(model.calls.length, 2);
});

test("empty reply with finish_reason stop is not retried (negative)", async () => {
  const { result, model } = await run([script.text("", { finishReason: "stop" }), script.text("never")]);
  assert.equal(failedReason(result), "empty_reply");
  assert.equal(model.calls.length, 1);
});

test("unknown tool: never executed, the model is told, and it can recover (negative)", async () => {
  const { registry, executed } = spyRegistry();
  const { result, model } = await run([script.unknownTool("delete_everything"), script.text("ok")], {}, { registry });
  assert.deepEqual(executed, []);
  assert.equal(result.state.kind, "done");
  assert.ok(result.events.some((e) => e.type === "tool_rejected" && e.why === "unknown_or_not_offered"));
  const seen = model.calls[1]?.messages.at(-1)?.content ?? "";
  assert.match(seen, /no tool named "delete_everything"/);
});

test("unknown tool: a real, offered tool does execute (positive control)", async () => {
  const { registry, executed } = spyRegistry();
  await run([script.toolCall("echo", { x: "a" }), script.text("ok")], {}, { registry });
  assert.deepEqual(executed, ["echo"]);
});

test("a registered tool that is not offered this turn is rejected like an unknown one", async () => {
  const { registry, executed } = spyRegistry();
  const { result } = await run([script.toolCall("boom", {}), script.text("ok")], { toolsPerTurn: 1 }, { registry, prompt: "echo this" });
  assert.deepEqual(executed, []);
  assert.ok(result.events.some((e) => e.type === "tool_rejected"));
});

test("a tool that throws becomes an error message, not a crashed run", async () => {
  const { registry } = spyRegistry();
  const { result } = await run([script.toolCall("boom", {}), script.text("ok")], {}, { registry });
  assert.equal(result.state.kind, "done");
  assert.ok(result.events.some((e) => e.type === "tool_rejected" && e.why === "tool_threw"));
});

test("bad arguments are rejected by the tool schema and never run", async () => {
  const { registry, executed } = spyRegistry();
  const { result } = await run([script.toolCall("echo", { x: 5 }), script.text("ok")], {}, { registry });
  assert.deepEqual(executed, []);
  assert.ok(result.events.some((e) => e.type === "tool_rejected" && e.why === "bad_arguments"));
});

test("narrowing: toolsPerTurn offers the best 5 of 24, and the real tool is among them", async () => {
  const { model } = await run([script.text("ok")], { toolsPerTurn: 5 }, { registry: standardRegistry(24), prompt: "Use the calculator for 3 + 4" });
  assert.equal(model.calls[0]?.tools.length, 5);
  assert.ok(model.calls[0]?.tools.some((t) => t.name === "calculator"));
});

test("narrowing: without it all 24 tools are offered (positive control)", async () => {
  const { model } = await run([script.text("ok")], {}, { registry: standardRegistry(24) });
  assert.equal(model.calls[0]?.tools.length, 24);
});

test("temperature is explicit: unset sends nothing, 0 sends 0", async () => {
  const unset = await run([script.text("ok")], { temperature: undefined });
  const zero = await run([script.text("ok")], { temperature: 0 });
  assert.equal(unset.model.calls[0]?.options.temperature, undefined);
  assert.equal(zero.model.calls[0]?.options.temperature, 0);
});

test("byte cap end to end: a big lookup is stubbed, read_result is offered and returns the rest", async () => {
  const steps = [
    script.toolCall("lookup", { query: "big report sensor reading" }, "a"),
    script.toolCall("read_result", { id: "r1", offset: 400 }, "b"),
    script.text("ok"),
  ];
  const { result, model } = await run(steps, { toolCapBytes: 500, toolsPerTurn: 1 }, { prompt: "look up the big report" });
  assert.equal(result.state.kind, "done");
  assert.match(model.calls[1]?.messages.at(-1)?.content ?? "", /\[TRUNCATED/);
  assert.ok(model.calls[1]?.tools.some((t) => t.name === "read_result"), "read_result appears once a stub exists");
  assert.match(model.calls[2]?.messages.at(-1)?.content ?? "", /characters 400 to/);
  assert.ok((result.store.get("r1")?.length ?? 0) > 2000, "the full text is still held");
});
