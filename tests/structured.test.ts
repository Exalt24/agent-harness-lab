import assert from "node:assert/strict";
import { test } from "node:test";
import { z } from "zod";
import { FakeClock } from "../src/clock.ts";
import { script, ScriptedModel } from "../src/model.ts";
import type { ScriptStep } from "../src/model.ts";
import { structuredOutput } from "../src/structured.ts";

const Schema = z.object({ name: z.string(), count: z.number().int().min(0) });
const FAST = { callTimeoutMs: 25, backoffBaseMs: 10, totalTimeoutMs: 60_000 };

async function ask(steps: ScriptStep[]) {
  const model = new ScriptedModel(steps, { clock: new FakeClock() });
  const result = await structuredOutput({ model, prompt: "describe it", schema: Schema, config: FAST, clock: new FakeClock(), jitter: () => 0 });
  return { result, model };
}

test("valid answer on the first try returns the parsed data (positive control)", async () => {
  const { result, model } = await ask([script.text('{"name":"a","count":3}')]);
  assert.deepEqual(result, { ok: true, data: { name: "a", count: 3 }, attempts: 1 });
  assert.equal(model.calls.length, 1);
});

test("a fenced answer is accepted, nothing else is repaired", async () => {
  const { result } = await ask([script.text('```json\n{"name":"a","count":3}\n```')]);
  assert.equal(result.ok, true);
});

test("schema violation: re-asked once with the exact validation error, then succeeds", async () => {
  const { result, model } = await ask([script.text('{"name":"a","count":-1}'), script.text('{"name":"a","count":1}')]);
  assert.equal(result.ok, true);
  assert.equal(model.calls.length, 2);
  const reask = model.calls[1]?.messages.at(-1)?.content ?? "";
  assert.match(reask, /count/, "the error names the failing field");
  assert.match(reask, /rejected/);
});

test("malformed JSON: re-asked once, then succeeds", async () => {
  const { result, model } = await ask([script.malformedJson(), script.text('{"name":"a","count":1}')]);
  assert.equal(result.ok, true);
  assert.match(model.calls[1]?.messages.at(-1)?.content ?? "", /not valid JSON/);
});

test("two bad answers fail closed with a reason and NO data (negative)", async () => {
  const { result, model } = await ask([script.text('{"name":"a","count":"many"}'), script.text('{"name":"a","count":"lots"}'), script.text('{"name":"a","count":1}')]);
  assert.equal(result.ok, false);
  assert.ok(!("data" in result), "an unvalidated answer is never returned");
  if (!result.ok) {
    assert.equal(result.reason, "schema_violation");
    assert.equal(result.attempts, 2);
  }
  assert.equal(model.calls.length, 2, "exactly one re-ask, never more");
});

test("two answers that are not JSON fail with invalid_json", async () => {
  const { result } = await ask([script.text("sure!"), script.text("here you go")]);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.reason, "invalid_json");
});

test("valid JSON of the wrong shape (an array, null, a string) is rejected", async () => {
  for (const bad of ["[]", "null", '"text"', "42"]) {
    const { result } = await ask([script.text(bad), script.text(bad)]);
    assert.equal(result.ok, false, bad);
  }
});

test("a server failure is a reason, not data: 500 is retried, 400 is not", async () => {
  const retried = await ask([script.http(500), script.text('{"name":"a","count":1}')]);
  assert.equal(retried.result.ok, true);
  const refused = await ask([script.http(400), script.text('{"name":"a","count":1}')]);
  assert.equal(refused.result.ok, false);
  if (!refused.result.ok) assert.equal(refused.result.reason, "model_http_4xx");
  assert.equal(refused.model.calls.length, 1);
});
