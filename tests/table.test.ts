import assert from "node:assert/strict";
import { test } from "node:test";
import { LEGAL_TRANSITIONS } from "../src/loop.ts";
import type { RunResult, StateKind } from "../src/loop.ts";
import { script } from "../src/model.ts";
import type { ScriptStep } from "../src/model.ts";
import { run, spyRegistry } from "./helpers.ts";

// Every fault type crossed with where in the run it lands (first call, after one tool round, after
// two) and with the step limit either far away or exactly one call away. The expectation for each
// pair is written by hand in `expect`, so the table reads as the loop's specification.

type Outcome = { kind: "done"; warnings?: number } | { kind: "failed"; reason: string };

interface Fault {
  name: string;
  step: ScriptStep;
  // A fault that needs one more model call to recover cannot recover when the limit is the fault call itself.
  needsAnotherStep: boolean;
  expect: Outcome;
}

const faults: Fault[] = [
  { name: "http 500 (retried)", step: script.http(500), needsAnotherStep: false, expect: { kind: "done" } },
  { name: "dropped connection (retried)", step: script.transport(), needsAnotherStep: false, expect: { kind: "done" } },
  { name: "hang past the timeout (retried)", step: script.hang(), needsAnotherStep: false, expect: { kind: "done" } },
  { name: "http 400", step: script.http(400), needsAnotherStep: false, expect: { kind: "failed", reason: "model_http_4xx" } },
  { name: "malformed body", step: script.malformedBody(), needsAnotherStep: false, expect: { kind: "failed", reason: "model_malformed" } },
  { name: "empty reply with finish length", step: script.emptyLength(), needsAnotherStep: true, expect: { kind: "done" } },
  { name: "empty reply with finish stop", step: script.text("", { finishReason: "stop" }), needsAnotherStep: false, expect: { kind: "failed", reason: "empty_reply" } },
  { name: "unknown tool", step: script.unknownTool("ghost"), needsAnotherStep: true, expect: { kind: "done" } },
  { name: "tool that throws", step: script.toolCall("boom", {}), needsAnotherStep: true, expect: { kind: "done" } },
  { name: "bad tool arguments", step: script.toolCall("echo", { x: 5 }), needsAnotherStep: true, expect: { kind: "done" } },
  { name: "truncated prompt report", step: script.truncated("recovered", 3), needsAnotherStep: false, expect: { kind: "done", warnings: 1 } },
];

function expected(f: Fault, atLimit: boolean): Outcome {
  if (atLimit && f.needsAnotherStep) return { kind: "failed", reason: "step_limit" };
  return f.expect;
}

function assertWellFormed(r: RunResult) {
  const t = r.events.flatMap((e) => (e.type === "transition" ? [e] : []));
  assert.equal(t[0]?.from, "running", "the first transition starts at running");
  let prev: StateKind = "running";
  for (const e of t) {
    assert.equal(e.from, prev, "each transition starts where the last one ended");
    assert.ok(LEGAL_TRANSITIONS[e.from].includes(e.to), `${e.from} -> ${e.to} is legal`);
    prev = e.to;
  }
  assert.ok(prev === "done" || prev === "failed", "the last transition lands in a terminal state");
  assert.equal(t.filter((e) => e.to === "done" || e.to === "failed").length, 1, "exactly one terminal transition");
  assert.deepEqual(r.events.map((e) => e.seq), r.events.map((_, i) => i), "the event log is gapless");
}

for (const fault of faults) {
  for (const position of [0, 1, 2]) {
    for (const atLimit of [false, true]) {
      test(`table: ${fault.name} | after ${position} tool rounds | ${atLimit ? "at the step limit" : "limit far away"}`, async () => {
        const { registry, executed } = spyRegistry();
        const before = Array.from({ length: position }, (_, i) => script.toolCall("echo", { x: `r${i}` }, `c${i}`));
        const steps = [...before, fault.step, script.text("recovered")];
        const { result } = await run(steps, { maxSteps: atLimit ? position + 1 : 10 }, { registry, prompt: "word ".repeat(300) });
        assertWellFormed(result);
        const want = expected(fault, atLimit);
        assert.equal(result.state.kind, want.kind);
        if (want.kind === "failed" && result.state.kind === "failed") assert.equal(result.state.reason, want.reason);
        if (want.kind === "done") assert.equal(result.warnings.length, want.warnings ?? 0);
        // Only registered tools may ever have run.
        assert.ok(executed.every((n) => n === "echo" || n === "boom"), `only registered tools may execute, saw ${executed.join(",")}`);
        assert.ok(executed.length >= Math.min(position, result.steps), "the earlier rounds ran");
      });
    }
  }
}
