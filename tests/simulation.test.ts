// LIFESPAN: permanent
import assert from "node:assert/strict";
import { test } from "node:test";
import { z } from "zod";
import { FakeClock } from "../src/clock.ts";
import { STUB_MARKER } from "../src/context.ts";
import { LEGAL_TRANSITIONS, runAgent } from "../src/loop.ts";
import type { LoopConfig, RunResult } from "../src/loop.ts";
import { script, ScriptedModel } from "../src/model.ts";
import type { ScriptStep } from "../src/model.ts";
import { structuredOutput } from "../src/structured.ts";
import { standardRegistry } from "../src/tools.ts";
import { rng } from "./helpers.ts";

const SEEDS = 20;
const STEPS = 50;

// One random script of 50 model steps. Every fault the loop knows about can land anywhere, and
// calls repeat, point at tools that do not exist, and ask for results that may not exist.
function randomScript(r: () => number): ScriptStep[] {
  const pick = <T,>(xs: T[]): T => xs[Math.floor(r() * xs.length)]!;
  let n = 0;
  return Array.from({ length: STEPS }, (): ScriptStep => {
    const x = r();
    const latencyMs = Math.floor(r() * 3000);
    const reply = (s: ScriptStep): ScriptStep => (s.kind === "reply" ? { kind: "reply", reply: { ...s.reply, latencyMs } } : s);
    if (x < 0.03) return reply(script.text("final answer"));
    if (x < 0.3) return reply(script.toolCall("calculator", { expression: `${++n}+${n}` }, `c${n}`));
    if (x < 0.38) return reply(script.toolCall("calculator", { expression: "1+1" }));
    if (x < 0.46) return reply(script.unknownTool(pick(["ghost_a", "ghost_b", "rm_rf"])));
    if (x < 0.56) return reply(script.toolCall("lookup", { query: "big report sensor reading" }));
    if (x < 0.64) return reply(script.toolCall("read_result", { id: pick(["r1", "r2", "r9"]), offset: Math.floor(r() * 3000) }));
    if (x < 0.72) return script.http(pick([500, 502, 503]));
    if (x < 0.74) return script.http(pick([400, 404, 429]));
    if (x < 0.8) return script.transport();
    if (x < 0.82) return script.hang();
    if (x < 0.83) return script.malformedBody();
    if (x < 0.87) return script.emptyLength();
    if (x < 0.88) return script.text("", { finishReason: "stop" });
    if (x < 0.93) return reply(script.truncated("partial", 2));
    return reply(script.toolCall("calculator", { expression: pick(["process.exit()", "((", "1/0", "7*6"]) }));
  });
}

const CFG: Partial<LoopConfig> = {
  maxSteps: 20,
  maxRepeatedCalls: 2,
  callTimeoutMs: 20,
  totalTimeoutMs: 40_000,
  backoffBaseMs: 200,
  backoffCapMs: 3000,
  toolCapBytes: 500,
  onTruncation: "warn",
};

function invariants(seed: number, r: RunResult, clock: FakeClock, model: ScriptedModel) {
  const at = `seed ${seed}`;
  assert.ok(r.state.kind === "done" || r.state.kind === "failed", `${at}: ends terminal`);
  if (r.state.kind === "failed") assert.ok(r.state.reason.length > 0 && r.state.detail.length > 0, `${at}: a failure carries a reason`);
  if (r.state.kind === "done") assert.ok(r.state.answer.length > 0, `${at}: an answer is never empty`);
  assert.ok(r.steps <= CFG.maxSteps!, `${at}: steps ${r.steps} within the limit`);
  assert.ok(clock.now() <= CFG.totalTimeoutMs!, `${at}: clock ${clock.now()} within the time budget`);
  let prev = "running";
  for (const e of r.events) {
    if (e.type !== "transition") continue;
    assert.equal(e.from, prev, `${at}: transitions chain`);
    assert.ok(LEGAL_TRANSITIONS[e.from].includes(e.to), `${at}: legal transition`);
    prev = e.to;
  }
  assert.ok(prev === "done" || prev === "failed", `${at}: last transition is terminal`);
  const real = new Set(standardRegistry(5).names());
  assert.ok(r.toolTrace.every((t) => real.has(t.name)), `${at}: an unknown tool call never executes`);
  const repeats = new Map<string, number>();
  for (const t of r.toolTrace) repeats.set(`${t.name}${JSON.stringify(t.args)}`, (repeats.get(`${t.name}${JSON.stringify(t.args)}`) ?? 0) + 1);
  assert.ok([...repeats.values()].every((c) => c <= CFG.maxRepeatedCalls!), `${at}: no call ran more than the repeat limit`);
  // A stub is never the only copy: every marker in what the model saw names a stored result that is bigger than the stub.
  for (const call of model.calls) {
    for (const m of call.messages) {
      if (m.role !== "tool" || !m.content.includes(STUB_MARKER)) continue;
      const id = /id=(r\d+)/.exec(m.content)?.[1];
      assert.ok(id !== undefined, `${at}: the marker names a result id`);
      assert.ok((r.store.get(id) ?? "").length > m.content.length, `${at}: the full text behind ${id} is held`);
    }
  }
  const stubbed = r.events.filter((e) => e.type === "tool_executed" && e.stubbed).length;
  assert.equal(r.store.size, stubbed, `${at}: one stored copy per stub`);
  assert.ok(model.calls.length <= (CFG.maxSteps! + 1) * (3 + 1), `${at}: model calls stay bounded`);
}

const outcomes = new Map<string, number>();

for (let seed = 1; seed <= SEEDS; seed++) {
  test(`simulation: loop, seed ${seed}, ${STEPS} scripted steps with random faults`, async () => {
    const r = rng(seed);
    const clock = new FakeClock();
    const model = new ScriptedModel(randomScript(r), { clock, whenExhausted: script.text("exhausted") });
    const result = await runAgent({ model, registry: standardRegistry(5), prompt: "word ".repeat(100), config: CFG, clock, jitter: r });
    invariants(seed, result, clock, model);
    const key = result.state.kind === "failed" ? `failed:${result.state.reason}` : "done";
    outcomes.set(key, (outcomes.get(key) ?? 0) + 1);
  });
}

test("simulation: the seeds are not vacuous, they reach several different endings", () => {
  assert.ok(outcomes.size >= 3, `only saw ${[...outcomes.keys()].join(", ")}`);
});

const Shape = z.object({ label: z.string().min(1), n: z.number().int() });

function replyOptions(r: () => number): { raw: string; valid: boolean } {
  const x = r();
  if (x < 0.35) return { raw: JSON.stringify({ label: "a", n: Math.floor(r() * 9) }), valid: true };
  if (x < 0.45) return { raw: "```json\n" + JSON.stringify({ label: "b", n: 2 }) + "\n```", valid: true };
  if (x < 0.6) return { raw: JSON.stringify({ label: "", n: 1 }), valid: false };
  if (x < 0.75) return { raw: JSON.stringify({ label: "a", n: 1.5 }), valid: false };
  if (x < 0.9) return { raw: '{"label": "a", "n": ', valid: false };
  return { raw: "Sure, here it is.", valid: false };
}

for (let seed = 1; seed <= SEEDS; seed++) {
  test(`simulation: structured output, seed ${seed}, ${STEPS} requests, never returns unvalidated data`, async () => {
    const r = rng(1000 + seed);
    let accepted = 0;
    for (let i = 0; i < STEPS; i++) {
      const picks = [replyOptions(r), replyOptions(r)];
      const steps: ScriptStep[] = [];
      if (r() < 0.2) steps.push(script.http(500));
      steps.push(script.text(picks[0]!.raw), script.text(picks[1]!.raw));
      const clock = new FakeClock();
      const model = new ScriptedModel(steps, { clock });
      const out = await structuredOutput({ model, prompt: "go", schema: Shape, config: { callTimeoutMs: 20, backoffBaseMs: 5 }, clock, jitter: () => 0 });
      const oracle = picks[0]!.valid || picks[1]!.valid;
      assert.equal(out.ok, oracle, `seed ${seed} request ${i}: ok must equal "one of the first two replies is valid"`);
      if (out.ok) {
        accepted++;
        assert.ok(Shape.safeParse(out.data).success, "returned data passes the schema");
        assert.equal(out.attempts, picks[0]!.valid ? 1 : 2);
      } else {
        assert.ok(!("data" in out), "a failure carries no data");
        assert.ok(model.calls.length <= 3, "at most one re-ask");
      }
    }
    assert.ok(accepted > 0 && accepted < STEPS, "the seed mixes accepted and refused answers");
  });
}
