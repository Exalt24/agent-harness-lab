// LIFESPAN: permanent
import assert from "node:assert/strict";
import { test } from "node:test";
import { FakeClock } from "../src/clock.ts";
import { runAgent } from "../src/loop.ts";
import type { RunResult } from "../src/loop.ts";
import { script, ScriptedModel } from "../src/model.ts";
import { standardRegistry } from "../src/tools.ts";
import { CASES } from "../eval/cases.ts";
import { CONFIGS, classify, pooled, runEval, scriptedSteps, summarize } from "../eval/engine.ts";
import type { TrialRecord } from "../eval/engine.ts";
import { formatRate, overlaps, wilson } from "../eval/stats.ts";

const near = (a: number | undefined, b: number, tol = 0.0005) => assert.ok(a !== undefined && Math.abs(a - b) < tol, `${a} is not within ${tol} of ${b}`);

test("wilson: reference values for 0/10, 5/10, 10/10 and 1/4", () => {
  const zero = wilson(0, 10);
  near(zero?.low, 0);
  near(zero?.high, 0.2775);
  const half = wilson(5, 10);
  near(half?.low, 0.2366);
  near(half?.high, 0.7634);
  const all = wilson(10, 10);
  near(all?.low, 0.7225);
  near(all?.high, 1);
  const q = wilson(1, 4);
  near(q?.low, 0.0455);
  near(q?.high, 0.6994);
});

test("wilson: no scored trials gives no interval, a bad count throws, and the interval always holds the rate", () => {
  assert.equal(wilson(0, 0), null);
  assert.throws(() => wilson(5, 4), RangeError);
  for (let n = 1; n <= 30; n++) {
    for (let k = 0; k <= n; k++) {
      const ci = wilson(k, n)!;
      assert.ok(ci.low >= 0 && ci.high <= 1 && ci.low <= k / n + 1e-12 && k / n <= ci.high + 1e-12, `${k}/${n}`);
    }
  }
});

test("a rate is never printed without its interval, and zero scored trials prints no rate", () => {
  assert.match(formatRate(7, 10, 0), /7\/10 = 70% \(95% CI \d+% to \d+%\)/);
  assert.match(formatRate(7, 10, 2), /2 ERROR/);
  assert.equal(formatRate(0, 0, 4), "no scored trials, 4 ERROR");
  assert.ok(!/%/.test(formatRate(0, 0, 4)));
});

test("overlaps: wide intervals overlap, far apart ones do not, a missing one is not comparable", () => {
  assert.equal(overlaps(wilson(5, 10), wilson(6, 10)), true);
  assert.equal(overlaps(wilson(0, 40), wilson(40, 40)), false);
  assert.equal(overlaps(wilson(0, 0), wilson(1, 2)), null);
});

async function runCase(steps: ReturnType<typeof scriptedSteps>, caseIndex = 0): Promise<RunResult> {
  const clock = new FakeClock();
  return runAgent({ model: new ScriptedModel(steps, { clock }), registry: standardRegistry(5), prompt: CASES[caseIndex]!.prompt, config: { callTimeoutMs: 20, backoffBaseMs: 5, maxRetries: 1, maxSteps: 6 }, clock, jitter: () => 0 });
}

test("classify: a correct run PASSES, a wrong answer and a skipped tool FAIL", async () => {
  const c = CASES[0]!;
  assert.equal(classify(c, await runCase(c.good)).outcome, "pass");
  assert.deepEqual(classify(c, await runCase([script.toolCall("calculator", { expression: "17*23+4" }), script.text("It is 400.")])), { outcome: "fail", reason: "answer does not contain 395" });
  assert.deepEqual(classify(c, await runCase([script.text("395")])), { outcome: "fail", reason: "did not call calculator" });
});

test("classify: 8400 is not 840 and 1395 is not 395", async () => {
  const two = CASES[2]!;
  const wrong = await runCase([script.toolCall("lookup", { query: "standard plan requests per day" }), script.toolCall("calculator", { expression: "120*70" }), script.text("8400")], 2);
  assert.equal(classify(two, wrong).outcome, "fail");
  const one = CASES[0]!;
  const r = await runCase([script.toolCall("calculator", { expression: "1" }), script.text("1395")]);
  assert.equal(classify(one, r).outcome, "fail");
});

test("classify: a server failure, a timeout and an empty answer are ERROR, never a zero score", async () => {
  const c = CASES[0]!;
  assert.equal(classify(c, await runCase([script.http(500), script.http(500)])).outcome, "error");
  assert.equal(classify(c, await runCase([script.http(400)])).outcome, "error");
  assert.equal(classify(c, await runCase([script.hang(), script.hang()])).outcome, "error");
  assert.equal(classify(c, await runCase([script.text("", { finishReason: "stop" })])).outcome, "error");
});

test("classify: a model that loops on one call is a FAIL, because that is the model's behaviour", async () => {
  const same = script.toolCall("calculator", { expression: "1+1" });
  assert.deepEqual(classify(CASES[0]!, await runCase([same, same, same])), { outcome: "fail", reason: "repeated_tool_call" });
});

test("summarize: errors are counted apart and stay out of the denominator", () => {
  const rec = (outcome: TrialRecord["outcome"], trial: number): TrialRecord => ({ case: "a", config: "x", trial, outcome, reason: "r", steps: 1, ms: 1, tools: [], truncationWarnings: 0, answer: "" });
  const cell = summarize([rec("pass", 0), rec("pass", 1), rec("fail", 2), rec("error", 3), rec("error", 4)])[0]!;
  assert.deepEqual([cell.pass, cell.fail, cell.error, cell.scored], [2, 1, 2, 3]);
  assert.equal(cell.rate, 2 / 3);
  const allErrors = summarize([rec("error", 0), rec("error", 1)])[0]!;
  assert.equal(allErrors.rate, null, "no scored trial means no rate, never 0%");
  assert.equal(allErrors.ci, null);
});

test("scripted eval: the same run twice gives byte-identical trials (deterministic)", async () => {
  const a = await runEval({ backend: "scripted", trials: 10, configs: CONFIGS });
  const b = await runEval({ backend: "scripted", trials: 10, configs: CONFIGS });
  const strip = (t: TrialRecord[]) => JSON.stringify(t.map((x) => ({ ...x, ms: 0 })));
  assert.equal(strip(a.trials), strip(b.trials));
  assert.equal(a.trials.length, CONFIGS.length * CASES.length * 10);
  assert.equal(a.complete, true);
});

test("scripted eval: it exercises PASS, FAIL and ERROR so the scoring paths are all covered", async () => {
  const { trials } = await runEval({ backend: "scripted", trials: 10, configs: CONFIGS });
  for (const outcome of ["pass", "fail", "error"] as const) assert.ok(trials.some((t) => t.outcome === outcome), `no ${outcome} trial`);
  const p = pooled(trials, CONFIGS[0]!.id);
  assert.equal(p.pass + p.fail + p.error, CASES.length * 10);
});

test("resume: finished trials are not run again, and a time budget stops cleanly with complete=false", async () => {
  const first = await runEval({ backend: "scripted", trials: 4, configs: [CONFIGS[0]!], budgetMs: 0 });
  assert.equal(first.complete, false);
  assert.equal(first.trials.length, 0, "the budget is checked before each trial");
  const full = await runEval({ backend: "scripted", trials: 4, configs: [CONFIGS[0]!] });
  const resumed = await runEval({ backend: "scripted", trials: 4, configs: [CONFIGS[0]!], existing: full.trials.slice(0, 5) });
  assert.equal(resumed.trials.length, full.trials.length);
  assert.equal(new Set(resumed.trials.map((t) => `${t.case}|${t.config}|${t.trial}`)).size, full.trials.length, "no trial appears twice");
});

test("pooled by group: the tool-choice pool leaves the reading case out, and both pools account for every trial", async () => {
  const { trials } = await runEval({ backend: "scripted", trials: 10, configs: [CONFIGS[0]!] });
  const all = pooled(trials, CONFIGS[0]!.id);
  const choice = pooled(trials, CONFIGS[0]!.id, "tool-choice");
  const reading = pooled(trials, CONFIGS[0]!.id, "reading");
  assert.equal(choice.pass + choice.fail + choice.error, 30);
  assert.equal(reading.pass + reading.fail + reading.error, 10);
  assert.equal(all.pass, choice.pass + reading.pass);
});
