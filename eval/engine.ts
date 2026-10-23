import { FakeClock } from "../src/clock.ts";
import { DEFAULT_CONFIG, runAgent } from "../src/loop.ts";
import type { LoopConfig, RunResult } from "../src/loop.ts";
import { script, ScriptedModel } from "../src/model.ts";
import type { ModelClient, ScriptStep } from "../src/model.ts";
import { standardRegistry } from "../src/tools.ts";
import { CASES } from "./cases.ts";
import type { EvalCase } from "./cases.ts";
import { wilson } from "./stats.ts";
import type { Interval } from "./stats.ts";

// One configuration is one setting of the two things the eval compares.
export interface EvalConfig {
  id: string;
  temperature: number | undefined;
  toolCount: number;
}

export const CONFIGS: EvalConfig[] = [
  { id: "temp-unset-tools5", temperature: undefined, toolCount: 5 },
  { id: "temp0-tools5", temperature: 0, toolCount: 5 },
  { id: "temp-unset-tools24", temperature: undefined, toolCount: 24 },
  { id: "temp0-tools24", temperature: 0, toolCount: 24 },
];

// PASS and FAIL are scored. ERROR means the trial measured nothing about the model (the server
// broke, the run timed out, the reply was empty), so it is counted apart and never as a zero.
export type Outcome = "pass" | "fail" | "error";

export interface TrialRecord {
  case: string;
  config: string;
  trial: number;
  outcome: Outcome;
  reason: string;
  steps: number;
  ms: number;
  tools: string[];
  truncationWarnings: number;
  // The first 160 characters of the final answer, kept so a score can be audited by reading it.
  answer: string;
}

// Failures that are the model's behaviour count as FAIL. Every other way a run can end is ERROR.
const MODEL_BEHAVIOUR = new Set(["step_limit", "repeated_tool_call"]);

export function classify(c: EvalCase, r: RunResult): { outcome: Outcome; reason: string } {
  if (r.state.kind === "failed") {
    return MODEL_BEHAVIOUR.has(r.state.reason)
      ? { outcome: "fail", reason: r.state.reason }
      : { outcome: "error", reason: `${r.state.reason}: ${r.state.detail}` };
  }
  if (r.state.answer.trim() === "") return { outcome: "error", reason: "empty answer" };
  const v = c.check(r);
  return v.pass ? { outcome: "pass", reason: "ok" } : { outcome: "fail", reason: v.reason };
}

export interface Cell {
  case: string;
  config: string;
  pass: number;
  fail: number;
  error: number;
  scored: number;
  rate: number | null;
  ci: Interval | null;
  errorReasons: string[];
  truncationWarnings: number;
}

export function summarize(trials: TrialRecord[], caseFilter?: string): Cell[] {
  const keys = [...new Set(trials.map((t) => `${t.case}|${t.config}`))];
  return keys.flatMap((k) => {
    const [cs, cf] = k.split("|") as [string, string];
    if (caseFilter && cs !== caseFilter) return [];
    const own = trials.filter((t) => t.case === cs && t.config === cf);
    const pass = own.filter((t) => t.outcome === "pass").length;
    const fail = own.filter((t) => t.outcome === "fail").length;
    const error = own.filter((t) => t.outcome === "error").length;
    const scored = pass + fail;
    return [{
      case: cs, config: cf, pass, fail, error, scored,
      rate: scored > 0 ? pass / scored : null,
      ci: wilson(pass, scored),
      errorReasons: [...new Set(own.filter((t) => t.outcome === "error").map((t) => t.reason))],
      truncationWarnings: own.reduce((n, t) => n + t.truncationWarnings, 0),
    }];
  });
}

// Pools the cases of one configuration into a single cell. With `group` only that group's cases count.
export function pooled(trials: TrialRecord[], configId: string, group?: EvalCase["group"]): Cell {
  const ids = new Set(CASES.filter((c) => !group || c.group === group).map((c) => c.id));
  const own = trials.filter((t) => t.config === configId && ids.has(t.case));
  const pass = own.filter((t) => t.outcome === "pass").length;
  const fail = own.filter((t) => t.outcome === "fail").length;
  const scored = pass + fail;
  return {
    case: group ?? "all", config: configId, pass, fail, scored, error: own.length - scored,
    rate: scored > 0 ? pass / scored : null, ci: wilson(pass, scored),
    errorReasons: [...new Set(own.filter((t) => t.outcome === "error").map((t) => t.reason))],
    truncationWarnings: own.reduce((n, t) => n + t.truncationWarnings, 0),
  };
}

// A small deterministic hash so a scripted trial always replays the same way.
function seedOf(s: string): number {
  let h = 2166136261;
  for (const ch of s) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return h >>> 0;
}

function unit(seed: number): number {
  let t = (seed + 0x6d2b79f5) >>> 0;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

// The scripted backend checks the pipeline, not a model. Each trial draws one scenario from its own
// seed: mostly a correct run, and the rest the faults a scoring pipeline has to classify correctly.
export function scriptedSteps(c: EvalCase, config: string, trial: number): ScriptStep[] {
  const x = unit(seedOf(`${c.id}/${config}/${trial}`));
  if (x < 0.6) return c.good;
  if (x < 0.7) return [script.http(500), ...c.good];
  if (x < 0.8) return [script.text("It is probably 42.")];
  if (x < 0.88) return [script.text("I will not use a tool for that.")];
  if (x < 0.92) return [script.http(500), script.http(500), script.http(500)];
  if (x < 0.96) return [script.text("", { finishReason: "stop" })];
  return [script.toolCall("calculator", { expression: "1+1" }), script.toolCall("calculator", { expression: "1+1" }), script.toolCall("calculator", { expression: "1+1" })];
}

export interface EvalOptions {
  backend: "scripted" | "ollama";
  model?: ModelClient;
  trials: number;
  configs: EvalConfig[];
  cases?: EvalCase[];
  existing?: TrialRecord[];
  // Stop cleanly between trials once this many ms have passed, so a long run can be chunked.
  budgetMs?: number;
  onTrial?: (t: TrialRecord, all: TrialRecord[]) => void;
}

export async function runEval(o: EvalOptions): Promise<{ trials: TrialRecord[]; complete: boolean }> {
  const trials: TrialRecord[] = [...(o.existing ?? [])];
  const started = Date.now();
  const done = new Set(trials.map((t) => `${t.case}|${t.config}|${t.trial}`));
  for (const config of o.configs) {
    for (const c of o.cases ?? CASES) {
      for (let i = 0; i < o.trials; i++) {
        if (done.has(`${c.id}|${config.id}|${i}`)) continue;
        if (o.budgetMs !== undefined && Date.now() - started >= o.budgetMs) return { trials, complete: false };
        const cfg: Partial<LoopConfig> = { temperature: config.temperature, maxSteps: 6, maxRetries: 1 };
        let model: ModelClient;
        let clock: FakeClock | undefined;
        if (o.backend === "scripted") {
          clock = new FakeClock();
          model = new ScriptedModel(scriptedSteps(c, config.id, i), { clock });
          Object.assign(cfg, { callTimeoutMs: 20, backoffBaseMs: 10 });
        } else {
          model = o.model!;
          Object.assign(cfg, { callTimeoutMs: 120_000, totalTimeoutMs: 300_000, numCtx: DEFAULT_CONFIG.numCtx });
        }
        const t0 = Date.now();
        const result = await runAgent({ model, registry: standardRegistry(config.toolCount), prompt: c.prompt, config: cfg, ...(clock ? { clock, jitter: () => 0 } : {}) });
        const v = classify(c, result);
        const rec: TrialRecord = {
          case: c.id, config: config.id, trial: i, outcome: v.outcome, reason: v.reason,
          steps: result.steps, ms: Date.now() - t0, tools: result.toolTrace.map((t) => t.name),
          truncationWarnings: result.warnings.length,
          answer: result.state.kind === "done" ? result.state.answer.slice(0, 160) : "",
        };
        trials.push(rec);
        o.onTrial?.(rec, trials);
      }
    }
  }
  return { trials, complete: true };
}
