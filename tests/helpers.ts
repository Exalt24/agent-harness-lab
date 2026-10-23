import { z } from "zod";
import { FakeClock } from "../src/clock.ts";
import { estimatePromptTokens } from "../src/context.ts";
import { runAgent } from "../src/loop.ts";
import type { LoopConfig, RunResult } from "../src/loop.ts";
import { ScriptedModel } from "../src/model.ts";
import type { ScriptStep, ScriptedModelOptions } from "../src/model.ts";
import { standardRegistry, ToolRegistry } from "../src/tools.ts";
import type { Tool } from "../src/tools.ts";

export interface Harness {
  result: RunResult;
  model: ScriptedModel;
  clock: FakeClock;
}

// Fast defaults: a short call timeout so a scripted hang costs milliseconds, and a fake clock so
// backoff and the time budget cost nothing.
export const FAST: Partial<LoopConfig> = { callTimeoutMs: 25, backoffBaseMs: 100, backoffCapMs: 1000, totalTimeoutMs: 60_000 };

export async function run(
  steps: ScriptStep[],
  config: Partial<LoopConfig> = {},
  opts: { registry?: ToolRegistry; prompt?: string; jitter?: () => number; model?: Partial<ScriptedModelOptions> } = {},
): Promise<Harness> {
  const clock = new FakeClock();
  const model = new ScriptedModel(steps, { clock, ...opts.model });
  const result = await runAgent({
    model,
    registry: opts.registry ?? standardRegistry(5),
    prompt: opts.prompt ?? "What is 17 * 23?",
    config: { ...FAST, ...config },
    clock,
    jitter: opts.jitter ?? (() => 0),
  });
  return { result, model, clock };
}

export const honestTokens = (m: Parameters<NonNullable<ScriptedModelOptions["reportedPromptTokens"]>>[0], t: Parameters<NonNullable<ScriptedModelOptions["reportedPromptTokens"]>>[1]) =>
  estimatePromptTokens(m, t);

export function transitions(r: RunResult): string[] {
  return r.events.flatMap((e) => (e.type === "transition" ? [`${e.from}>${e.to}`] : []));
}

export function failedReason(r: RunResult): string | undefined {
  return r.state.kind === "failed" ? r.state.reason : undefined;
}

// A registry whose tools count their own executions, so a test can prove a call did or did not run.
export function spyRegistry(): { registry: ToolRegistry; executed: string[]; throwing: Tool } {
  const executed: string[] = [];
  const mk = (name: string, fn: (a: Record<string, unknown>) => string): Tool => ({
    spec: { name, description: `${name} tool`, parameters: { type: "object", properties: { x: { type: "string" } } } },
    schema: z.object({ x: z.string().optional() }),
    run: (a) => {
      executed.push(name);
      return fn(a);
    },
  });
  const throwing = mk("boom", () => {
    throw new Error("kaboom");
  });
  const registry = new ToolRegistry([mk("echo", (a) => `echo:${String(a["x"])}`), throwing]);
  return { registry, executed, throwing };
}

// A small seeded generator (mulberry32) so a simulation failure replays from its seed.
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
