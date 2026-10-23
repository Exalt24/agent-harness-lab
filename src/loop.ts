import { realClock } from "./clock.ts";
import type { Clock } from "./clock.ts";
import { capToolResult, estimatePromptTokens, estimateTokens, fitMessages, ResultStore } from "./context.ts";
import type { Message, ModelClient, ToolCall } from "./model.ts";
import { callWithRetry } from "./retry.ts";
import type { CallFailReason } from "./retry.ts";
import type { ToolRegistry } from "./tools.ts";

export type FailReason = CallFailReason | "step_limit" | "repeated_tool_call" | "empty_reply" | "prompt_truncated";

// The whole run is one of these four. A transition is only legal along the table below, and every
// one is written to the event log.
export type RunState =
  | { kind: "running" }
  | { kind: "tool_pending"; calls: ToolCall[] }
  | { kind: "done"; answer: string }
  | { kind: "failed"; reason: FailReason; detail: string };

export type StateKind = RunState["kind"];

export const LEGAL_TRANSITIONS: Record<StateKind, StateKind[]> = {
  running: ["tool_pending", "done", "failed"],
  tool_pending: ["running", "failed"],
  done: [],
  failed: [],
};

export type LoopEvent = { seq: number; at: number } & (
  | { type: "run_start"; offered: number }
  | { type: "transition"; from: StateKind; to: StateKind; reason?: FailReason }
  | { type: "model_call"; attempt: number; timeoutMs: number }
  | { type: "model_error"; kind: string; status: number | undefined; retryable: boolean }
  | { type: "backoff"; ms: number }
  | { type: "model_reply"; step: number; promptTokens: number; finishReason: string; toolCalls: number }
  | { type: "truncation_suspected"; estimated: number; reported: number }
  | { type: "context_trimmed"; dropped: number }
  | { type: "tool_requested"; name: string }
  | { type: "tool_rejected"; name: string; why: "unknown_or_not_offered" | "bad_arguments" | "tool_threw" }
  | { type: "tool_executed"; name: string; stubbed: boolean }
  | { type: "empty_retry_narrowed"; offered: string[] }
);

type EventBody = LoopEvent extends infer E ? (E extends unknown ? Omit<E, "seq" | "at"> : never) : never;

export interface LoopConfig {
  systemPrompt: string;
  temperature: number | undefined;
  numCtx: number;
  maxSteps: number;
  maxRepeatedCalls: number;
  totalTimeoutMs: number;
  callTimeoutMs: number;
  maxRetries: number;
  backoffBaseMs: number;
  backoffCapMs: number;
  toolCapBytes: number;
  ctxReserveTokens: number;
  // Suspect truncation when the server reports fewer prompt tokens than this share of our estimate.
  truncationRatio: number;
  truncationMinTokens: number;
  onTruncation: "warn" | "fail";
  // Offer only the best-ranked N tools each turn. undefined offers every registered tool.
  toolsPerTurn: number | undefined;
  narrowTo: number;
}

export const DEFAULT_CONFIG: LoopConfig = {
  systemPrompt: "You are a careful assistant. Use the tools when they help and answer briefly.",
  temperature: undefined,
  numCtx: 8192,
  maxSteps: 8,
  maxRepeatedCalls: 2,
  totalTimeoutMs: 120_000,
  callTimeoutMs: 60_000,
  maxRetries: 2,
  backoffBaseMs: 250,
  backoffCapMs: 4000,
  toolCapBytes: 2000,
  ctxReserveTokens: 1024,
  truncationRatio: 0.6,
  truncationMinTokens: 100,
  onTruncation: "warn",
  toolsPerTurn: undefined,
  narrowTo: 3,
};

export interface ToolTraceEntry {
  name: string;
  args: Record<string, unknown>;
  output: string;
}

export interface RunResult {
  state: Extract<RunState, { kind: "done" | "failed" }>;
  events: LoopEvent[];
  steps: number;
  warnings: string[];
  usage: { promptTokens: number; completionTokens: number };
  toolTrace: ToolTraceEntry[];
  messages: Message[];
  store: ResultStore;
}

export interface RunInput {
  model: ModelClient;
  registry: ToolRegistry;
  prompt: string;
  config?: Partial<LoopConfig>;
  clock?: Clock;
  jitter?: () => number;
}

// Same call, same key, whatever order the model wrote the arguments in.
function stable(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(stable).join(",")}]`;
  if (v && typeof v === "object") {
    const o = v as Record<string, unknown>;
    return `{${Object.keys(o).sort().map((k) => `${k}:${stable(o[k])}`).join(",")}}`;
  }
  return String(v);
}

export async function runAgent(input: RunInput): Promise<RunResult> {
  const cfg: LoopConfig = { ...DEFAULT_CONFIG, ...input.config };
  const clock = input.clock ?? realClock;
  const { model, registry, prompt } = input;
  const store = new ResultStore();
  const events: LoopEvent[] = [];
  const warnings: string[] = [];
  const toolTrace: ToolTraceEntry[] = [];
  const usage = { promptTokens: 0, completionTokens: 0 };
  const seenCalls = new Map<string, number>();
  const deadline = clock.now() + cfg.totalTimeoutMs;
  let messages: Message[] = [
    { role: "system", content: cfg.systemPrompt },
    { role: "user", content: prompt },
  ];
  let state: RunState = { kind: "running" } as RunState;
  let steps = 0;
  let narrowed: string[] | undefined;

  const emit = (e: EventBody) => void events.push({ seq: events.length, at: clock.now(), ...e } as LoopEvent);
  const go = (next: RunState) => {
    if (!LEGAL_TRANSITIONS[state.kind].includes(next.kind)) throw new Error(`illegal transition ${state.kind} -> ${next.kind}`);
    emit({ type: "transition", from: state.kind, to: next.kind, ...(next.kind === "failed" ? { reason: next.reason } : {}) });
    state = next;
  };
  const kindNow = (): StateKind => state.kind;
  const fail = (reason: FailReason, detail: string) => go({ kind: "failed", reason, detail });

  const offeredNow = (): string[] => {
    let names = narrowed ?? registry.names();
    if (cfg.toolsPerTurn !== undefined && !narrowed) names = registry.rank(prompt).slice(0, cfg.toolsPerTurn);
    // A stub is only useful if the model can read it back, so read_result is offered once one exists.
    if (store.size > 0 && registry.get("read_result") && !names.includes("read_result")) names = [...names, "read_result"];
    return names;
  };

  emit({ type: "run_start", offered: offeredNow().length });

  while (state.kind === "running") {
    if (steps >= cfg.maxSteps) {
      fail("step_limit", `no final answer within ${cfg.maxSteps} steps`);
      break;
    }
    if (clock.now() >= deadline) {
      fail("time_budget", "total time budget used up");
      break;
    }
    const offered = offeredNow();
    const specs = registry.specs(offered);
    const toolTokens = estimateTokens(JSON.stringify(specs));
    const fit = fitMessages(messages, cfg.numCtx - cfg.ctxReserveTokens, toolTokens);
    if (fit.dropped > 0) emit({ type: "context_trimmed", dropped: fit.dropped });
    messages = fit.messages;

    const call = await callWithRetry(model, messages, specs, { temperature: cfg.temperature, numCtx: cfg.numCtx }, {
      clock,
      deadline,
      callTimeoutMs: cfg.callTimeoutMs,
      policy: {
        maxRetries: cfg.maxRetries,
        baseMs: cfg.backoffBaseMs,
        capMs: cfg.backoffCapMs,
        jitter: input.jitter ?? Math.random,
      },
      emit,
    });
    if (!call.ok) {
      fail(call.reason, call.detail);
      break;
    }
    steps++;
    const reply = call.result;
    usage.promptTokens += reply.usage.promptTokens;
    usage.completionTokens += reply.usage.completionTokens;
    emit({ type: "model_reply", step: steps, promptTokens: reply.usage.promptTokens, finishReason: reply.finishReason, toolCalls: reply.toolCalls.length });

    // Lesson 1: a server can answer 200 with half the prompt gone. Compare what it says it read to what we sent.
    const estimated = estimatePromptTokens(messages, specs);
    const reported = reply.usage.promptTokens;
    if (reported > 0 && estimated >= cfg.truncationMinTokens && reported < cfg.truncationRatio * estimated) {
      emit({ type: "truncation_suspected", estimated, reported });
      warnings.push(`step ${steps}: server read ${reported} prompt tokens but about ${estimated} were sent, the prompt may have been truncated`);
      if (cfg.onTruncation === "fail") {
        fail("prompt_truncated", warnings[warnings.length - 1]!);
        break;
      }
    }

    if (reply.toolCalls.length > 0) {
      messages.push({ role: "assistant", content: reply.message, toolCalls: reply.toolCalls });
      go({ kind: "tool_pending", calls: reply.toolCalls });
      for (const tc of reply.toolCalls) {
        emit({ type: "tool_requested", name: tc.name });
        const key = `${tc.name}:${stable(tc.arguments)}`;
        const seen = (seenCalls.get(key) ?? 0) + 1;
        seenCalls.set(key, seen);
        if (seen > cfg.maxRepeatedCalls) {
          fail("repeated_tool_call", `${tc.name} called ${seen} times with the same arguments`);
          break;
        }
        const tool = registry.get(tc.name);
        let output: string;
        if (!tool || !offered.includes(tc.name)) {
          emit({ type: "tool_rejected", name: tc.name, why: "unknown_or_not_offered" });
          output = `error: there is no tool named "${tc.name}". Available tools: ${offered.join(", ")}`;
        } else {
          const parsed = tool.schema.safeParse(tc.arguments);
          if (!parsed.success) {
            emit({ type: "tool_rejected", name: tc.name, why: "bad_arguments" });
            output = `error: bad arguments for ${tc.name}: ${parsed.error.issues.map((i) => `${i.path.join(".")} ${i.message}`).join("; ")}`;
          } else {
            try {
              const raw = tool.run(parsed.data, { store, capBytes: cfg.toolCapBytes });
              const capped = capToolResult(raw, store, cfg.toolCapBytes);
              emit({ type: "tool_executed", name: tc.name, stubbed: capped.stubbed });
              toolTrace.push({ name: tc.name, args: parsed.data, output: raw });
              output = capped.text;
            } catch (e) {
              // A throwing tool is the tool's fault, not the run's: tell the model and keep going.
              emit({ type: "tool_rejected", name: tc.name, why: "tool_threw" });
              output = `error: ${tc.name} failed: ${(e as Error).message}`;
            }
          }
        }
        messages.push({ role: "tool", content: output, toolName: tc.name });
      }
      if (kindNow() === "tool_pending") go({ kind: "running" });
      continue;
    }

    const text = reply.message.trim();
    if (text.length > 0) {
      go({ kind: "done", answer: text });
    } else if (reply.finishReason === "length" && !narrowed) {
      // Lesson 5: do not nudge with a longer prompt. Same messages, fewer tools, once.
      const keep = Math.max(1, Math.min(cfg.narrowTo, offered.length - 1));
      narrowed = registry.rank(prompt).filter((n) => offered.includes(n)).slice(0, keep);
      emit({ type: "empty_retry_narrowed", offered: narrowed });
    } else {
      fail("empty_reply", `empty reply with finish reason ${reply.finishReason}`);
    }
  }

  if (state.kind !== "done" && state.kind !== "failed") throw new Error("loop exited in a non-terminal state");
  return { state, events, steps, warnings, usage, toolTrace, messages, store };
}
