import type { Clock } from "./clock.ts";
import { ModelError } from "./model.ts";
import type { ChatOptions, ChatResult, Message, ModelClient, ToolSpec } from "./model.ts";

export type CallFailReason =
  | "time_budget"
  | "model_timeout"
  | "model_unavailable"
  | "model_http_4xx"
  | "model_malformed";

export interface RetryPolicy {
  maxRetries: number;
  baseMs: number;
  capMs: number;
  // Returns a number in [0, 1). Injected so tests can pin it and a simulation can seed it.
  jitter: () => number;
}

export type CallOutcome =
  | { ok: true; result: ChatResult }
  | { ok: false; reason: CallFailReason; detail: string };

// Exponential growth with a cap, then "equal jitter": half fixed, half random. The delay for
// attempt n therefore lies in [exp/2, exp] where exp = min(cap, base * 2^n).
export function backoffDelay(attempt: number, p: RetryPolicy): number {
  const exp = Math.min(p.capMs, p.baseMs * 2 ** attempt);
  return Math.round(exp / 2 + (exp / 2) * p.jitter());
}

export interface CallDeps {
  clock: Clock;
  deadline: number;
  callTimeoutMs: number;
  policy: RetryPolicy;
  emit: (e: { type: "model_call"; attempt: number; timeoutMs: number } | { type: "model_error"; kind: string; status: number | undefined; retryable: boolean } | { type: "backoff"; ms: number }) => void;
}

// A client that ignores options.timeoutMs would hang the run, so the loop enforces it too.
function withTimeout<T>(work: Promise<T>, ms: number): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  const limit = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new ModelError("timeout", `no answer within ${ms} ms`)), ms);
    timer.unref();
  });
  return Promise.race([work, limit]).finally(() => clearTimeout(timer));
}

export async function callWithRetry(
  model: ModelClient,
  messages: Message[],
  tools: ToolSpec[],
  base: Omit<ChatOptions, "timeoutMs">,
  deps: CallDeps,
): Promise<CallOutcome> {
  const { clock, policy } = deps;
  for (let attempt = 0; ; attempt++) {
    const remaining = deps.deadline - clock.now();
    if (remaining <= 0) return { ok: false, reason: "time_budget", detail: "total time budget used up" };
    const timeoutMs = Math.min(deps.callTimeoutMs, remaining);
    deps.emit({ type: "model_call", attempt, timeoutMs });
    try {
      const pending = model.chat(messages, tools, { ...base, timeoutMs });
      pending.catch(() => {}); // when the race below loses, a late failure must not become unhandled
      return { ok: true, result: await withTimeout(pending, timeoutMs) };
    } catch (err) {
      const e = err instanceof ModelError ? err : new ModelError("transport", String(err));
      deps.emit({ type: "model_error", kind: e.kind, status: e.status, retryable: e.retryable });
      if (!e.retryable) {
        const reason = e.kind === "malformed" ? "model_malformed" : "model_http_4xx";
        return { ok: false, reason, detail: e.message };
      }
      if (e.kind === "timeout" && clock.now() >= deps.deadline) {
        return { ok: false, reason: "time_budget", detail: "total time budget used up during a call" };
      }
      if (attempt >= policy.maxRetries) {
        const reason = e.kind === "timeout" ? "model_timeout" : "model_unavailable";
        return { ok: false, reason, detail: `${e.message} (after ${attempt + 1} attempts)` };
      }
      const delay = backoffDelay(attempt, policy);
      if (clock.now() + delay >= deps.deadline) {
        return { ok: false, reason: "time_budget", detail: "no time left for another attempt" };
      }
      deps.emit({ type: "backoff", ms: delay });
      await clock.sleep(delay);
    }
  }
}
