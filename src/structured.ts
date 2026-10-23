import { z } from "zod";
import { realClock } from "./clock.ts";
import type { Clock } from "./clock.ts";
import { DEFAULT_CONFIG } from "./loop.ts";
import type { LoopConfig } from "./loop.ts";
import type { Message, ModelClient } from "./model.ts";
import { callWithRetry } from "./retry.ts";
import type { CallFailReason } from "./retry.ts";

export type StructuredResult<T> =
  | { ok: true; data: T; attempts: number }
  | { ok: false; reason: "invalid_json" | "schema_violation" | CallFailReason; detail: string; attempts: number };

type Checked<T> = { ok: true; data: T } | { ok: false; error: string; kind: "invalid_json" | "schema_violation" };

// Models often wrap an answer in a code fence. That is the only leniency: nothing is repaired.
function parseJson(raw: string): { ok: true; value: unknown } | { ok: false; error: string } {
  const text = raw.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  try {
    return { ok: true, value: JSON.parse(text) };
  } catch (e) {
    // COPY-OK: this text goes back to the model, not to a screen.
    return { ok: false, error: `is not valid JSON (${(e as Error).message})` };
  }
}

function check<T>(raw: string, schema: z.ZodType<T>): Checked<T> {
  const json = parseJson(raw);
  if (!json.ok) return { ok: false, error: json.error, kind: "invalid_json" };
  const parsed = schema.safeParse(json.value);
  if (!parsed.success) {
    const issues = parsed.error.issues.map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`).join("; ");
    return { ok: false, error: `does not match the schema: ${issues}`, kind: "schema_violation" };
  }
  return { ok: true, data: parsed.data };
}

export interface StructuredInput<T> {
  model: ModelClient;
  prompt: string;
  schema: z.ZodType<T>;
  config?: Partial<LoopConfig>;
  clock?: Clock;
  jitter?: () => number;
}

// Lesson 7. The model's answer is checked against the schema. A failure is sent back once with the
// exact validation error. A second failure ends the run with a reason and no data: this function
// never returns anything the schema did not accept.
export async function structuredOutput<T>(input: StructuredInput<T>): Promise<StructuredResult<T>> {
  const cfg: LoopConfig = { ...DEFAULT_CONFIG, ...input.config };
  const clock = input.clock ?? realClock;
  const deadline = clock.now() + cfg.totalTimeoutMs;
  const messages: Message[] = [
    // COPY-OK: this text goes to the model, not to a screen.
    { role: "system", content: `Reply with only a JSON object that matches this schema: ${JSON.stringify(z.toJSONSchema(input.schema))}` },
    { role: "user", content: input.prompt },
  ];
  let lastError = "";
  let lastKind: "invalid_json" | "schema_violation" = "invalid_json";
  for (let attempt = 1; attempt <= 2; attempt++) {
    const call = await callWithRetry(input.model, messages, [], { temperature: cfg.temperature, numCtx: cfg.numCtx }, {
      clock,
      deadline,
      callTimeoutMs: cfg.callTimeoutMs,
      policy: { maxRetries: cfg.maxRetries, baseMs: cfg.backoffBaseMs, capMs: cfg.backoffCapMs, jitter: input.jitter ?? Math.random },
      emit: () => {},
    });
    if (!call.ok) return { ok: false, reason: call.reason, detail: call.detail, attempts: attempt };
    const raw = call.result.message;
    const checked = check(raw, input.schema);
    if (checked.ok) return { ok: true, data: checked.data, attempts: attempt };
    lastError = checked.error;
    lastKind = checked.kind;
    messages.push({ role: "assistant", content: raw });
    // COPY-OK: this text goes to the model, not to a screen.
    messages.push({ role: "user", content: `Your reply was rejected: it ${checked.error}. Reply again with only the corrected answer.` });
  }
  return { ok: false, reason: lastKind, detail: lastError, attempts: 2 };
}
