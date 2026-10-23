import type { FakeClock } from "./clock.ts";

export type Role = "system" | "user" | "assistant" | "tool";

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface Message {
  role: Role;
  content: string;
  toolCalls?: ToolCall[];
  toolName?: string;
}

export interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface ChatOptions {
  // Explicit on purpose. undefined means "send nothing and take the server default", which is a
  // setting the eval measures, never an accident.
  temperature?: number | undefined;
  numCtx: number;
  timeoutMs: number;
}

export type FinishReason = "stop" | "length" | "tool_calls" | (string & {});

export interface ChatResult {
  message: string;
  toolCalls: ToolCall[];
  usage: { promptTokens: number; completionTokens: number };
  finishReason: FinishReason;
}

export interface ModelClient {
  chat(messages: Message[], tools: ToolSpec[], options: ChatOptions): Promise<ChatResult>;
}

export type ModelErrorKind = "timeout" | "transport" | "http" | "malformed";

export class ModelError extends Error {
  readonly kind: ModelErrorKind;
  readonly status: number | undefined;
  constructor(kind: ModelErrorKind, message: string, status?: number) {
    super(message);
    this.name = "ModelError";
    this.kind = kind;
    this.status = status;
  }
  // Retry transport trouble, timeouts and 5xx. A 4xx means the request is wrong, so sending it again cannot help.
  get retryable(): boolean {
    if (this.kind === "timeout" || this.kind === "transport") return true;
    return this.kind === "http" && this.status !== undefined && this.status >= 500;
  }
}

interface OllamaToolCall {
  function?: { name?: string; arguments?: Record<string, unknown> };
}

export class OllamaClient implements ModelClient {
  readonly model: string;
  readonly baseUrl: string;
  readonly think: boolean;
  constructor(model: string, opts: { baseUrl?: string; think?: boolean } = {}) {
    this.model = model;
    this.baseUrl = opts.baseUrl ?? "http://127.0.0.1:11434";
    this.think = opts.think ?? false;
  }

  async chat(messages: Message[], tools: ToolSpec[], options: ChatOptions): Promise<ChatResult> {
    const ollamaOptions: Record<string, unknown> = { num_ctx: options.numCtx };
    if (options.temperature !== undefined) ollamaOptions["temperature"] = options.temperature;
    const body: Record<string, unknown> = {
      model: this.model,
      stream: false,
      think: this.think,
      messages: messages.map(toOllamaMessage),
      options: ollamaOptions,
    };
    if (tools.length > 0) {
      body["tools"] = tools.map((t) => ({ type: "function", function: t }));
    }
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/api/chat`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(options.timeoutMs),
      });
    } catch (err) {
      if (err instanceof Error && (err.name === "TimeoutError" || err.name === "AbortError")) {
        throw new ModelError("timeout", `no answer within ${options.timeoutMs} ms`);
      }
      throw new ModelError("transport", err instanceof Error ? err.message : String(err));
    }
    if (!res.ok) {
      throw new ModelError("http", `HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`, res.status);
    }
    let data: {
      message?: { content?: string; tool_calls?: OllamaToolCall[] };
      prompt_eval_count?: number;
      eval_count?: number;
      done_reason?: string;
    };
    try {
      data = (await res.json()) as typeof data;
    } catch {
      throw new ModelError("malformed", "response body was not JSON");
    }
    const toolCalls: ToolCall[] = (data.message?.tool_calls ?? []).flatMap((c, i) =>
      c.function?.name ? [{ id: `call_${i}`, name: c.function.name, arguments: c.function.arguments ?? {} }] : [],
    );
    return {
      message: data.message?.content ?? "",
      toolCalls,
      usage: { promptTokens: data.prompt_eval_count ?? 0, completionTokens: data.eval_count ?? 0 },
      finishReason: toolCalls.length > 0 ? "tool_calls" : (data.done_reason ?? "stop"),
    };
  }
}

function toOllamaMessage(m: Message): Record<string, unknown> {
  if (m.role === "tool") return { role: "tool", content: m.content, tool_name: m.toolName };
  const out: Record<string, unknown> = { role: m.role, content: m.content };
  if (m.toolCalls?.length) {
    out["tool_calls"] = m.toolCalls.map((c) => ({ function: { name: c.name, arguments: c.arguments } }));
  }
  return out;
}

// One scripted reply. Every fault the loop has to survive has a constructor here.
export type ScriptStep =
  | { kind: "reply"; reply: Partial<ChatResult> & { latencyMs?: number } }
  | { kind: "hang" }
  | { kind: "http"; status: number }
  | { kind: "transport" }
  | { kind: "malformed_body" };

export const script = {
  text: (message: string, extra: Partial<ChatResult> = {}): ScriptStep => ({
    kind: "reply",
    reply: { message, finishReason: "stop", ...extra },
  }),
  toolCall: (name: string, args: Record<string, unknown> = {}, id = "c1"): ScriptStep => ({
    kind: "reply",
    reply: { message: "", toolCalls: [{ id, name, arguments: args }], finishReason: "tool_calls" },
  }),
  unknownTool: (name = "does_not_exist"): ScriptStep => script.toolCall(name, {}),
  emptyLength: (): ScriptStep => ({ kind: "reply", reply: { message: "", finishReason: "length" } }),
  // The server answered 200 but only evaluated part of the prompt.
  truncated: (message: string, reportedPromptTokens: number): ScriptStep => ({
    kind: "reply",
    reply: { message, finishReason: "stop", usage: { promptTokens: reportedPromptTokens, completionTokens: 5 } },
  }),
  malformedJson: (): ScriptStep => script.text('{"name": "x", "count": '),
  http: (status: number): ScriptStep => ({ kind: "http", status }),
  hang: (): ScriptStep => ({ kind: "hang" }),
  transport: (): ScriptStep => ({ kind: "transport" }),
  malformedBody: (): ScriptStep => ({ kind: "malformed_body" }),
};

export interface RecordedCall {
  messages: Message[];
  tools: ToolSpec[];
  options: ChatOptions;
}

export interface ScriptedModelOptions {
  clock?: FakeClock;
  // Without this an exhausted script throws, so a test never passes by accident.
  whenExhausted?: ScriptStep;
  // Tokens reported for a reply that does not set its own usage. Defaults to 0, which the loop treats as "not reported".
  reportedPromptTokens?: (messages: Message[], tools: ToolSpec[]) => number;
}

// Replays a list of steps. A "hang" never answers, so only the caller's timeout ends it. When a
// clock is given, latency and timeouts move that clock instead of the wall.
export class ScriptedModel implements ModelClient {
  readonly calls: RecordedCall[] = [];
  private i = 0;
  private readonly steps: ScriptStep[];
  private readonly opts: ScriptedModelOptions;
  constructor(steps: ScriptStep[], opts: ScriptedModelOptions = {}) {
    this.steps = steps;
    this.opts = opts;
  }

  get consumed(): number {
    return this.i;
  }

  async chat(messages: Message[], tools: ToolSpec[], options: ChatOptions): Promise<ChatResult> {
    this.calls.push({ messages: structuredClone(messages), tools, options });
    const step = this.steps[this.i++] ?? this.opts.whenExhausted;
    if (!step) throw new Error(`ScriptedModel script exhausted at call ${this.i}`);
    const clock = this.opts.clock;
    switch (step.kind) {
      case "hang":
        return this.never(options);
      case "http":
        throw new ModelError("http", `HTTP ${step.status}`, step.status);
      case "transport":
        throw new ModelError("transport", "connection reset");
      case "malformed_body":
        throw new ModelError("malformed", "response body was not JSON");
      case "reply": {
        // A reply slower than the caller's timeout never arrives in time, so it behaves like a hang.
        if ((step.reply.latencyMs ?? 0) > options.timeoutMs) return this.never(options);
        clock?.advance(step.reply.latencyMs ?? 0);
        const toolCalls = step.reply.toolCalls ?? [];
        const sent = this.opts.reportedPromptTokens?.(messages, tools) ?? 0;
        return {
          message: step.reply.message ?? "",
          toolCalls,
          usage: step.reply.usage ?? { promptTokens: sent, completionTokens: 10 },
          finishReason: step.reply.finishReason ?? (toolCalls.length > 0 ? "tool_calls" : "stop"),
        };
      }
    }
  }

  // Never answers. The caller's timeout is the only way out. This timer only moves the fake clock to
  // where the caller's timer fires, and is unref'd so it never keeps a process alive.
  private never(options: ChatOptions): Promise<ChatResult> {
    const clock = this.opts.clock;
    if (clock) setTimeout(() => clock.advance(options.timeoutMs), options.timeoutMs).unref();
    return new Promise<ChatResult>(() => {});
  }
}
