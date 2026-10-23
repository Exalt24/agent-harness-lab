import type { Message, ToolSpec } from "./model.ts";

// An ESTIMATE, not a tokenizer: ceil(chars / 4). Real tokenizers differ by model and by content
// (JSON and code usually cost more tokens per char). The loop only uses it to notice a gross
// mismatch with what the server reports and to decide when to trim, never as an exact count.
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export function estimatePromptTokens(messages: Message[], tools: ToolSpec[]): number {
  let n = 0;
  for (const m of messages) {
    n += estimateTokens(m.content) + 4;
    for (const c of m.toolCalls ?? []) n += estimateTokens(c.name + JSON.stringify(c.arguments));
  }
  for (const t of tools) n += estimateTokens(JSON.stringify(t));
  return n;
}

// Full text of every capped tool result. A stub in the context is never the only copy.
export class ResultStore {
  private readonly items = new Map<string, string>();
  put(text: string): string {
    const id = `r${this.items.size + 1}`;
    this.items.set(id, text);
    return id;
  }
  get(id: string): string | undefined {
    return this.items.get(id);
  }
  get size(): number {
    return this.items.size;
  }
}

export const STUB_MARKER = "[TRUNCATED";

function headBytes(text: string, n: number): string {
  return Buffer.from(text, "utf8").subarray(0, n).toString("utf8").replace(/�+$/, "");
}

function tailBytes(text: string, n: number): string {
  return Buffer.from(text, "utf8").subarray(-n).toString("utf8").replace(/^�+/, "");
}

export interface CapResult {
  text: string;
  stubbed: boolean;
  id?: string;
}

// A result at or under the byte cap passes through untouched. A bigger one becomes head + visible
// marker + tail, and the full text goes to the store under an id the model can read back.
export function capToolResult(text: string, store: ResultStore, capBytes: number): CapResult {
  const total = Buffer.byteLength(text, "utf8");
  if (total <= capBytes) return { text, stubbed: false };
  const id = store.put(text);
  const keep = Math.max(1, Math.floor(capBytes / 2));
  const head = headBytes(text, keep);
  const tail = tailBytes(text, keep);
  const omitted = total - Buffer.byteLength(head, "utf8") - Buffer.byteLength(tail, "utf8");
  // COPY-OK: this text is read by the model, not shown in an app screen.
  const marker = `${STUB_MARKER}: ${omitted} of ${total} bytes omitted. Full result id=${id}. Use read_result to read it.]`;
  return { text: `${head}\n${marker}\n${tail}`, stubbed: true, id };
}

export const READ_RESULT_SPEC: ToolSpec = {
  name: "read_result",
  description: "Read part of a tool result that was cut short. Offsets count characters.",
  parameters: {
    type: "object",
    properties: {
      id: { type: "string", description: "The result id from the TRUNCATED marker" },
      offset: { type: "number", description: "First character to read, default 0" },
      length: { type: "number", description: "How many characters to read" },
    },
    required: ["id"],
  },
};

export function readResult(store: ResultStore, args: Record<string, unknown>, capBytes: number): string {
  const id = typeof args["id"] === "string" ? args["id"] : "";
  const full = store.get(id);
  if (full === undefined) return `error: no stored result with id "${id}"`;
  const offset = Number.isInteger(args["offset"]) ? Math.max(0, args["offset"] as number) : 0;
  const asked = Number.isInteger(args["length"]) ? (args["length"] as number) : capBytes;
  // A slice is capped by characters at half the byte cap, so even 2-byte text stays under the cap.
  const length = Math.max(1, Math.min(asked, Math.floor(capBytes / 2)));
  if (offset >= full.length) return `error: offset ${offset} is past the end (${full.length} characters)`;
  const slice = full.slice(offset, offset + length);
  const end = offset + slice.length;
  const more = end < full.length ? ` Next offset ${end}.` : " End of result.";
  return `[characters ${offset} to ${end} of ${full.length}.${more}]\n${slice}`;
}

export interface FitResult {
  messages: Message[];
  dropped: number;
}

export const TRIM_NOTE_PREFIX = "[context trimmed:";

// Keeps the system prompt and the first user message always. Drops the oldest tool exchanges whole
// (an assistant tool call together with its results, so the pairing never breaks) until the
// estimate fits. The newest exchange is never dropped, so the model always sees its last result.
export function fitMessages(messages: Message[], budgetTokens: number, toolTokens = 0): FitResult {
  const clean = messages.filter((m) => !m.content.startsWith(TRIM_NOTE_PREFIX));
  const head = clean.slice(0, 2);
  const units: Message[][] = [];
  for (let i = 2; i < clean.length; i++) {
    const m = clean[i]!;
    const unit = [m];
    if (m.role === "assistant" && m.toolCalls?.length) {
      while (clean[i + 1]?.role === "tool") unit.push(clean[++i]!);
    }
    units.push(unit);
  }
  const cost = (u: Message[][]) => estimatePromptTokens([...head, ...u.flat()], []) + toolTokens;
  // Under budget: hand back the messages as they are, an earlier trim note included.
  if (cost(units) <= budgetTokens) return { messages, dropped: 0 };
  const oldNote = messages.find((m) => m.content.startsWith(TRIM_NOTE_PREFIX))?.content ?? "";
  const earlier = Number(/trimmed: (\d+) earlier/.exec(oldNote)?.[1] ?? 0);
  let dropped = 0;
  while (units.length > 1 && cost(units) + 20 > budgetTokens) {
    dropped += units.shift()!.length;
  }
  const note: Message = {
    role: "user",
    // COPY-OK: this text is read by the model, not shown in an app screen.
    content: `${TRIM_NOTE_PREFIX} ${earlier + dropped} earlier messages were dropped to fit the window. Stored results stay readable with read_result.]`,
  };
  return { messages: [...head, note, ...units.flat()], dropped };
}
