import { z } from "zod";
import { READ_RESULT_SPEC, readResult } from "./context.ts";
import type { ResultStore } from "./context.ts";
import type { ToolSpec } from "./model.ts";

export interface ToolContext {
  store: ResultStore;
  capBytes: number;
}

export interface Tool {
  spec: ToolSpec;
  schema: z.ZodType<Record<string, unknown>>;
  run(args: Record<string, unknown>, ctx: ToolContext): string;
}

// Arithmetic by a small recursive descent parser. There is no eval and no Function: the input is
// only ever read as numbers, + - * / % ^, unary minus and parentheses.
export function calculate(expr: string): number {
  const src = expr.replace(/\s+/g, "");
  if (src.length === 0 || src.length > 200) throw new Error("expression is empty or too long");
  let pos = 0;
  const peek = () => src[pos];
  function number_(): number {
    const m = /^\d+(\.\d+)?/.exec(src.slice(pos));
    if (!m) throw new Error(`unexpected "${peek() ?? "end"}" at ${pos}`);
    pos += m[0].length;
    return Number(m[0]);
  }
  function atom(): number {
    if (peek() === "(") {
      pos++;
      const v = sum();
      if (peek() !== ")") throw new Error("missing )");
      pos++;
      return v;
    }
    if (peek() === "-") {
      pos++;
      return -atom();
    }
    return number_();
  }
  function power(): number {
    const base = atom();
    if (peek() === "^") {
      pos++;
      return base ** power();
    }
    return base;
  }
  function product(): number {
    let v = power();
    while (peek() === "*" || peek() === "/" || peek() === "%") {
      const op = src[pos++];
      const r = power();
      if (op === "*") v *= r;
      else if (r === 0) throw new Error("division by zero");
      else v = op === "/" ? v / r : v % r;
    }
    return v;
  }
  function sum(): number {
    let v = product();
    while (peek() === "+" || peek() === "-") {
      const op = src[pos++];
      const r = product();
      v = op === "+" ? v + r : v - r;
    }
    return v;
  }
  const result = sum();
  if (pos !== src.length) throw new Error(`unexpected "${peek()}" at ${pos}`);
  if (!Number.isFinite(result)) throw new Error("result is not a finite number");
  return result;
}

const calculator: Tool = {
  spec: {
    name: "calculator",
    description: "Evaluate an arithmetic expression such as (12 + 5) * 3. Supports + - * / % ^ and parentheses.",
    parameters: {
      type: "object",
      properties: { expression: { type: "string", description: "The arithmetic expression" } },
      required: ["expression"],
    },
  },
  schema: z.object({ expression: z.string() }),
  run: (args) => {
    try {
      return String(calculate(String(args["expression"])));
    } catch (e) {
      return `error: ${(e as Error).message}`;
    }
  },
};

export const DOCS: Record<string, string> = {
  "relay-service": "The relay service listens on port 8472 and keeps lead records for 30 days.",
  "standard-plan": "The standard plan allows 120 requests per day and 3 seats.",
  "pro-plan": "The pro plan allows 900 requests per day and 25 seats.",
  "retention-policy": "Audit logs are kept for 400 days. Backups run nightly at 02:00 UTC.",
  "big-report": Array.from({ length: 400 }, (_, i) => `Line ${i + 1}: sensor reading ${(i * 37) % 101}.`).join("\n"),
};

function words(s: string): Set<string> {
  return new Set(s.toLowerCase().match(/[a-z0-9]+/g) ?? []);
}

function overlap(a: Set<string>, b: Set<string>): number {
  let n = 0;
  for (const w of a) if (b.has(w)) n++;
  return n;
}

const lookup: Tool = {
  spec: {
    name: "lookup",
    description: "Search the small internal document set and return the best matching document.",
    parameters: {
      type: "object",
      properties: { query: { type: "string", description: "Keywords to search for" } },
      required: ["query"],
    },
  },
  schema: z.object({ query: z.string() }),
  run: (args) => {
    const q = words(String(args["query"]));
    let best: [string, number] = ["", 0];
    for (const [name, text] of Object.entries(DOCS)) {
      const score = overlap(q, words(`${name.replace(/-/g, " ")} ${text}`));
      if (score > best[1]) best = [name, score];
    }
    return best[1] === 0 ? "no matching document" : `${best[0]}: ${DOCS[best[0]]}`;
  },
};

// Decoys exist only to make the offered list longer. They are real entries in the list the model
// sees and they answer honestly that they do nothing, so a model that picks one gets no help.
const DECOY_NAMES = [
  ["get_weather", "Get the current weather for a city"],
  ["translate_text", "Translate text into another language"],
  ["send_email", "Send an email message to a contact"],
  ["create_calendar_event", "Create an event on the calendar"],
  ["search_web", "Search the public web for pages"],
  ["convert_currency", "Convert an amount between currencies"],
  ["get_stock_price", "Get the latest price for a stock ticker"],
  ["set_reminder", "Set a reminder for later"],
  ["read_file", "Read a file from disk"],
  ["write_file", "Write text to a file on disk"],
  ["list_directory", "List the files in a directory"],
  ["run_shell", "Run a shell command"],
  ["get_time", "Get the current time in a time zone"],
  ["summarize_text", "Summarize a long text"],
  ["generate_image", "Generate an image from a prompt"],
  ["get_news", "Get recent news headlines"],
  ["book_flight", "Book a flight between two airports"],
  ["check_inventory", "Check stock levels for a product"],
  ["create_ticket", "Open a support ticket"],
  ["query_database", "Run a read-only database query"],
  ["post_message", "Post a message to a chat channel"],
] as const;

export function decoyTools(count: number): Tool[] {
  return DECOY_NAMES.slice(0, count).map(([name, description]) => ({
    spec: {
      name,
      description,
      parameters: { type: "object", properties: { input: { type: "string" } }, required: ["input"] },
    },
    schema: z.object({ input: z.string() }),
    run: () => `${name} is not available in this lab`,
  }));
}

const readResultTool: Tool = {
  spec: READ_RESULT_SPEC,
  schema: z.object({ id: z.string(), offset: z.number().optional(), length: z.number().optional() }),
  run: (args, ctx) => readResult(ctx.store, args, ctx.capBytes),
};

export class ToolRegistry {
  private readonly tools = new Map<string, Tool>();
  constructor(tools: Tool[]) {
    for (const t of tools) this.tools.set(t.spec.name, t);
  }
  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }
  names(): string[] {
    return [...this.tools.keys()];
  }
  specs(names?: string[]): ToolSpec[] {
    const pick = names ?? this.names();
    return pick.flatMap((n) => {
      const t = this.tools.get(n);
      return t ? [t.spec] : [];
    });
  }
  // The names most related to a prompt, best first. Word overlap with the name and description.
  // Ties keep registration order, so the real tools (registered first) win a tie with a decoy.
  rank(prompt: string): string[] {
    const q = words(prompt);
    const scored = this.names().map((n, i) => {
      const t = this.tools.get(n)!;
      return { n, i, s: overlap(q, words(`${n.replace(/_/g, " ")} ${t.spec.description}`)) };
    });
    return scored.sort((a, b) => b.s - a.s || a.i - b.i).map((x) => x.n);
  }
}

// The registry the eval uses: the two real tools, read_result, and `total - 3` decoys.
export function standardRegistry(total = 3): ToolRegistry {
  return new ToolRegistry([calculator, lookup, readResultTool, ...decoyTools(Math.max(0, total - 3))]);
}
