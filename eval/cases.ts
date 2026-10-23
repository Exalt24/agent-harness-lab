import type { RunResult } from "../src/loop.ts";
import { script } from "../src/model.ts";
import type { ScriptStep } from "../src/model.ts";

export type Verdict = { pass: true } | { pass: false; reason: string };

export interface EvalCase {
  id: string;
  // tool-choice: did the model pick and call the right tools. reading: did it read a cut result correctly.
  group: "tool-choice" | "reading";
  prompt: string;
  // The tools a correct run must have called. A right number with no tool call is a FAIL, because
  // dropped tool calls are the thing this eval is built to measure.
  mustCall: string[];
  check(result: RunResult): Verdict;
  // What the scripted backend replays for a correct run of this case.
  good: ScriptStep[];
}

function answerOf(r: RunResult): string {
  return r.state.kind === "done" ? r.state.answer : "";
}

function tools(r: RunResult): string[] {
  return r.toolTrace.map((t) => t.name);
}

// Every case needs the right tool call AND the right figure in the answer, as a whole number so
// that 8400 does not pass for 840.
function make(id: string, group: EvalCase["group"], prompt: string, mustCall: string[], expect: string, good: ScriptStep[]): EvalCase {
  const re = new RegExp(`(^|[^0-9.])${expect}([^0-9]|$)`);
  return {
    id,
    group,
    prompt,
    mustCall,
    good,
    check(r) {
      const used = tools(r);
      for (const t of mustCall) if (!used.includes(t)) return { pass: false, reason: `did not call ${t}` };
      if (!re.test(answerOf(r).replace(/,/g, ""))) return { pass: false, reason: `answer does not contain ${expect}` };
      return { pass: true };
    },
  };
}

export const CASES: EvalCase[] = [
  make("arithmetic", "tool-choice", "What is 17 * 23 + 4?", ["calculator"], "395", [
    script.toolCall("calculator", { expression: "17 * 23 + 4" }),
    script.text("17 * 23 + 4 is 395."),
  ]),
  make("lookup", "tool-choice", "Which port does the relay service listen on?", ["lookup"], "8472", [
    script.toolCall("lookup", { query: "relay service port" }),
    script.text("The relay service listens on port 8472."),
  ]),
  make("two-step", "tool-choice", "How many requests per day does the standard plan allow? Give me that number times 7.", ["lookup", "calculator"], "840", [
    script.toolCall("lookup", { query: "standard plan requests per day" }),
    script.toolCall("calculator", { expression: "120 * 7" }),
    script.text("The standard plan allows 120 requests per day, so 840 in a week."),
  ]),
  // The tool result is about 12 KB and is cut to a head and a tail with a marker. The last line sits in the tail.
  make("cut-result", "reading", "Look up the big report and tell me the sensor reading on its last line.", ["lookup"], "17", [
    script.toolCall("lookup", { query: "big report sensor reading" }),
    script.text("The last line, Line 400, has a sensor reading of 17."),
  ]),
];
