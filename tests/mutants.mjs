// LIFESPAN: permanent
// Mutation harness. Each mutant removes or breaks ONE guard in a temp copy of the project and the
// suite must go red. A mutant the suite does not catch, or an anchor that no longer matches the
// source exactly once, fails the whole run. The temp copies live in .mutants/ inside the project so
// node finds node_modules by walking up; nothing is linked, and each copy is deleted after its run.
import { spawnSync } from "node:child_process";
import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/** @type {{name: string, guard: string, edits: {file: string, anchor: string, to: string}[]}[]} */
export const MUTANTS = [
  { name: "step-limit", guard: "hard step limit", edits: [{ file: "src/loop.ts", anchor: "if (steps >= cfg.maxSteps) {", to: "if (false) {" }] },
  { name: "repeat-call-guard", guard: "repeated identical call guard", edits: [{ file: "src/loop.ts", anchor: "if (seen > cfg.maxRepeatedCalls) {", to: "if (false) {" }] },
  { name: "per-call-timeout", guard: "per call timeout", edits: [{ file: "src/retry.ts", anchor: "const timeoutMs = Math.min(deps.callTimeoutMs, remaining);", to: "const timeoutMs = 1e9;" }] },
  {
    name: "total-time-budget",
    guard: "total time budget (both layers)",
    edits: [
      { file: "src/loop.ts", anchor: "if (clock.now() >= deadline) {", to: "if (false) {" },
      { file: "src/retry.ts", anchor: 'if (remaining <= 0) return { ok: false, reason: "time_budget", detail: "total time budget used up" };', to: "" },
      { file: "src/retry.ts", anchor: "if (e.kind === \"timeout\" && clock.now() >= deps.deadline) {", to: "if (false) {" },
      { file: "src/retry.ts", anchor: "if (clock.now() + delay >= deps.deadline) {", to: "if (false) {" },
    ],
  },
  { name: "retry-only-on-5xx", guard: "retry only transport and 5xx", edits: [{ file: "src/model.ts", anchor: "this.status >= 500", to: "this.status >= 400" }] },
  { name: "no-backoff-growth", guard: "exponential backoff", edits: [{ file: "src/retry.ts", anchor: "const exp = Math.min(p.capMs, p.baseMs * 2 ** attempt);", to: "const exp = p.baseMs;" }] },
  { name: "stub-marker", guard: "visible marker in a byte-cap stub", edits: [{ file: "src/context.ts", anchor: "`${STUB_MARKER}: ${omitted} of", to: "`${omitted} of" }] },
  { name: "stub-not-stored", guard: "full text kept for a stub", edits: [{ file: "src/context.ts", anchor: "const id = store.put(text);", to: 'const id = "r0";' }] },
  { name: "no-byte-cap", guard: "byte cap itself", edits: [{ file: "src/context.ts", anchor: "if (total <= capBytes) return", to: "if (true) return" }] },
  { name: "truncation-detection", guard: "suspected truncation detection", edits: [{ file: "src/loop.ts", anchor: "reported < cfg.truncationRatio * estimated", to: "false" }] },
  { name: "empty-length-narrowing", guard: "empty reply with length retries narrower once", edits: [{ file: "src/loop.ts", anchor: 'reply.finishReason === "length" && !narrowed', to: "false" }] },
  { name: "unknown-tool-executes", guard: "unknown or unoffered tool never executes", edits: [{ file: "src/loop.ts", anchor: "if (!tool || !offered.includes(tc.name)) {", to: "if (!tool) {" }] },
  { name: "schema-validation", guard: "structured output schema check", edits: [{ file: "src/structured.ts", anchor: "const parsed = schema.safeParse(json.value);", to: "const parsed = { success: true, data: json.value };" }] },
  { name: "fail-closed", guard: "structured output fails closed", edits: [{ file: "src/structured.ts", anchor: "return { ok: false, reason: lastKind, detail: lastError, attempts: 2 };", to: "return { ok: true, data: {}, attempts: 2 };" }] },
  { name: "reask-once", guard: "structured output re-asks once", edits: [{ file: "src/structured.ts", anchor: "attempt <= 2", to: "attempt <= 1" }] },
  { name: "temperature-dropped", guard: "explicit temperature sent", edits: [{ file: "src/model.ts", anchor: "if (options.temperature !== undefined) ollamaOptions", to: "if (false) ollamaOptions" }] },
  { name: "keep-first-question", guard: "trim keeps system prompt and first question", edits: [{ file: "src/context.ts", anchor: "const head = clean.slice(0, 2);", to: "const head = clean.slice(0, 1);" }] },
  { name: "read-result-offered", guard: "read_result offered once a stub exists", edits: [{ file: "src/loop.ts", anchor: "if (store.size > 0 && registry.get(\"read_result\")", to: "if (false && registry.get(\"read_result\")" }] },
];

function count(text, needle) {
  let n = 0;
  for (let i = text.indexOf(needle); i !== -1; i = text.indexOf(needle, i + needle.length)) n++;
  return n;
}

function runSuite(cwd) {
  const r = spawnSync(process.execPath, ["--test", "--test-reporter=spec", "--test-timeout=4000", "tests/*.test.ts"], { cwd, encoding: "utf8", timeout: 120_000 });
  const out = `${r.stdout ?? ""}${r.stderr ?? ""}`;
  const failed = Number(/ℹ fail (\d+)/.exec(out)?.[1] ?? NaN);
  return { green: r.status === 0 && failed === 0, status: r.status, signal: r.signal, failed, out };
}

function copyProject(dest) {
  rmSync(dest, { recursive: true, force: true });
  mkdirSync(dest, { recursive: true });
  const skipReports = (from) => !from.includes(`${join("eval", "reports")}`);
  for (const part of ["src", "tests", "eval", "package.json", "tsconfig.json"]) {
    cpSync(join(root, part), join(dest, part), { recursive: true, filter: skipReports });
  }
}

const args = process.argv.slice(2);
const only = args.includes("--only") ? args[args.indexOf("--only") + 1] : undefined;
const shardArg = args.includes("--shard") ? args[args.indexOf("--shard") + 1] : undefined;
const noBaseline = args.includes("--no-baseline");

let list = MUTANTS.filter((m) => (only ? m.name === only : true));
if (shardArg) {
  const [i, n] = shardArg.split("/").map(Number);
  list = list.filter((_, k) => k % n === i - 1);
}

if (!noBaseline) {
  const dest = join(root, ".mutants", "_baseline");
  copyProject(dest);
  const base = runSuite(dest);
  rmSync(dest, { recursive: true, force: true });
  if (!base.green) {
    console.error("BASELINE IS RED: the unmutated copy fails its own suite, so no mutant result means anything.\n" + base.out.slice(-1500));
    process.exit(2);
  }
  console.log("baseline: green");
}

let killed = 0;
const survivors = [];
const stale = [];
for (const m of list) {
  const dest = join(root, ".mutants", m.name);
  copyProject(dest);
  let bad = false;
  for (const e of m.edits) {
    const path = join(dest, e.file);
    const text = readFileSync(path, "utf8");
    const found = count(text, e.anchor);
    if (found !== 1) {
      stale.push(`${m.name}: anchor in ${e.file} matched ${found} times (need exactly 1): ${e.anchor}`);
      bad = true;
      break;
    }
    writeFileSync(path, text.replace(e.anchor, () => e.to));
  }
  if (!bad) {
    const r = runSuite(dest);
    if (r.green) survivors.push(m.name);
    else {
      killed++;
      const how = r.signal ? `killed by ${r.signal}` : Number.isNaN(r.failed) ? "suite crashed" : `${r.failed} tests failed`;
      console.log(`caught   ${m.name.padEnd(24)} ${how}  (${m.guard})`);
    }
  }
  rmSync(dest, { recursive: true, force: true });
}
for (const s of survivors) console.log(`SURVIVED ${s}`);
for (const s of stale) console.log(`STALE    ${s}`);
console.log(`mutants: ${list.length}, caught: ${killed}, survived: ${survivors.length}, stale anchors: ${stale.length}`);
process.exit(survivors.length || stale.length ? 1 : 0);
