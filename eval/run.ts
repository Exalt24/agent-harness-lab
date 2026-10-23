// Usage:
//   node eval/run.ts --scripted [--trials 10] [--out file]
//   node eval/run.ts --ollama <model> [--trials 10] [--configs a,b] [--budget-min 8] [--resume] [--out file]
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { OllamaClient } from "../src/model.ts";
import { CONFIGS, pooled, runEval, summarize } from "./engine.ts";
import type { Cell, TrialRecord } from "./engine.ts";
import { formatRate, overlaps } from "./stats.ts";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const flag = (name: string): string | undefined => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);

const scripted = args.includes("--scripted");
const modelName = flag("--ollama");
if (!scripted && !modelName) {
  console.error("choose a backend: --scripted or --ollama <model>");
  process.exit(64);
}
const trialsPer = Number(flag("--trials") ?? 10);
const wanted = flag("--configs")?.split(",");
const configs = wanted ? CONFIGS.filter((c) => wanted.includes(c.id)) : CONFIGS;
if (configs.length === 0) {
  console.error(`no such configuration, choose from ${CONFIGS.map((c) => c.id).join(", ")}`);
  process.exit(64);
}
const date = new Date().toLocaleDateString("sv-SE"); // local calendar date, YYYY-MM-DD
const label = scripted ? "scripted" : `ollama-${modelName!.replace(/[^a-z0-9.]+/gi, "-")}`;
const outPath = resolve(flag("--out") ?? join(root, "eval", "reports", `${date}-${label}.json`));
const budgetMs = flag("--budget-min") ? Number(flag("--budget-min")) * 60_000 : undefined;

interface Report {
  date: string;
  backend: string;
  model: string | null;
  trialsPerCell: number;
  complete: boolean;
  environment: Record<string, unknown>;
  pooled: Cell[];
  pooledToolChoice: Cell[];
  cells: Cell[];
  trials: TrialRecord[];
}

async function environment(): Promise<Record<string, unknown>> {
  const env: Record<string, unknown> = { node: process.version, platform: process.platform, temperatureUnsetMeans: "no temperature sent, the model's own default applies", numCtx: 8192, think: false };
  if (scripted) return env;
  try {
    env["ollamaVersion"] = ((await (await fetch("http://127.0.0.1:11434/api/version")).json()) as { version: string }).version;
    const show = (await (await fetch("http://127.0.0.1:11434/api/show", { method: "POST", body: JSON.stringify({ model: modelName }) })).json()) as { details?: Record<string, unknown> };
    env["modelDetails"] = show.details;
  } catch (e) {
    env["ollamaProbe"] = `failed: ${(e as Error).message}`;
  }
  return env;
}

let existing: TrialRecord[] = [];
if (args.includes("--resume") && existsSync(outPath)) existing = (JSON.parse(readFileSync(outPath, "utf8")) as Report).trials;

const env = await environment();
const save = (trials: TrialRecord[], complete: boolean) => {
  const report: Report = {
    date, backend: scripted ? "scripted" : "ollama", model: modelName ?? null, trialsPerCell: trialsPer, complete,
    environment: env, pooled: configs.map((c) => pooled(trials, c.id)), pooledToolChoice: configs.map((c) => pooled(trials, c.id, "tool-choice")), cells: summarize(trials), trials,
  };
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(report, null, 2) + "\n");
  return report;
};

const { trials, complete } = await runEval({
  backend: scripted ? "scripted" : "ollama",
  ...(modelName ? { model: new OllamaClient(modelName) } : {}),
  trials: trialsPer,
  configs,
  existing,
  ...(budgetMs !== undefined ? { budgetMs } : {}),
  onTrial: (t, all) => {
    console.log(`${t.config} ${t.case} #${t.trial}: ${t.outcome.toUpperCase()} ${t.reason} (${t.ms} ms)`);
    if (all.length % 5 === 0) save(all, false);
  },
});
const report = save(trials, complete);

console.log(`\n${scripted ? "scripted backend (checks the pipeline, says nothing about any model)" : `ollama ${modelName}`}, ${date}, ${trialsPer} trials per cell`);
console.log("\nPer configuration, all cases pooled:");
for (const p of report.pooled) console.log(`  ${p.config.padEnd(20)} ${formatRate(p.pass, p.scored, p.error)}`);
console.log("\nPer configuration, tool-choice cases only (arithmetic, lookup, two-step):");
for (const p of report.pooledToolChoice) console.log(`  ${p.config.padEnd(20)} ${formatRate(p.pass, p.scored, p.error)}`);
console.log("\nPer case and configuration:");
for (const c of report.cells) console.log(`  ${c.case.padEnd(11)} ${c.config.padEnd(20)} ${formatRate(c.pass, c.scored, c.error)}`);
const byId = new Map(report.pooledToolChoice.map((p) => [p.config, p]));
const pair = (a: string, b: string, what: string) => {
  const x = byId.get(a);
  const y = byId.get(b);
  if (!x || !y) return;
  const o = overlaps(x.ci, y.ci);
  console.log(`  ${what}: ${o === null ? "not comparable" : o ? "the intervals overlap, so this run cannot separate them" : "the intervals do not overlap"}`);
};
console.log("\nComparisons on the tool-choice cases (95% intervals):");
pair("temp-unset-tools5", "temp0-tools5", "temperature unset vs 0, 5 tools");
pair("temp-unset-tools24", "temp0-tools24", "temperature unset vs 0, 24 tools");
pair("temp0-tools5", "temp0-tools24", "5 vs 24 tools, temperature 0");
pair("temp-unset-tools5", "temp-unset-tools24", "5 vs 24 tools, temperature unset");
console.log(`\nreport: ${outPath}${complete ? "" : "\nINCOMPLETE: the time budget ended the run, repeat the command with --resume"}`);
process.exit(complete ? 0 : 3);
