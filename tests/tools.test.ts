import assert from "node:assert/strict";
import { test } from "node:test";
import { ResultStore } from "../src/context.ts";
import { calculate, standardRegistry } from "../src/tools.ts";

test("calculator: precedence, parentheses, unary minus, power and modulo", () => {
  assert.equal(calculate("17 * 23 + 4"), 395);
  assert.equal(calculate("(1 + 2) * 3"), 9);
  assert.equal(calculate("-4 + 10"), 6);
  assert.equal(calculate("2 ^ 3 ^ 2"), 512);
  assert.equal(calculate("10 % 4"), 2);
  assert.equal(calculate("1.5 * 4"), 6);
});

test("calculator: refuses anything that is not arithmetic, so there is nothing to eval", () => {
  for (const bad of ["process.exit(1)", "1; 2", "require('fs')", "constructor", "1 +", "(1", "1 / 0", "", "9".repeat(300), "2 ^ 99999"]) {
    assert.throws(() => calculate(bad), Error, `should refuse: ${bad}`);
  }
});

test("calculator tool: answers with an error string instead of throwing", () => {
  const tool = standardRegistry(5).get("calculator")!;
  const ctx = { store: new ResultStore(), capBytes: 1000 };
  assert.equal(tool.run({ expression: "6*7" }, ctx), "42");
  assert.match(tool.run({ expression: "alert(1)" }, ctx), /^error:/);
});

test("lookup: finds the matching document and says so when nothing matches", () => {
  const tool = standardRegistry(5).get("lookup")!;
  const ctx = { store: new ResultStore(), capBytes: 1000 };
  assert.match(tool.run({ query: "which port does the relay service use" }, ctx), /8472/);
  assert.match(tool.run({ query: "standard plan requests" }, ctx), /120 requests per day/);
  assert.equal(tool.run({ query: "zzz qqq" }, ctx), "no matching document");
});

test("registry: sizes, and the real tools are always the first three", () => {
  assert.equal(standardRegistry(5).names().length, 5);
  assert.equal(standardRegistry(24).names().length, 24);
  assert.deepEqual(standardRegistry(24).names().slice(0, 3), ["calculator", "lookup", "read_result"]);
});

test("registry rank: a prompt about arithmetic ranks the calculator first, one about documents ranks lookup first", () => {
  const r = standardRegistry(24);
  assert.equal(r.rank("Evaluate this arithmetic expression")[0], "calculator");
  assert.equal(r.rank("Search the document set for the retention policy")[0], "lookup");
});

test("registry specs: an unknown name is skipped, not invented", () => {
  assert.deepEqual(standardRegistry(5).specs(["calculator", "ghost"]).map((s) => s.name), ["calculator"]);
});
