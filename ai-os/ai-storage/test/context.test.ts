/**
 * The invariant every number this component produces depends on.
 *
 * If a read can be silently trimmed, the benchmark's denominator — *tokens the
 * model actually had to see* — is a fiction, and so is the headline ratio. So
 * these are not tests of a helper; they are the tests that decide whether any
 * result from ai-storage means anything.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  BudgetLedger,
  ContextLimitExceeded,
  DEFAULT_BUDGET,
  approxCounter,
  assertBudget,
  budgetFor,
} from "../src/context/budget.ts";

test("the default lanes fit inside the default total", () => {
  assertBudget({ ...DEFAULT_BUDGET });
  const sum =
    DEFAULT_BUDGET.harness +
    DEFAULT_BUDGET.task +
    DEFAULT_BUDGET.navigation +
    DEFAULT_BUDGET.memory +
    DEFAULT_BUDGET.generation;
  assert.equal(sum, DEFAULT_BUDGET.total, "the specification's five numbers must add to 8192");
});

test("a budget whose lanes overflow its total is refused when it is built", () => {
  assert.throws(
    () => assertBudget({ ...DEFAULT_BUDGET, memory: DEFAULT_BUDGET.memory + 1 }),
    /more than the 8192 available/,
  );
});

test("an oversized read is refused, never truncated", () => {
  const led = new BudgetLedger();
  const before = led.remainingIn("memory");
  assert.throws(
    () => led.spend("memory", before + 1, "open fewer notes"),
    (err: unknown) => {
      assert.ok(err instanceof ContextLimitExceeded);
      assert.equal(err.detail.error, "MEMORY_CONTEXT_LIMIT");
      assert.equal(err.detail.lane, "memory");
      assert.equal(err.detail.requestedTokens, before + 1);
      assert.equal(err.detail.availableTokens, before);
      return true;
    },
  );
  // Nothing was charged. A refusal that half-charges is a refusal that leaves
  // the ledger describing a read that did not happen.
  assert.equal(led.remainingIn("memory"), before);
  assert.equal(led.spent(), 0);
});

test("the error carries what was asked for and what was left", () => {
  const led = new BudgetLedger();
  led.spend("memory", 2000, "");
  try {
    led.spend("memory", 400, "narrow the query");
    assert.fail("should have thrown");
  } catch (err) {
    assert.ok(err instanceof ContextLimitExceeded);
    // These two numbers are what a harness needs to ask for less. A bare
    // "context exceeded" gives it nothing to do.
    assert.equal(err.detail.requestedTokens, 400);
    assert.equal(err.detail.availableTokens, 300);
    assert.equal(err.detail.hint, "narrow the query");
  }
});

test("lanes are separate: spending navigation cannot eat the memory lane", () => {
  const led = new BudgetLedger();
  led.spend("navigation", DEFAULT_BUDGET.navigation, "walked the whole index");
  assert.equal(led.remainingIn("navigation"), 0);
  // The failure this prevents: a Librarian that navigates beautifully and then
  // has no room to read the note it found.
  assert.equal(led.remainingIn("memory"), DEFAULT_BUDGET.memory);
  led.spend("memory", DEFAULT_BUDGET.memory, "read the note");
  assert.equal(led.spent(), DEFAULT_BUDGET.navigation + DEFAULT_BUDGET.memory);
});

test("fits() agrees with spend(), so nothing is assembled to be thrown away", () => {
  const led = new BudgetLedger();
  for (const n of [0, 1, 2299, 2300, 2301]) {
    const fits = led.fits("memory", n);
    if (fits) continue;
    assert.throws(() => led.spend("memory", n, ""), ContextLimitExceeded);
  }
  assert.ok(led.fits("memory", DEFAULT_BUDGET.memory));
  assert.ok(!led.fits("memory", DEFAULT_BUDGET.memory + 1));
});

test("a scaled budget still fits, and still spends every token", () => {
  for (const total of [2048, 4096, 8192, 16384, 32768, 262144]) {
    const b = budgetFor(total);
    assert.equal(b.total, total);
    const sum = b.harness + b.task + b.navigation + b.memory + b.generation;
    assert.equal(sum, total, `lanes at ${total} must add to the total exactly`);
    assert.ok(b.generation > 0, "generation must not be starved by rounding");
  }
});

test("a budget of nothing is refused rather than scaled to nothing", () => {
  assert.throws(() => budgetFor(0), /positive integer/);
  assert.throws(() => budgetFor(-1), /positive integer/);
  assert.throws(() => budgetFor(1.5), /positive integer/);
});

test("the estimating counter says in its own description that it is an estimate", () => {
  // A run record that carries "chars/4" instead of a tokenizer name is a run
  // record whose ratios are approximate, and the reader can see that without
  // having to ask.
  assert.match(approxCounter.describe, /not a tokenizer/);
  assert.equal(approxCounter.count(""), 0);
  assert.equal(approxCounter.count("abcd"), 1);
  assert.equal(approxCounter.count("abcde"), 2);
});

test("a snapshot reports every lane and the total they add to", () => {
  const led = new BudgetLedger();
  led.spend("harness", 100, "");
  led.spend("navigation", 250, "");
  const snap = led.snapshot();
  assert.equal(snap.harness, 100);
  assert.equal(snap.navigation, 250);
  assert.equal(snap.memory, 0);
  assert.equal(snap.total, 350);
});
