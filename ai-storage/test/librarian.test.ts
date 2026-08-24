/**
 * The loop, every way it ends, and one end-to-end navigation.
 *
 * No weights are involved. What these assert is that the *harness* behaves —
 * that a model which gets lost is recorded as lost rather than as failed, that
 * a lane running out is a result the model saw first, that a repeat is caught
 * where it happens. Whether Qwen can navigate is `bench/`, and it has not been
 * run.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { approxCounter } from "../src/context/budget.ts";
import { buildCorpus, viewOf } from "../src/bench/corpus.ts";
import { OracleNavigator, ScriptedModel } from "../src/model/fake.ts";
import { runLibrarian, librarianTools } from "../src/agents/librarian.ts";

const corpus = (size: number, plant = 1) =>
  buildCorpus({ size, plant, counter: approxCounter, seed: 7 });

test("the oracle finds a planted fact and reads a fraction of the corpus", async () => {
  const c = corpus(400);
  const fact = c.planted[0]!;
  const res = await runLibrarian(fact.question, {
    model: new OracleNavigator(fact.question, fact.answer),
    view: viewOf(c),
    counter: approxCounter,
  });

  assert.equal(res.ending, "done", res.error ?? "");
  assert.ok(res.answer?.includes(fact.answer), `answer was ${res.answer}`);
  assert.ok(res.cites.includes(fact.noteId), "it must cite the note it read the answer from");

  // The headline ratio. The oracle is the ceiling, not a result about a model.
  const loaded = res.spent["navigation"]! + res.spent["memory"]!;
  assert.ok(loaded > 0, "it has to read something");
  assert.ok(
    c.corpusTokens / loaded > 20,
    `ratio ${(c.corpusTokens / loaded).toFixed(1)}× — the store should be worth navigating`,
  );
  // And it must not have needed the whole corpus, which is the entire point.
  assert.ok(loaded < c.corpusTokens / 10);
});

test("the ratio grows with the corpus, which is the claim being made", async () => {
  // If reading scales with the store, the hierarchy buys nothing. This is that
  // property, asserted on the ceiling — a real model has to be measured
  // separately and might not have it.
  const ratios: number[] = [];
  for (const size of [200, 800, 3200]) {
    const c = corpus(size);
    const fact = c.planted[0]!;
    const res = await runLibrarian(fact.question, {
      model: new OracleNavigator(fact.question, fact.answer),
      view: viewOf(c),
      counter: approxCounter,
    });
    assert.equal(res.ending, "done", `${size}: ${res.error ?? ""}`);
    ratios.push(c.corpusTokens / (res.spent["navigation"]! + res.spent["memory"]!));
  }
  assert.ok(
    ratios[2]! > ratios[0]!,
    `ratios ${ratios.map((r) => r.toFixed(1)).join(", ")} — a hierarchy that does not scale is a list`,
  );
});

test("a model that answers in prose is recorded as prose, not as a failure", async () => {
  // Otherwise "could not navigate" and "navigated and ignored the protocol"
  // land in the same bucket, and they need different fixes.
  const c = corpus(50);
  const res = await runLibrarian("anything", {
    model: new ScriptedModel([{ kind: "text", text: "The policy is Zeta-17." }]),
    view: viewOf(c),
    counter: approxCounter,
  });
  assert.equal(res.ending, "prose");
  assert.equal(res.answer, "The policy is Zeta-17.");
  assert.deepEqual(res.cites, []);
});

test("a model that never stops hits the step cap and says so", async () => {
  const c = corpus(50);
  const res = await runLibrarian("anything", {
    model: new ScriptedModel(
      Array.from({ length: 20 }, (_, i) => ({
        kind: "tools" as const,
        calls: [{ name: "memory_index", arguments: { path: "/" + "architecture".slice(0, 1 + (i % 8)) } }],
      })),
    ),
    view: viewOf(c),
    counter: approxCounter,
    maxSteps: 5,
  });
  assert.equal(res.ending, "step_cap");
  assert.match(res.error!, /STEP_CAP: 5/);
  assert.equal(res.steps.length, 5);
});

test("the same call twice returns REPEATED_TOOL_LOOP rather than the same answer again", async () => {
  const c = corpus(50);
  const res = await runLibrarian("anything", {
    model: new ScriptedModel([
      { kind: "tools", calls: [{ name: "memory_index", arguments: { path: "/" } }] },
      { kind: "tools", calls: [{ name: "memory_index", arguments: { path: "/" } }] },
      { kind: "tools", calls: [{ name: "memory_done", arguments: { answer: "x", cites: [] } }] },
    ]),
    view: viewOf(c),
    counter: approxCounter,
  });
  assert.equal(res.ending, "done");
  // The second call cost nothing: the model was already stuck, and taking its
  // remaining budget away helps nobody.
  assert.equal(res.steps[1]!.calls[0]!.tokens, 0);
  assert.equal(res.steps[1]!.calls[0]!.ok, false);
});

test("a tool this role does not have comes back as a result, not a side effect", async () => {
  const c = corpus(50);
  const res = await runLibrarian("anything", {
    model: new ScriptedModel([
      { kind: "tools", calls: [{ name: "write_file", arguments: { path: "/etc/passwd", body: "x" } }] },
      { kind: "tools", calls: [{ name: "memory_done", arguments: { answer: "no", cites: [] } }] },
    ]),
    view: viewOf(c),
    counter: approxCounter,
  });
  assert.equal(res.ending, "done");
  assert.equal(res.steps[0]!.calls[0]!.ok, false);
  assert.equal(res.steps[0]!.calls[0]!.name, "write_file");
});

test("two unreadable replies end the run; there is no sampling until it parses", async () => {
  const c = corpus(50);
  const res = await runLibrarian("anything", {
    model: new ScriptedModel([
      { kind: "throw", message: "arguments are not JSON" },
      { kind: "throw", message: "arguments are not JSON" },
      { kind: "throw", message: "arguments are not JSON" },
    ]),
    view: viewOf(c),
    counter: approxCounter,
  });
  assert.equal(res.ending, "agent_failure");
  assert.match(res.error!, /MEMORY_AGENT_FAILURE/);
});

test("opening more than the memory lane holds is refused, and the model sees it first", async () => {
  const c = corpus(600);
  const ids = c.notes.slice(0, 40).map((n) => n.id);
  const res = await runLibrarian("anything", {
    model: new ScriptedModel(
      ids.map((id) => ({ kind: "tools" as const, calls: [{ name: "memory_open", arguments: { id } }] })),
    ),
    view: viewOf(c),
    counter: approxCounter,
    maxSteps: 40,
  });
  assert.ok(
    res.ending === "context_limit" || res.ending === "step_cap",
    `ended ${res.ending}`,
  );
  // Whatever the ending, the lane was never overspent. That is the invariant.
  assert.ok(res.spent["memory"]! <= 2300);
});

test("what was charged is what the tools returned, lane by lane", async () => {
  const c = corpus(200);
  const fact = c.planted[0]!;
  const res = await runLibrarian(fact.question, {
    model: new OracleNavigator(fact.question, fact.answer),
    view: viewOf(c),
    counter: approxCounter,
  });
  const fromSteps = res.steps.flatMap((s) => s.calls).reduce((n, c2) => n + c2.tokens, 0);
  const fromLedger = res.spent["navigation"]! + res.spent["memory"]!;
  assert.equal(fromSteps, fromLedger, "the per-call totals must add up to the ledger");
  // The prompt and the question were charged too, to their own lanes.
  assert.ok(res.spent["harness"]! > 0);
  assert.ok(res.spent["task"]! > 0);
});

test("the tool list handed to the model is exactly the role's capability", async () => {
  const c = corpus(50);
  const model = new ScriptedModel([{ kind: "text", text: "done" }]);
  await runLibrarian("q", { model, view: viewOf(c), counter: approxCounter });
  const names = model.seen[0]!.tools.map((t) => t.name).sort();
  assert.deepEqual(names, [
    "memory_done",
    "memory_find_exact",
    "memory_index",
    "memory_open",
    "memory_source",
  ]);
  assert.equal(librarianTools(viewOf(c)).length, 5);
  for (const n of names) assert.ok(!/write|delete|exec|shell/.test(n));
});
