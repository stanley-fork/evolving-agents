/**
 * The index, and the two things splitting must never do.
 *
 * It must never leave a node over budget, and it must never lose a note. The
 * second is asserted over generated stores rather than over one example,
 * because losing a note is the kind of failure that shows up on the one shape
 * nobody wrote a case for.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { approxCounter, type TokenCounter } from "../src/context/budget.ts";
import {
  DEFAULT_LIMITS,
  IndexTooDeep,
  IndexTooLarge,
  assertWithinBudget,
  budgetForNode,
  depthOf,
  planSplit,
  reachableNotes,
  renderNode,
  splitUntilFits,
  type IndexEntry,
  type IndexNode,
} from "../src/index/tree.ts";

/** One token per entry, so a budget of N means N entries exactly. */
const perLine: TokenCounter = {
  describe: "one token per line — a test fixture, not a tokenizer",
  count: (t) => t.split("\n").length,
};

const notes = (names: string[]): IndexEntry[] =>
  names.map((name) => ({ name, kind: "note" as const }));

test("a node renders directories before notes, each alphabetical", () => {
  const node: IndexNode = {
    path: "/architecture",
    entries: [
      { name: "zeta", kind: "note" },
      { name: "storage", kind: "dir" },
      { name: "alpha", kind: "note" },
      { name: "agents", kind: "dir" },
    ],
  };
  assert.equal(
    renderNode(node),
    ["/architecture", "  agents/", "  storage/", "  alpha", "  zeta"].join("\n"),
  );
});

test("the text that is counted is the text that is sent", () => {
  // One function, so a budget cannot be computed against a different string
  // from the one the model sees.
  const node: IndexNode = { path: "/x", entries: notes(["a", "b", "c"]) };
  assert.equal(perLine.count(renderNode(node)), 4);
});

test("the root gets the root budget and everything else the node budget", () => {
  assert.equal(budgetForNode({ path: "/", entries: [] }), DEFAULT_LIMITS.rootMaxTokens);
  assert.equal(budgetForNode({ path: "/a", entries: [] }), DEFAULT_LIMITS.nodeMaxTokens);
});

test("a node over budget is refused rather than rendered", () => {
  const limits = { ...DEFAULT_LIMITS, nodeMaxTokens: 5 };
  const node: IndexNode = { path: "/big", entries: notes(["a", "b", "c", "d", "e", "f"]) };
  assert.throws(
    () => assertWithinBudget(node, perLine, limits),
    (err: unknown) => {
      assert.ok(err instanceof IndexTooLarge);
      assert.equal(err.detail.path, "/big");
      assert.equal(err.detail.budget, 5);
      assert.equal(err.detail.tokens, 7);
      return true;
    },
  );
});

test("a node too deep is refused", () => {
  const limits = { ...DEFAULT_LIMITS, maximumDepth: 2 };
  assert.equal(depthOf("/a/b/c"), 3);
  assert.throws(() => assertWithinBudget({ path: "/a/b/c", entries: [] }, perLine, limits),
    IndexTooDeep);
});

test("a node within budget is not split", () => {
  const node: IndexNode = { path: "/small", entries: notes(["a", "b"]) };
  assert.equal(planSplit(node, perLine, { ...DEFAULT_LIMITS, nodeMaxTokens: 10 }), null);
});

test("splitting groups by prefix and leaves the parent pointing at the children", () => {
  const limits = { ...DEFAULT_LIMITS, nodeMaxTokens: 4 };
  const node: IndexNode = {
    path: "/architecture",
    entries: notes(["agent-caps", "agent-routing", "storage-limits", "storage-scopes"]),
  };
  const plan = planSplit(node, perLine, limits);
  assert.ok(plan);
  assert.deepEqual(
    plan.parent.entries.map((e) => `${e.kind}:${e.name}`).sort(),
    ["dir:agent", "dir:storage"],
  );
  assert.deepEqual(plan.children.map((c) => c.path).sort(), [
    "/architecture/agent",
    "/architecture/storage",
  ]);
  // The parent keeps any directories it already had.
  const withDir: IndexNode = { ...node, entries: [...node.entries, { name: "kept", kind: "dir" }] };
  const plan2 = planSplit(withDir, perLine, limits)!;
  assert.ok(plan2.parent.entries.some((e) => e.kind === "dir" && e.name === "kept"));
});

test("when prefixes do not divide the node, it falls back to buckets", () => {
  // Every name shares a prefix, so grouping by it would produce one child the
  // size of the parent — a split that does not split.
  const limits = { ...DEFAULT_LIMITS, nodeMaxTokens: 6 };
  const names = Array.from({ length: 40 }, (_, i) => `note${String(i).padStart(3, "0")}`);
  const node: IndexNode = { path: "/flat", entries: notes(names) };
  const plan = planSplit(node, perLine, limits)!;
  assert.ok(plan.children.length > 1, "one child is not a split");
  const total = plan.children.reduce((n, c) => n + c.entries.length, 0);
  assert.equal(total, names.length);
});

test("splitting until it fits leaves nothing over budget and loses nothing", () => {
  for (const count of [1, 2, 7, 40, 250, 1000]) {
    for (const shape of ["flat", "prefixed", "mixed"] as const) {
      const names = makeNames(count, shape);
      const root: IndexNode = { path: "/", entries: notes(names) };
      const limits = { ...DEFAULT_LIMITS, rootMaxTokens: 12, nodeMaxTokens: 12 };

      let out;
      try {
        out = splitUntilFits(root, perLine, limits);
      } catch (err) {
        // The one legitimate refusal: a store so uniform that it runs out of
        // depth. It must say so rather than return a tree that is still over
        // budget.
        assert.ok(err instanceof IndexTooDeep || err instanceof IndexTooLarge);
        continue;
      }

      for (const node of out.nodes) assertWithinBudget(node, perLine, limits);

      const kept = reachableNotes(out.nodes);
      assert.equal(kept.size, names.length, `${shape}/${count}: a split lost a note`);
      for (const n of names) assert.ok(kept.has(n), `${shape}/${count}: ${n} became unreachable`);
    }
  }
});

test("a single note that cannot fit a node on its own is reported, not swallowed", () => {
  // A naming problem rather than a structural one, and the error names it.
  const limits = { ...DEFAULT_LIMITS, nodeMaxTokens: 1 };
  const node: IndexNode = { path: "/x", entries: notes(["only-one"]) };
  assert.throws(() => splitUntilFits(node, perLine, limits), IndexTooLarge);
});

test("the real estimating counter also settles", () => {
  // The fixture counter is one token per line; the estimate is chars/4. A split
  // rule that only converges for one of them is a split rule tuned to a test.
  const names = makeNames(600, "mixed");
  const root: IndexNode = { path: "/", entries: notes(names) };
  const out = splitUntilFits(root, approxCounter, DEFAULT_LIMITS);
  for (const node of out.nodes) assertWithinBudget(node, approxCounter, DEFAULT_LIMITS);
  assert.equal(reachableNotes(out.nodes).size, names.length);
});

function makeNames(count: number, shape: "flat" | "prefixed" | "mixed"): string[] {
  const out: string[] = [];
  const groups = ["agent", "storage", "model", "permission", "experiment", "failure"];
  for (let i = 0; i < count; i += 1) {
    if (shape === "flat") out.push(`note${String(i).padStart(4, "0")}`);
    else if (shape === "prefixed") out.push(`${groups[i % groups.length]}-${i}`);
    else out.push(i % 3 === 0 ? `note${i}` : `${groups[i % groups.length]}-${i}`);
  }
  return out;
}
