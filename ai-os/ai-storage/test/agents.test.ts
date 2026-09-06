/**
 * The specialists, against a scripted model.
 *
 * What these assert is the code around the model: that a proposal citing bytes
 * it was not shown is thrown out, that a barren passage is a real outcome, that
 * an invented note id cannot supersede a real note, that a coordinator with no
 * capability of its own cannot acquire one. Whether the model makes good
 * judgements is `bench/`, and it has not been run.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { approxCounter } from "../src/context/budget.ts";
import { archiveRange, archiveSource, archivistBudget } from "../src/agents/archivist.ts";
import { placeNote, placementIsAllowed, normalise, reconcile } from "../src/agents/reconciler.ts";
import { keepSource, shortlist } from "../src/agents/memory-keeper.ts";
import { FileStore } from "../src/backend/filesystem.ts";
import { NoteRepository } from "../src/knowledge/repository.ts";
import { LexicalIndex, tokenise } from "../src/search/lexical.ts";
import type { LocalModel, StructuredResult } from "../src/model/local-model.ts";
import type { KnowledgeNote, Scope } from "../src/knowledge/schema.ts";

const SCOPE: Scope = { type: "project", id: "demo" };

/** A model that returns a fixed structured value, or throws. */
function structuredModel(values: unknown[]): LocalModel & { calls: number } {
  let i = 0;
  const m = {
    calls: 0,
    describe: { engine: "scripted", baseUrl: "none", model: "scripted" },
    async structured<T>(): Promise<StructuredResult<T>> {
      m.calls += 1;
      const v = values[i++] ?? values[values.length - 1];
      if (v instanceof Error) throw v;
      return {
        value: v as T,
        usage: { promptTokens: 1, reasoningTokens: 0, completionTokens: 1, latencyMs: 1, timeToFirstTokenMs: null },
      };
    },
    async complete(): Promise<never> {
      throw new Error("not scripted");
    },
    async tools(): Promise<never> {
      throw new Error("not scripted");
    },
    async served() {
      return { id: "scripted", contextLength: null, readFrom: "none" };
    },
  } as unknown as LocalModel & { calls: number };
  return m;
}

const TEXT = "A".repeat(200) + " the build fails under Node 20 because package X requires Node 22. " + "B".repeat(200);

// ---- Archivist -------------------------------------------------------------

test("a proposal citing bytes it was not shown is thrown out", async () => {
  const model = structuredModel([
    {
      notes: [
        {
          title: "Node 22 is required",
          type: "constraint",
          claim: "The build fails under Node 20 because package X requires Node 22.",
          keywords: ["node", "build"],
          source: { artifact: "a.md", from: 0, to: 50 },
        },
        {
          title: "Invented",
          type: "fact",
          claim: "Something read from elsewhere.",
          keywords: [],
          // Outside the passage it was handed. A guess with a byte range on it
          // is the most dangerous shape a guess can take.
          source: { artifact: "a.md", from: 900, to: 950 },
        },
      ],
    },
  ]);
  const out = await archiveRange("a.md", TEXT, { from: 0, to: 200 }, {
    model,
    counter: approxCounter,
    budget: archivistBudget(),
  });
  assert.equal(out.proposals.length, 1);
  assert.equal(out.rejected.length, 1);
  assert.match(out.rejected[0]!.why, /outside the passage/);
});

test("a passage with nothing in it is a real answer, not a failure", async () => {
  const model = structuredModel([{ notes: [] }]);
  const out = await archiveRange("a.md", TEXT, { from: 0, to: 100 }, {
    model,
    counter: approxCounter,
    budget: archivistBudget(),
  });
  assert.deepEqual(out.proposals, []);
  assert.deepEqual(out.rejected, []);
  assert.equal(out.error, null);
});

test("two unusable replies stop the archivist; there is no sampling until it parses", async () => {
  const model = structuredModel([new Error("not JSON"), new Error("not JSON"), new Error("not JSON")]);
  const out = await archiveRange("a.md", TEXT, { from: 0, to: 100 }, {
    model,
    counter: approxCounter,
    budget: archivistBudget(),
  });
  assert.match(out.error!, /MEMORY_AGENT_FAILURE/);
  assert.equal(model.calls, 3, "one attempt and two retries, then stop");
});

test("a chunk that does not fit is reported rather than trimmed", async () => {
  const model = structuredModel([{ notes: [] }]);
  const huge = "x".repeat(200_000);
  const out = await archiveRange("a.md", huge, { from: 0, to: 200_000 }, {
    model,
    counter: approxCounter,
    budget: archivistBudget(),
  });
  assert.match(out.error!, /MEMORY_CONTEXT_LIMIT/);
  assert.equal(model.calls, 0, "nothing may be sent that was going to be cut");
});

test("walking a source covers all of it, and a barren chunk still counts as read", async () => {
  const model = structuredModel([{ notes: [] }]);
  const ranges: Array<{ from: number; to: number }> = [];
  for await (const out of archiveSource("a.md", TEXT, [], {
    model,
    counter: approxCounter,
    budget: archivistBudget(),
    chunkBytes: 100,
  }))
    ranges.push(out.range);
  assert.equal(ranges[0]!.from, 0);
  assert.equal(ranges[ranges.length - 1]!.to, TEXT.length);
  // Otherwise a passage with no knowledge in it is re-read on every run and
  // indexing never terminates.
  assert.equal(ranges.length, Math.ceil(TEXT.length / 100));
});

// ---- Reconciler ------------------------------------------------------------

const note = (id: string, claim: string): KnowledgeNote => ({
  version: 1,
  id,
  scope: SCOPE,
  title: claim.slice(0, 40),
  type: "fact",
  claim,
  keywords: ["node", "build"],
  evidence: [{ sourceId: "s", path: "p", from: 0, to: 10, sha256: "a".repeat(64) }],
  relations: { related: [], supersedes: [] },
  state: "active",
  createdAt: "2026-01-01T00:00:00.000Z",
  updatedAt: "2026-01-01T00:00:00.000Z",
  revision: "rev1",
});

const proposal = {
  title: "Node 22 is required",
  type: "constraint" as const,
  claim: "The build fails under Node 20.",
  keywords: ["node"],
  source: { artifact: "a.md", from: 0, to: 10 },
};

test("nothing to compare against is not a question worth a model call", async () => {
  const model = structuredModel([{ action: "same", existing: "kn_whatever" }]);
  assert.deepEqual(await reconcile(proposal, [], { model }), { action: "new" });
  assert.equal(model.calls, 0);
});

test("an invented id cannot supersede a real note", async () => {
  const model = structuredModel([{ action: "supersedes", existing: "kn_does-not-exist_00000000" }]);
  const verdict = await reconcile(proposal, [note("kn_a_1", "old claim")], { model });
  // Falls back to `new`. Guessing which note it meant would let a hallucinated
  // reference retire a real one; a duplicate costs less than a lost claim.
  assert.deepEqual(verdict, { action: "new" });
});

test("conflict is carried through with its explanation", async () => {
  const model = structuredModel([
    { action: "conflict", existing: "kn_a_1", reason: "two runs, two tolerances" },
  ]);
  const verdict = await reconcile(proposal, [note("kn_a_1", "old claim")], { model });
  assert.equal(verdict.action, "conflict");
  assert.match((verdict as { explanation: string }).explanation, /two tolerances/);
});

// ---- Indexer ---------------------------------------------------------------

test("a path the model invented is discarded, not resolved", () => {
  assert.equal(normalise("/a/../../etc"), "/");
  assert.equal(normalise("a/b/"), "/a/b");
  assert.equal(normalise("//a//b"), "/a/b");
});

test("placement may create one level under an existing parent, and no more", () => {
  const existing = new Set(["/", "/deployment", "/deployment/keys"]);
  assert.deepEqual(placementIsAllowed({ directory: "/deployment/keys", aliases: [], related: [] }, existing), {
    ok: true,
    create: null,
  });
  assert.deepEqual(placementIsAllowed({ directory: "/deployment/rotation", aliases: [], related: [] }, existing), {
    ok: true,
    create: "/deployment/rotation",
  });
  // Four levels the model invented would grow a tree nobody navigates, one
  // note per directory — the index failing in the shape hardest to notice.
  const deep = placementIsAllowed({ directory: "/a/b/c/d", aliases: [], related: [] }, existing);
  assert.equal(deep.ok, false);
});

test("the indexer's answer is normalised before anything else sees it", async () => {
  const model = structuredModel([{ directory: "deployment/../../etc", aliases: [], related: [] }]);
  const placement = await placeNote(proposal, [{ path: "/", entries: [] }], { model });
  assert.equal(placement.directory, "/");
});

// ---- MemoryKeeper ----------------------------------------------------------

test("the keeper counts every outcome, including the ones that did nothing", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ai-storage-keeper-"));
  const store = new FileStore(dir);
  await store.init();
  const repo = new NoteRepository(store);
  await store.writeText(`projects/demo/sources/a.md`, TEXT);

  const model = structuredModel([
    {
      notes: [
        {
          title: "Node 22 is required",
          type: "constraint",
          claim: "The build fails under Node 20 because package X requires Node 22.",
          keywords: ["node", "build"],
          source: { artifact: "a.md", from: 0, to: 120 },
        },
        {
          title: "Bad citation",
          type: "fact",
          claim: "Read from bytes that do not exist.",
          keywords: [],
          source: { artifact: "a.md", from: 0, to: 999_999 },
        },
      ],
    },
    // Every later call — reconciler and further chunks — says new / nothing.
    { action: "new" },
  ]);

  const report = await keepSource("a.md", TEXT, [], {
    repository: repo,
    reader: store.sourceReader("projects/demo"),
    scope: SCOPE,
    model,
    counter: approxCounter,
    now: () => "2026-08-24T10:00:00.000Z",
    chunkBytes: 500,
  });

  assert.equal(report.proposed >= 1, true);
  // One citation ran past the end of the source. Counted separately from a
  // schema failure, because they are different problems with different fixes.
  assert.ok(report.rejectedByEvidence >= 1 || report.rejectedByValidator >= 1);
  assert.ok(report.read.length >= 1);
  assert.ok(report.created >= 1);
  const ids = await repo.ids(SCOPE);
  assert.ok(ids.length >= 1);
  await rm(dir, { recursive: true, force: true });
});

test("the reconciler is shown a bounded shortlist, not the whole scope", () => {
  const many = Array.from({ length: 500 }, (_, i) => note(`kn_n${i}_0000000${i % 10}`, `claim about node ${i}`));
  const short = shortlist(many, "node build", 5);
  assert.equal(short.length, 5);
  // Otherwise the Reconciler's prompt grows with the store, which is the
  // failure this whole component exists to avoid, inside one of its own agents.
  assert.ok(short.every((n) => many.includes(n)));
});

// ---- lexical ---------------------------------------------------------------

test("search is exact, weighted, and the same twice", () => {
  const ix = LexicalIndex.of([
    note("kn_a_00000001", "The build fails under Node 20 because package X requires Node 22."),
    note("kn_b_00000002", "Node is irrelevant to the storage layer."),
  ]);
  const first = ix.search("node build");
  const second = ix.search("node build");
  assert.deepEqual(first, second, "the same query must give the same order");
  assert.equal(first[0]!.id, "kn_a_00000001");
  // Counts travel with the score, because a score alone cannot be checked.
  assert.ok(first[0]!.matched >= 1);
  assert.equal(ix.search("nodes").length, 0, "exact means exact — no stemming");
  assert.deepEqual(tokenise("Q4_K_M and n_ctx"), ["q4_k_m", "and", "n_ctx"]);
});

test("a note matching more of the query outranks one matching one word often", () => {
  const spammy = note("kn_spam_00000001", ("node ".repeat(40)).trim());
  const real = note("kn_real_00000002", "node build fails");
  const ix = LexicalIndex.of([spammy, real]);
  const hits = ix.search("node build fails");
  assert.equal(hits[0]!.id, "kn_real_00000002");
});
