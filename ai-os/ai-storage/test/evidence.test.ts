/**
 * Whether a citation is true.
 *
 * The model says *I read this at bytes 1100 to 2450 of report.md*. These tests
 * are the ones that decide whether that sentence survives, and every one of
 * them starts from a proposal that a schema would have accepted.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import {
  EvidenceRejected,
  allowAll,
  evidenceFor,
  recheck,
  sha256Of,
  type Acl,
  type SourceReader,
} from "../src/provenance/evidence.ts";
import { proposalFrom } from "../src/knowledge/schema.ts";
import type { Range } from "../src/provenance/ranges.ts";

/** A store that holds bytes and resolves names itself. The model never sees it. */
function storeOf(files: Record<string, string>): SourceReader & { set(k: string, v: string): void } {
  const bytes = new Map<string, Uint8Array>();
  for (const [k, v] of Object.entries(files)) bytes.set(k, new TextEncoder().encode(v));
  return {
    set(k, v) {
      bytes.set(k, new TextEncoder().encode(v));
    },
    async sizeOf(a: string) {
      return bytes.get(a)?.byteLength ?? null;
    },
    async slice(a: string, r: Range) {
      const b = bytes.get(a);
      if (!b) throw new Error("no such source");
      return b.subarray(r.from, r.to);
    },
    async idOf(a: string) {
      return "src_" + a;
    },
    async pathOf(a: string) {
      return a;
    },
  };
}

const BODY = "0123456789".repeat(20); // 200 bytes
const store = () => storeOf({ "notes/a.md": BODY });

const prop = (from: number, to: number, artifact = "notes/a.md") =>
  proposalFrom({
    title: "t",
    type: "fact",
    claim: "c",
    keywords: [],
    source: { artifact, from, to },
  });

test("a citation into bytes that exist becomes evidence with a digest of the slice", async () => {
  const e = await evidenceFor(prop(10, 20), store());
  assert.equal(e.from, 10);
  assert.equal(e.to, 20);
  assert.equal(e.path, "notes/a.md");
  assert.equal(e.sourceId, "src_notes/a.md");
  // Of the slice, not of the file. This is the whole reason a regenerated
  // artifact does not invalidate every note that cited any part of it.
  const expect = createHash("sha256").update(BODY.slice(10, 20)).digest("hex");
  assert.equal(e.sha256, expect);
  assert.notEqual(e.sha256, createHash("sha256").update(BODY).digest("hex"));
});

test("a citation into a source that does not exist is refused", async () => {
  await assert.rejects(
    () => evidenceFor(prop(0, 10, "notes/invented.md"), store()),
    (err: unknown) => {
      assert.ok(err instanceof EvidenceRejected);
      assert.equal(err.detail.reason, "NO_SUCH_SOURCE");
      return true;
    },
  );
});

test("a citation past the end of a source is refused, not clamped", async () => {
  // Clamping would mint a digest of bytes the model never claimed to read.
  await assert.rejects(
    () => evidenceFor(prop(190, 400), store()),
    (err: unknown) => {
      assert.ok(err instanceof EvidenceRejected);
      assert.equal(err.detail.reason, "RANGE_OUTSIDE_SOURCE");
      return true;
    },
  );
});

test("the whole source is allowed; one byte more is not", async () => {
  await assert.doesNotReject(() => evidenceFor(prop(0, 200), store()));
  await assert.rejects(() => evidenceFor(prop(0, 201), store()), EvidenceRejected);
});

test("an ACL denial happens before anything is read", async () => {
  let reads = 0;
  const s = store();
  const counting: SourceReader = { ...s, slice: async (a, r) => (reads += 1, s.slice(a, r)) };
  const acl: Acl = { denies: (a) => (a.startsWith("notes/") ? "outside this agent's scope" : null) };
  await assert.rejects(
    () => evidenceFor(prop(0, 10), counting, { acl }),
    (err: unknown) => {
      assert.ok(err instanceof EvidenceRejected);
      assert.equal(err.detail.reason, "FORBIDDEN");
      return true;
    },
  );
  assert.equal(reads, 0, "a forbidden source must not be read in order to find out it is forbidden");
});

test("the flow id travels with the evidence when there is one", async () => {
  const e = await evidenceFor(prop(0, 5), store(), { acl: allowAll, flowId: "flow-7" });
  assert.equal(e.flowId, "flow-7");
  const none = await evidenceFor(prop(0, 5), store());
  assert.equal(none.flowId, undefined);
});

test("intact, changed and gone are three answers, and changed is not repaired", async () => {
  const s = store();
  const e = await evidenceFor(prop(0, 10), s);

  assert.deepEqual((await recheck(e, s)).verdict, "intact");

  // Same length, different bytes: the claim may still be true and the note no
  // longer points at what it was read from. Somebody has to see that.
  s.set("notes/a.md", "XXXXXXXXXX" + BODY.slice(10));
  const changed = await recheck(e, s);
  assert.equal(changed.verdict, "changed");
  assert.notEqual(changed.sha256, e.sha256);
  // The note was not rewritten. Re-hashing here would turn "unverifiable" into
  // "verified", which is the exact inversion this component exists to prevent.
  assert.equal(e.sha256, sha256Of(new TextEncoder().encode(BODY.slice(0, 10))));

  s.set("notes/a.md", "short");
  assert.equal((await recheck(e, s)).verdict, "gone");
});

test("a source that shrank below the cited range is gone, not changed", async () => {
  const s = store();
  const e = await evidenceFor(prop(150, 200), s);
  s.set("notes/a.md", BODY.slice(0, 100));
  const r = await recheck(e, s);
  assert.equal(r.verdict, "gone");
  assert.equal(r.sha256, null);
});

test("a reader that disagrees with itself is refused rather than reconciled", async () => {
  // sizeOf says the range is inside; slice returns nothing. Two answers about
  // the same bytes that do not agree is not a store to mint evidence from.
  const broken: SourceReader = {
    async sizeOf() {
      return 1000;
    },
    async slice() {
      return new Uint8Array(0);
    },
    async idOf() {
      return "s";
    },
    async pathOf() {
      return "p";
    },
  };
  await assert.rejects(
    () => evidenceFor(prop(0, 10), broken),
    (err: unknown) => {
      assert.ok(err instanceof EvidenceRejected);
      assert.equal(err.detail.reason, "EMPTY_SLICE");
      return true;
    },
  );
});
