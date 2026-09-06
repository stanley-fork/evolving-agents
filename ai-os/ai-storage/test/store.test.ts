/**
 * The store on disk: atomic, journalled, and unable to forget.
 *
 * These write real files into a temporary directory, because the properties
 * being asserted — a rename is atomic, a crash leaves a pending journal entry,
 * a revision survives the change that replaced it — are properties of the
 * filesystem and a mock would only assert my beliefs about it.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { FileStore, PathEscapes, atomicWrite, resolveInside } from "../src/backend/filesystem.ts";
import { NoteRepository, mintId, scopeDir, slugify } from "../src/knowledge/repository.ts";
import type { KnowledgeProposal, Scope, EvidenceReference } from "../src/knowledge/schema.ts";
import { ALLOWED, PromotionRefused, demote, mayPromote, promote, promotionsFrom } from "../src/promotion/promote.ts";
import { aclFor, mayReadScope, visibleTo } from "../src/security/acl.ts";

const SCOPE: Scope = { type: "project", id: "demo" };
const CTX = { at: "2026-08-24T10:00:00.000Z", actor: "test" };

const proposal = (over: Partial<KnowledgeProposal> = {}): KnowledgeProposal => ({
  title: "Project ACLs use group:web-project-*",
  type: "constraint",
  claim: "Every project ACL is written as group:web-project-<id>.",
  keywords: ["acl", "group"],
  source: { artifact: "acl.md", from: 0, to: 40 },
  ...over,
});

const evidence = (sha = "a".repeat(64)): EvidenceReference[] => [
  { sourceId: "s1", path: "sources/acl.md", from: 0, to: 40, sha256: sha },
];

async function freshStore(): Promise<{ store: FileStore; dir: string }> {
  const dir = await mkdtemp(join(tmpdir(), "ai-storage-"));
  const store = new FileStore(dir);
  await store.init();
  return { store, dir };
}

test("a name never becomes a path by concatenation", () => {
  const root = "/srv/store";
  assert.equal(resolveInside(root, "notes/a.json"), "/srv/store/notes/a.json");
  for (const bad of ["../secrets", "/etc/passwd", "a/../../b", "", "."])
    assert.throws(() => resolveInside(root, bad), PathEscapes, `"${bad}" must be refused`);
});

test("a write is never seen half-finished", async () => {
  const { dir } = await freshStore();
  const path = join(dir, "deep", "nested", "file.json");
  await atomicWrite(path, '{"a":1}');
  assert.equal(await readFile(path, "utf8"), '{"a":1}');
  // Overwrite: the old bytes or the new ones, never a mix. The rename gives
  // that; what this asserts is that no temporary file is left behind.
  await atomicWrite(path, '{"a":2}');
  assert.equal(await readFile(path, "utf8"), '{"a":2}');
  const store = new FileStore(dir);
  assert.deepEqual(
    (await store.list("deep/nested")).filter((n) => n.endsWith(".tmp")),
    [],
    "a temporary file survived the write",
  );
  await rm(dir, { recursive: true, force: true });
});

test("the same bytes serialise the same way twice", async () => {
  const { store, dir } = await freshStore();
  await store.writeJson("a.json", { b: 1, a: { d: 4, c: 3 } });
  const first = await store.readText("a.json");
  await store.writeJson("a.json", { a: { c: 3, d: 4 }, b: 1 });
  assert.equal(await store.readText("a.json"), first, "key order must not depend on insertion");
  await rm(dir, { recursive: true, force: true });
});

test("a crash mid-operation leaves a pending journal entry", async () => {
  const { store, dir } = await freshStore();
  assert.deepEqual(await store.pending(), []);
  await assert.rejects(
    () =>
      store.transact("note.create", ["notes/x.json"], CTX.at, async () => {
        await store.writeJson("notes/x.json", { half: true });
        throw new Error("power cut");
      }),
    /power cut/,
  );
  const pending = await store.pending();
  assert.equal(pending.length, 1);
  assert.equal(pending[0]!.op, "note.create");
  assert.deepEqual(pending[0]!.paths, ["notes/x.json"]);
  // A clean operation leaves nothing pending, and pruning never removes one
  // that is.
  await store.transact("note.create", ["notes/y.json"], CTX.at, async () => {
    await store.writeJson("notes/y.json", { ok: true });
  });
  await store.pruneJournal();
  assert.equal((await store.pending()).length, 1, "pruning must not remove a pending entry");
  await rm(dir, { recursive: true, force: true });
});

test("an id carries the title's words and the content's hash", () => {
  const id = mintId(SCOPE, proposal(), evidence());
  assert.match(id, /^kn_project-acls-use-group-web-project_[0-9a-f]{8}$/);
  // Same content, same id — which is what makes a re-run after a crash
  // idempotent rather than duplicative.
  assert.equal(id, mintId(SCOPE, proposal(), evidence()));
  // Different claim, different id.
  assert.notEqual(id, mintId(SCOPE, proposal({ claim: "Something else." }), evidence()));
  // Different scope, different id: the same claim at flow and system level are
  // different notes with different lifetimes.
  assert.notEqual(id, mintId({ type: "system", id: "system" }, proposal(), evidence()));
  // Different evidence, different id.
  assert.notEqual(id, mintId(SCOPE, proposal(), evidence("b".repeat(64))));
});

test("slugify cuts at a word, never mid-word", () => {
  assert.equal(
    slugify("Deployment key rotation for project Sigma"),
    "deployment-key-rotation-for-project-sigma",
  );
  assert.equal(slugify("Deployment key rotation for project Sigma", 20), "deployment-key");
  assert.equal(slugify("A: B/C — D"), "a-b-c-d");
  assert.equal(slugify("…"), "note");
  assert.ok(!slugify("supercalifragilistic expialidocious antidisestablishment").endsWith("-"));
});

test("creating the same note twice creates one note", async () => {
  const { store, dir } = await freshStore();
  const repo = new NoteRepository(store);
  const a = await repo.create(SCOPE, proposal(), evidence(), CTX);
  const b = await repo.create(SCOPE, proposal(), evidence(), CTX);
  assert.equal(a.created, true);
  assert.equal(b.created, false);
  assert.equal(a.note.id, b.note.id);
  assert.equal((await repo.ids(SCOPE)).length, 1);
  await rm(dir, { recursive: true, force: true });
});

test("a revision keeps what the note used to say", async () => {
  const { store, dir } = await freshStore();
  const repo = new NoteRepository(store);
  const { note } = await repo.create(SCOPE, proposal(), evidence(), CTX);
  const next = await repo.revise(SCOPE, note.id, { claim: "Corrected." }, { ...CTX, at: "2026-08-24T11:00:00.000Z" });
  assert.equal(next.revision, "rev2");
  assert.equal(next.claim, "Corrected.");
  const history = await repo.history(SCOPE, note.id);
  assert.ok(history.length >= 2);
  assert.equal(history[0]!.revision, "rev1");
  assert.equal(history[0]!.claim, proposal().claim, "the earlier claim must still be readable");

  const restored = await repo.restore(SCOPE, note.id, "rev1", { ...CTX, at: "2026-08-24T12:00:00.000Z" });
  assert.equal(restored.claim, proposal().claim);
  assert.equal(restored.revision, "rev3", "a restore is itself a revision, not a rewind");
  await rm(dir, { recursive: true, force: true });
});

test("superseding points both ways and keeps the old note readable", async () => {
  const { store, dir } = await freshStore();
  const repo = new NoteRepository(store);
  const older = (await repo.create(SCOPE, proposal(), evidence(), CTX)).note;
  const newer = (await repo.create(SCOPE, proposal({ claim: "ACLs now use team:*." }), evidence("c".repeat(64)), CTX)).note;
  await repo.supersede(SCOPE, older.id, newer.id, CTX);

  const o = (await repo.get(SCOPE, older.id))!;
  const n = (await repo.get(SCOPE, newer.id))!;
  assert.equal(o.state, "superseded");
  assert.equal(o.relations.supersededBy, newer.id);
  assert.ok(n.relations.supersedes.includes(older.id), "a one-way link leaves a reader stranded");
  assert.equal(o.claim, proposal().claim, "the superseded claim is still there");
  await assert.rejects(() => repo.supersede(SCOPE, older.id, older.id, CTX), /cannot supersede itself/);
  await rm(dir, { recursive: true, force: true });
});

test("a contradiction is recorded and both notes stay active", async () => {
  const { store, dir } = await freshStore();
  const repo = new NoteRepository(store);
  const a = (await repo.create(SCOPE, proposal({ claim: "The gate tolerance is 1e-4." }), evidence(), CTX)).note;
  const b = (await repo.create(SCOPE, proposal({ claim: "The gate tolerance is 1e-6." }), evidence("d".repeat(64)), CTX)).note;
  await repo.recordConflict(SCOPE, a.id, b.id, "two runs, two tolerances, neither retracted", CTX);

  assert.equal((await repo.get(SCOPE, a.id))!.state, "active");
  assert.equal((await repo.get(SCOPE, b.id))!.state, "active");
  const conflicts = await repo.conflicts(SCOPE);
  assert.equal(conflicts.length, 1);
  assert.match(conflicts[0]!.why, /neither retracted/);
  await rm(dir, { recursive: true, force: true });
});

test("promotion follows the stated table and nothing else", () => {
  assert.deepEqual([...ALLOWED.flow], ["project"]);
  assert.ok(mayPromote("flow", "project"));
  assert.ok(mayPromote("project", "system"));
  // Two levels at once, and every backwards move.
  assert.ok(!mayPromote("flow", "system"));
  assert.ok(!mayPromote("system", "project"));
  assert.ok(!mayPromote("user", "system"));
});

test("a promotion with no reason is refused, and one with a reason is recorded", async () => {
  const { store, dir } = await freshStore();
  const repo = new NoteRepository(store);
  const from: Scope = { type: "flow", id: "flow-1" };
  const to: Scope = { type: "project", id: "demo" };
  const { note } = await repo.create(from, proposal(), evidence(), CTX);

  await assert.rejects(
    () => promote(store, repo, { note, from, to, reason: "   ", ctx: CTX }),
    PromotionRefused,
  );
  await assert.rejects(
    () => promote(store, repo, { note, from, to: { type: "system", id: "system" }, reason: "x", ctx: CTX }),
    /not an allowed promotion/,
  );

  const { record, note: copy } = await promote(store, repo, {
    note,
    from,
    to,
    reason: "this constraint applies to every project using the harness",
    ctx: CTX,
  });
  assert.equal(record.from, "flow");
  assert.equal(record.to, "project");
  assert.match(record.reason, /every project/);
  // The original stays where it was: a flow's own record survives its
  // conclusion being generalised.
  assert.ok(await repo.get(from, note.id));
  assert.deepEqual((await promotionsFrom(store, from)).map((r) => r.becomes), [copy.id]);
  // Evidence travels unchanged rather than being re-derived.
  assert.equal(copy.evidence[0]!.sha256, note.evidence[0]!.sha256);
  await rm(dir, { recursive: true, force: true });
});

test("demotion withdraws the copy rather than deleting it", async () => {
  const { store, dir } = await freshStore();
  const repo = new NoteRepository(store);
  const from: Scope = { type: "project", id: "demo" };
  const to: Scope = { type: "system", id: "system" };
  const { note } = await repo.create(from, proposal(), evidence(), CTX);
  const { record } = await promote(store, repo, { note, from, to, reason: "applies everywhere", ctx: CTX });

  await assert.rejects(() => demote(store, repo, record, CTX, ""), /needs a reason/);
  await demote(store, repo, record, { ...CTX, at: "2026-08-25T00:00:00.000Z" }, "it did not apply everywhere");

  const withdrawn = await repo.get(to, record.becomes);
  assert.ok(withdrawn, "a demoted note is withdrawn, not deleted — a missing file is a store that forgot");
  assert.equal(withdrawn.state, "withdrawn");
  await rm(dir, { recursive: true, force: true });
});

test("a scope reads outward, never inward, and never sideways", () => {
  const flowA = { scope: { type: "flow" as const, id: "a" } };
  assert.ok(mayReadScope(flowA, { type: "flow", id: "a" }));
  assert.ok(mayReadScope(flowA, { type: "project", id: "demo" }));
  assert.ok(mayReadScope(flowA, { type: "system", id: "system" }));
  // Another flow at the same level is somebody else's work.
  assert.ok(!mayReadScope(flowA, { type: "flow", id: "b" }));
  // And a project cannot read into a flow.
  assert.ok(!mayReadScope({ scope: { type: "project", id: "demo" } }, { type: "flow", id: "a" }));

  const all: Scope[] = [
    { type: "flow", id: "a" },
    { type: "flow", id: "b" },
    { type: "project", id: "demo" },
    { type: "system", id: "system" },
  ];
  assert.deepEqual(visibleTo(flowA, all).map((s) => `${s.type}:${s.id}`), [
    "flow:a",
    "project:demo",
    "system:system",
  ]);
});

test("reading a note does not grant reading the bytes behind it", () => {
  const scope: Scope = { type: "project", id: "demo" };
  const reader = { scope };
  const acl = aclFor(reader, [scope]);
  // Off by default, and the message says why rather than just "denied".
  assert.match(acl.denies("projects/demo/sources/a.md")!, /a claim somebody kept/);

  const withSources = aclFor({ scope, mayReadSources: true }, [scope]);
  assert.equal(withSources.denies("projects/demo/sources/a.md"), null);
  assert.match(withSources.denies("flows/other/sources/a.md")!, /may read/);
  assert.match(withSources.denies("../../etc/passwd")!, /leaves the store/);
});
