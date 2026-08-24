/**
 * The wall between what a model may say and what gets stored.
 *
 * Most of these are schema-valid replies that are still wrong, because that is
 * the case constrained decoding does not cover. Structured output solves
 * syntax; it produces a backwards byte range as happily as a forwards one.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  LIMITS,
  NOTE_TYPES,
  PROPOSAL_SCHEMA,
  ProposalRejected,
  SCOPE_ORDER,
  noteFrom,
  proposalFrom,
  type KnowledgeProposal,
} from "../src/knowledge/schema.ts";

const good = (over: Record<string, unknown> = {}) => ({
  title: "Project ACLs use group:web-project-*",
  type: "constraint",
  claim: "Every project ACL in this repository is written as group:web-project-<id>.",
  keywords: ["acl", "group", "project"],
  source: { artifact: "notes/acl.md", from: 100, to: 420 },
  ...over,
});

const minted = (over: Record<string, unknown> = {}) => ({
  id: "kn_0001",
  scope: { type: "project" as const, id: "ai-os" },
  evidence: [
    {
      sourceId: "src_1",
      path: "sources/notes/acl.md",
      from: 100,
      to: 420,
      sha256: "a".repeat(64),
    },
  ],
  at: "2026-08-24T00:00:00.000Z",
  revision: "rev1",
  ...over,
});

test("a well-formed proposal survives", () => {
  const p = proposalFrom(good());
  assert.equal(p.type, "constraint");
  assert.deepEqual(p.source, { artifact: "notes/acl.md", from: 100, to: 420 });
});

test("a range that runs backwards is refused, and so is one with no width", () => {
  assert.throws(() => proposalFrom(good({ source: { artifact: "a.md", from: 400, to: 100 } })),
    /source\.to/);
  assert.throws(() => proposalFrom(good({ source: { artifact: "a.md", from: 100, to: 100 } })),
    /points at nothing/);
});

test("a path that climbs out of the store is refused — a path is a capability", () => {
  for (const artifact of ["../secrets.md", "a/../../b.md", "/etc/passwd", "..", "x/.."]) {
    assert.throws(
      () => proposalFrom(good({ source: { artifact, from: 0, to: 10 } })),
      ProposalRejected,
      `"${artifact}" must be refused`,
    );
  }
  // A dot inside a name is not a climb.
  assert.doesNotThrow(() =>
    proposalFrom(good({ source: { artifact: "notes/v1.2.3/report.md", from: 0, to: 10 } })),
  );
});

test("a type outside the vocabulary is refused rather than coerced", () => {
  assert.throws(() => proposalFrom(good({ type: "insight" })), /not one of/);
  for (const t of NOTE_TYPES) assert.doesNotThrow(() => proposalFrom(good({ type: t })));
});

test("an over-long title or claim is refused rather than trimmed", () => {
  // Trimming would silently change what the note says it read.
  assert.throws(() => proposalFrom(good({ title: "x".repeat(LIMITS.titleMax + 1) })), /longer than/);
  assert.throws(() => proposalFrom(good({ claim: "x".repeat(LIMITS.claimMax + 1) })), /longer than/);
});

test("repeated keywords are de-duplicated, because repetition is not a false claim", () => {
  const p = proposalFrom(good({ keywords: ["acl", "ACL", "Acl", "group"] }));
  assert.deepEqual(p.keywords, ["acl", "group"]);
});

test("too many keywords is refused, not silently cut to twelve", () => {
  const many = Array.from({ length: LIMITS.keywordsMax + 1 }, (_, i) => "k" + i);
  assert.throws(() => proposalFrom(good({ keywords: many })), /more than/);
});

test("a non-integer or negative offset is refused", () => {
  assert.throws(() => proposalFrom(good({ source: { artifact: "a.md", from: 1.5, to: 10 } })),
    /not an integer/);
  assert.throws(() => proposalFrom(good({ source: { artifact: "a.md", from: -1, to: 10 } })),
    /negative/);
  assert.throws(() => proposalFrom(good({ source: { artifact: "a.md", from: 0, to: NaN } })),
    /not an integer/);
});

test("the proposal type has nowhere to put an id, a hash or a scope", () => {
  // The wall is structural, not a sentence in a prompt. Extra keys in the reply
  // are ignored because `proposalFrom` reads only what it names, so a model
  // that invents an id cannot get one into the store.
  const p = proposalFrom(
    good({ id: "kn_evil", sha256: "b".repeat(64), scope: { type: "system", id: "x" } }),
  );
  assert.deepEqual(Object.keys(p).sort(), ["claim", "keywords", "source", "title", "type"]);
});

test("the schema handed to the model matches the limits the validator enforces", () => {
  // Two copies of the same numbers is how a schema and a validator drift apart
  // until a reply is accepted by one and rejected by the other every time.
  const props = PROPOSAL_SCHEMA["properties"] as Record<string, Record<string, unknown>>;
  assert.equal(props["title"]!["maxLength"], LIMITS.titleMax);
  assert.equal(props["claim"]!["maxLength"], LIMITS.claimMax);
  assert.equal(props["keywords"]!["maxItems"], LIMITS.keywordsMax);
  assert.deepEqual(props["type"]!["enum"], [...NOTE_TYPES]);
  assert.equal(PROPOSAL_SCHEMA["additionalProperties"], false);
});

test("a note cannot be built without evidence", () => {
  const p = proposalFrom(good()) as KnowledgeProposal;
  assert.throws(() => noteFrom(p, minted({ evidence: [] })), /not a note, it is an assertion/);
});

test("a note cannot be built with a digest that is not a digest", () => {
  const p = proposalFrom(good());
  const bad = minted({
    evidence: [{ sourceId: "s", path: "p", from: 0, to: 10, sha256: "not-a-hash" }],
  });
  assert.throws(() => noteFrom(p, bad), /not a sha256/);
});

test("a built note carries the minted mechanics and the model's meaning, separately", () => {
  const p = proposalFrom(good());
  const n = noteFrom(p, minted());
  assert.equal(n.version, 1);
  assert.equal(n.id, "kn_0001");
  assert.equal(n.state, "active");
  assert.equal(n.createdAt, n.updatedAt);
  assert.equal(n.title, p.title);
  assert.equal(n.claim, p.claim);
  assert.deepEqual(n.relations, { related: [], supersedes: [] });
  // The claim is the model's. Everything that could be forged is not.
  assert.equal(n.evidence[0]!.sha256, "a".repeat(64));
});

test("scopes are ordered nearest first", () => {
  assert.deepEqual([...SCOPE_ORDER], ["flow", "project", "user", "system"]);
});
