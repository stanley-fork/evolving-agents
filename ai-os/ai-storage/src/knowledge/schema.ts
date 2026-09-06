/**
 * What a note is, what the model may propose, and the wall between the two.
 *
 * ## The rule this file enforces
 *
 * > The model decides meaning. Code decides mechanics.
 *
 * The model proposes a `KnowledgeProposal`: a title, a type, a claim, keywords,
 * and where in a source it read them. It never proposes an id, a hash, a
 * timestamp, a revision, a scope or a state — and the way that is enforced is
 * not a sentence in a prompt. It is that `KnowledgeProposal` has no field for
 * any of them, and `noteFrom` is the only way to make a `KnowledgeNote`.
 *
 * A prompt saying *do not invent ids* is a request. A type with no id field is
 * a wall.
 *
 * ## Why the validators are written out by hand
 *
 * A schema library would be shorter. It would also be a dependency whose
 * behaviour on the edge cases — a NaN offset, a `__proto__` key, a range where
 * `from` equals `to` — is somebody else's decision, and those edges are exactly
 * where a fabricated citation would get through. These validators are the
 * component's actual contract, so they are in the component.
 *
 * Structured output solves **syntax**. It does not solve **truth**. Everything
 * here runs *after* the model returned schema-valid JSON, and most of it exists
 * to catch schema-valid lies.
 *
 * Pure. No fetch, no clock, no DOM — see `noteFrom`'s `at` parameter.
 */

export type NoteType =
  | "fact"
  | "decision"
  | "constraint"
  | "procedure"
  | "failure"
  | "experiment"
  | "preference"
  | "observation";

export const NOTE_TYPES: readonly NoteType[] = [
  "fact",
  "decision",
  "constraint",
  "procedure",
  "failure",
  "experiment",
  "preference",
  "observation",
];

export type ScopeType = "flow" | "project" | "user" | "system";

/**
 * Retrieval order, nearest first.
 *
 * Local knowledge beats distant knowledge: a constraint this flow recorded
 * about this run outranks a system-wide generality, because the generality was
 * written without knowing about this run.
 */
export const SCOPE_ORDER: readonly ScopeType[] = ["flow", "project", "user", "system"];

export type NoteState = "active" | "superseded" | "withdrawn";

export interface Scope {
  type: ScopeType;
  id: string;
}

/** Where a claim was read, and proof that it is still there. */
export interface EvidenceReference {
  sourceId: string;
  path: string;
  /** Byte offset, inclusive. */
  from: number;
  /** Byte offset, exclusive. */
  to: number;
  /** Of the slice `[from, to)`, not of the whole file. */
  sha256: string;
  flowId?: string;
}

export interface KnowledgeNote {
  version: 1;
  id: string;
  scope: Scope;
  title: string;
  type: NoteType;
  claim: string;
  keywords: string[];
  evidence: EvidenceReference[];
  relations: {
    related: string[];
    supersedes: string[];
    supersededBy?: string;
  };
  state: NoteState;
  createdAt: string;
  updatedAt: string;
  revision: string;
}

/**
 * Everything the model is allowed to say, and nothing else.
 *
 * Note what is absent: `id`, `sha256`, `scope`, `state`, `createdAt`,
 * `revision`. Not "the model should not fill these in" — there is nowhere to
 * put them.
 */
export interface KnowledgeProposal {
  title: string;
  type: NoteType;
  claim: string;
  keywords: string[];
  source: {
    artifact: string;
    from: number;
    to: number;
  };
}

/** The JSON Schema handed to the model, generated from the limits below. */
export const LIMITS = {
  titleMax: 160,
  claimMax: 4000,
  keywordsMax: 12,
  keywordMax: 64,
} as const;

export const PROPOSAL_SCHEMA: Record<string, unknown> = {
  type: "object",
  additionalProperties: false,
  required: ["title", "type", "claim", "keywords", "source"],
  properties: {
    title: { type: "string", minLength: 1, maxLength: LIMITS.titleMax },
    type: { type: "string", enum: [...NOTE_TYPES] },
    claim: { type: "string", minLength: 1, maxLength: LIMITS.claimMax },
    keywords: {
      type: "array",
      maxItems: LIMITS.keywordsMax,
      items: { type: "string", minLength: 1, maxLength: LIMITS.keywordMax },
    },
    source: {
      type: "object",
      additionalProperties: false,
      required: ["artifact", "from", "to"],
      properties: {
        artifact: { type: "string", minLength: 1 },
        from: { type: "integer", minimum: 0 },
        to: { type: "integer", minimum: 1 },
      },
    },
  },
};

export class ProposalRejected extends Error {
  readonly field: string;
  constructor(field: string, why: string) {
    super(`ai-storage: proposal rejected at "${field}": ${why}`);
    this.name = "ProposalRejected";
    this.field = field;
  }
}

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

function str(v: unknown, field: string, max: number): string {
  if (typeof v !== "string") throw new ProposalRejected(field, "not a string");
  const s = v.trim();
  if (!s) throw new ProposalRejected(field, "empty");
  if (s.length > max) throw new ProposalRejected(field, `longer than ${max} characters`);
  return s;
}

function int(v: unknown, field: string): number {
  if (typeof v !== "number" || !Number.isInteger(v) || !Number.isFinite(v))
    throw new ProposalRejected(field, "not an integer");
  return v;
}

/**
 * Turn whatever came back into a proposal, or refuse.
 *
 * Every branch here is a schema-valid reply that is still wrong: a range that
 * runs backwards, a zero-width range, an artifact path that climbs out of the
 * store, a keyword list with the same word four times. Constrained decoding
 * produces all of these happily.
 */
export function proposalFrom(raw: unknown): KnowledgeProposal {
  if (!isObj(raw)) throw new ProposalRejected("(root)", "not an object");
  const title = str(raw["title"], "title", LIMITS.titleMax);
  const claim = str(raw["claim"], "claim", LIMITS.claimMax);

  const type = raw["type"];
  if (typeof type !== "string" || !NOTE_TYPES.includes(type as NoteType))
    throw new ProposalRejected("type", `not one of ${NOTE_TYPES.join(", ")}`);

  const rawKeywords = raw["keywords"];
  if (!Array.isArray(rawKeywords)) throw new ProposalRejected("keywords", "not an array");
  if (rawKeywords.length > LIMITS.keywordsMax)
    throw new ProposalRejected("keywords", `more than ${LIMITS.keywordsMax}`);
  const seen = new Set<string>();
  const keywords: string[] = [];
  for (const k of rawKeywords) {
    const s = str(k, "keywords[]", LIMITS.keywordMax).toLowerCase();
    // De-duplicated rather than rejected: a repeated keyword is a small model
    // being repetitive, not a false claim, and the repair loses nothing.
    if (seen.has(s)) continue;
    seen.add(s);
    keywords.push(s);
  }

  const source = raw["source"];
  if (!isObj(source)) throw new ProposalRejected("source", "not an object");
  const artifact = str(source["artifact"], "source.artifact", 1024);
  const from = int(source["from"], "source.from");
  const to = int(source["to"], "source.to");
  if (from < 0) throw new ProposalRejected("source.from", "negative");
  if (to <= from)
    throw new ProposalRejected(
      "source.to",
      `not after source.from (${from}..${to}) — a citation with no width points at nothing`,
    );
  if (artifact.includes("\0")) throw new ProposalRejected("source.artifact", "contains a NUL");
  if (artifact.startsWith("/") || /(^|[\\/])\.\.([\\/]|$)/.test(artifact))
    throw new ProposalRejected(
      "source.artifact",
      "absolute or climbing out of the store — a path is a capability, and the model does not get one",
    );

  return { title, type: type as NoteType, claim, keywords, source: { artifact, from, to } };
}

/** Everything the model does not supply, gathered where it can be seen. */
export interface Minted {
  id: string;
  scope: Scope;
  evidence: EvidenceReference[];
  /** ISO 8601. Passed in rather than read from a clock, so this file stays pure. */
  at: string;
  revision: string;
}

/**
 * The only way to build a note.
 *
 * The signature is the wall: a proposal on one side, minted mechanics on the
 * other, and no path that produces a `KnowledgeNote` from model output alone.
 *
 * Evidence is required and must be non-empty. A durable claim with no evidence
 * is the thing this whole component exists to refuse — it is the same rule the
 * user interface runs on, one layer down.
 */
export function noteFrom(proposal: KnowledgeProposal, minted: Minted): KnowledgeNote {
  if (!minted.evidence.length)
    throw new ProposalRejected(
      "evidence",
      "a durable note with no evidence is not a note, it is an assertion",
    );
  for (const e of minted.evidence) {
    if (!/^[0-9a-f]{64}$/.test(e.sha256))
      throw new ProposalRejected("evidence.sha256", "not a sha256 digest");
    if (!(e.to > e.from)) throw new ProposalRejected("evidence", "range has no width");
  }
  return {
    version: 1,
    id: minted.id,
    scope: minted.scope,
    title: proposal.title,
    type: proposal.type,
    claim: proposal.claim,
    keywords: proposal.keywords,
    evidence: minted.evidence,
    relations: { related: [], supersedes: [] },
    state: "active",
    createdAt: minted.at,
    updatedAt: minted.at,
    revision: minted.revision,
  };
}

/** Sort scopes nearest-first, for level-ordered recall. */
export function byScopeDistance(a: ScopeType, b: ScopeType): number {
  return SCOPE_ORDER.indexOf(a) - SCOPE_ORDER.indexOf(b);
}
