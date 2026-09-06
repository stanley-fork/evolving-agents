/**
 * Turning a proposal's citation into evidence, or refusing to.
 *
 * The model says *I read this claim at bytes 1100 to 2450 of `report.md`*. This
 * file is what decides whether that sentence is true. It is the pipeline from
 * the specification, and every step of it is a place a schema-valid proposal
 * dies:
 *
 * ```text
 *   proposal → source exists? → range inside it? → extract → sha256 → ACL → persist
 * ```
 *
 * ## The hash is of the slice
 *
 * Not of the file. A digest of the whole file says *this file has not changed*,
 * which is a much weaker statement and the wrong one: regenerate an artifact,
 * and every note that cited any part of it becomes unverifiable at once, even
 * the ones whose bytes are identical. A digest of `[from, to)` says *the bytes
 * this claim was read from still say what they said*, which is the claim the
 * note actually needs.
 *
 * ## The model never supplies a digest
 *
 * There is no parameter for one. If there were, a model could emit a plausible
 * 64 hex characters and the note would carry a hash of nothing. The digest is
 * computed here from bytes read here.
 *
 * ## Reading is a capability, not a path
 *
 * `SourceReader` is an interface, and the model never gets one. It proposes an
 * artifact *name*; something on this side resolves the name inside a store it
 * controls. A hallucinated `../../.ssh/id_rsa` fails at resolution because
 * resolution is not string concatenation.
 */
import { createHash } from "node:crypto";
import { assertRange, type Range } from "./ranges.ts";
import type { EvidenceReference, KnowledgeProposal } from "../knowledge/schema.ts";

/**
 * Bytes, addressed by a name the store understands.
 *
 * Implementations decide what a name means. The filesystem backend resolves it
 * under the scope's `sources/` directory and refuses anything that escapes;
 * a test backend keeps a map. Neither takes a path from a model.
 */
export interface SourceReader {
  /** Total bytes, or `null` when the store has no such source. */
  sizeOf(artifact: string): Promise<number | null>;
  /** The bytes of `[from, to)`. Throws if the range is outside the source. */
  slice(artifact: string, range: Range): Promise<Uint8Array>;
  /** A stable id for the source, for the evidence record. */
  idOf(artifact: string): Promise<string>;
  /** The path to record, which is the store's, never the model's. */
  pathOf(artifact: string): Promise<string>;
}

export type VerifyFailure =
  | { reason: "NO_SUCH_SOURCE"; artifact: string }
  | { reason: "RANGE_OUTSIDE_SOURCE"; artifact: string; range: Range; size: number }
  | { reason: "EMPTY_SLICE"; artifact: string; range: Range }
  | { reason: "FORBIDDEN"; artifact: string; by: string };

export class EvidenceRejected extends Error {
  readonly detail: VerifyFailure;
  constructor(detail: VerifyFailure) {
    super(`ai-storage: evidence rejected — ${detail.reason} for "${detail.artifact}"`);
    this.name = "EvidenceRejected";
    this.detail = detail;
  }
}

/** Whether an agent may read a source at all. Enforced below the model. */
export interface Acl {
  /** `null` when allowed; a reason when not. */
  denies(artifact: string): string | null;
}

export const allowAll: Acl = { denies: () => null };

export function sha256Of(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

/**
 * Verify one citation and mint the evidence for it.
 *
 * Deliberately does not take the note, the id, or anything else about where
 * this is going. It answers one question — *are these bytes there, and what do
 * they hash to* — and answering exactly one question is what makes it testable
 * against a store with no model anywhere near it.
 */
export async function evidenceFor(
  proposal: KnowledgeProposal,
  reader: SourceReader,
  opts: { acl?: Acl; flowId?: string } = {},
): Promise<EvidenceReference> {
  const acl = opts.acl ?? allowAll;
  const artifact = proposal.source.artifact;

  const denied = acl.denies(artifact);
  if (denied !== null)
    throw new EvidenceRejected({ reason: "FORBIDDEN", artifact, by: denied });

  const size = await reader.sizeOf(artifact);
  if (size === null) throw new EvidenceRejected({ reason: "NO_SUCH_SOURCE", artifact });

  const range = assertRange({ from: proposal.source.from, to: proposal.source.to });
  if (range.to > size)
    throw new EvidenceRejected({ reason: "RANGE_OUTSIDE_SOURCE", artifact, range, size });

  const bytes = await reader.slice(artifact, range);
  // A range that is inside the source and still produces nothing means the
  // reader and the size disagree. Refused rather than reconciled: a store whose
  // two answers about the same bytes differ is not a store to mint evidence
  // from.
  if (bytes.byteLength === 0)
    throw new EvidenceRejected({ reason: "EMPTY_SLICE", artifact, range });

  const out: EvidenceReference = {
    sourceId: await reader.idOf(artifact),
    path: await reader.pathOf(artifact),
    from: range.from,
    to: range.to,
    sha256: sha256Of(bytes),
  };
  if (opts.flowId) out.flowId = opts.flowId;
  return out;
}

/**
 * Re-check evidence against the store as it is now.
 *
 * The answer is one of three, and the third is the reason this function exists
 * as something separate from `evidenceFor`:
 *
 * - `intact` — the bytes still hash to what the note recorded.
 * - `changed` — they are there and they are different. The claim may still be
 *   true, but the note no longer points at what it was read from, and that is
 *   a fact somebody has to see.
 * - `gone` — the source or the range is no longer there.
 *
 * Nothing here repairs anything. A note whose evidence changed is not silently
 * re-hashed: re-hashing would turn *this claim is now unverifiable* into *this
 * claim is verified*, which is the exact inversion the component is built to
 * prevent.
 */
export async function recheck(
  evidence: EvidenceReference,
  reader: SourceReader,
): Promise<{ verdict: "intact" | "changed" | "gone"; sha256: string | null }> {
  const size = await reader.sizeOf(evidence.path);
  if (size === null || evidence.to > size) return { verdict: "gone", sha256: null };
  let bytes: Uint8Array;
  try {
    bytes = await reader.slice(evidence.path, { from: evidence.from, to: evidence.to });
  } catch {
    return { verdict: "gone", sha256: null };
  }
  if (bytes.byteLength === 0) return { verdict: "gone", sha256: null };
  const now = sha256Of(bytes);
  return { verdict: now === evidence.sha256 ? "intact" : "changed", sha256: now };
}
