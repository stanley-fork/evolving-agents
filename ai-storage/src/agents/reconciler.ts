/**
 * The Reconciler and the Indexer: what a new note *is*, and where it goes.
 *
 * ## Reconciler — the distinction the whole store rests on
 *
 * > `same` is not `similar`.
 *
 * Two notes about deployment keys are similar. Two notes saying *the key for
 * Sigma rotates on Zeta-17* are the same. Merging the first pair loses a claim;
 * keeping the second pair fills the store with duplicates that will later
 * disagree with each other for no reason.
 *
 * Four answers, and the fourth is the one most systems do not have:
 *
 * - `new` — nothing in the store says this.
 * - `same` — an existing note already says it. Keep the old one.
 * - `supersedes` — this replaces an existing note, which stays readable.
 * - `conflict` — this contradicts an existing note and neither replaces the
 *   other. **Both stay active.** A store that resolves contradictions by
 *   overwriting has decided which of two claims is true, which is not a
 *   decision available to it. Recording the disagreement is.
 *
 * ## Indexer — placement is semantic, splitting is not
 *
 * *Where does this new note belong* is a judgement about meaning, so the model
 * makes it. *Where does this overfull directory divide* is mechanics, so code
 * does — see `index/tree.ts`. The Indexer returns a directory and code checks
 * it exists or creates it under a controlled rule; the model never edits an
 * index file.
 */
import type { LocalModel } from "../model/local-model.ts";
import type { KnowledgeNote, KnowledgeProposal } from "../knowledge/schema.ts";
import type { IndexNode } from "../index/tree.ts";
import { renderNode } from "../index/tree.ts";

export type ReconcileResult =
  | { action: "new" }
  | { action: "same"; existing: string }
  | { action: "supersedes"; existing: string; reason: string }
  | { action: "conflict"; existing: string; explanation: string };

export const RECONCILER_SYSTEM = [
  "You compare a proposed note against notes already in the store, and say how they relate.",
  "",
  "same — an existing note already makes this claim. Different wording is still the same",
  "claim. Being about the same topic is NOT the same claim.",
  "",
  "supersedes — this replaces an existing note: same subject, and this one is later, more",
  "precise, or corrects it. The old note is kept and marked.",
  "",
  "conflict — this contradicts an existing note and you cannot tell which is right. Say so.",
  "Both are kept. Never pick a winner to tidy the store up.",
  "",
  "new — nothing in the store says this.",
  "",
  "When in doubt between same and new, answer new. A duplicate is cheap; a lost claim is not.",
].join("\n");

const RECONCILE_SCHEMA: Record<string, unknown> = {
  type: "object",
  additionalProperties: false,
  required: ["action"],
  properties: {
    action: { type: "string", enum: ["new", "same", "supersedes", "conflict"] },
    existing: { type: "string" },
    reason: { type: "string", maxLength: 400 },
  },
};

export interface ReconcileOptions {
  model: LocalModel;
  temperature?: number;
  thinking?: boolean;
  timeoutMs?: number;
  maxTokens?: number;
}

export async function reconcile(
  proposal: KnowledgeProposal,
  candidates: readonly KnowledgeNote[],
  opts: ReconcileOptions,
): Promise<ReconcileResult> {
  // Nothing to compare against is not a question worth a model call.
  if (!candidates.length) return { action: "new" };

  const user = [
    "PROPOSED",
    `title: ${proposal.title}`,
    `claim: ${proposal.claim}`,
    "",
    "EXISTING",
    ...candidates.map((c) => `${c.id}\n  title: ${c.title}\n  claim: ${c.claim}`),
  ].join("\n");

  const reply = await opts.model.structured<{ action: string; existing?: string; reason?: string }>({
    messages: [
      { role: "system", content: RECONCILER_SYSTEM },
      { role: "user", content: user },
    ],
    temperature: opts.temperature ?? 0.1,
    maxTokens: opts.maxTokens ?? 400,
    thinking: opts.thinking ?? true,
    timeoutMs: opts.timeoutMs ?? 120_000,
    schema: RECONCILE_SCHEMA,
    schemaName: "reconcile",
  });

  const known = new Set(candidates.map((c) => c.id));
  const v = reply.value;
  /**
   * An id the model invented is not an id.
   *
   * Falling back to `new` rather than to the nearest candidate: guessing which
   * note it meant would let a hallucinated reference supersede a real note.
   * `new` costs a duplicate; the alternative costs a claim.
   */
  if (v.action === "same" && v.existing && known.has(v.existing))
    return { action: "same", existing: v.existing };
  if (v.action === "supersedes" && v.existing && known.has(v.existing))
    return { action: "supersedes", existing: v.existing, reason: v.reason ?? "" };
  if (v.action === "conflict" && v.existing && known.has(v.existing))
    return { action: "conflict", existing: v.existing, explanation: v.reason ?? "" };
  return { action: "new" };
}

// ---- the Indexer -----------------------------------------------------------

export interface IndexPlacement {
  directory: string;
  aliases: string[];
  related: string[];
}

export const INDEXER_SYSTEM = [
  "You decide which directory a new note belongs in, so that somebody looking for it later",
  "would look there first.",
  "",
  "Choose from the directories you are shown. Answer with one path, exactly as it appears.",
  "If none of them fits, answer with the parent you would create it under and a single new",
  "name — one level, never a tree.",
  "",
  "Where the note would be *found*, not where it was produced. A note about a deployment",
  "key belongs under deployment even if it came out of an experiment.",
].join("\n");

const PLACEMENT_SCHEMA: Record<string, unknown> = {
  type: "object",
  additionalProperties: false,
  required: ["directory"],
  properties: {
    directory: { type: "string" },
    aliases: { type: "array", maxItems: 4, items: { type: "string" } },
    related: { type: "array", maxItems: 6, items: { type: "string" } },
  },
};

export async function placeNote(
  proposal: KnowledgeProposal,
  choices: readonly IndexNode[],
  opts: ReconcileOptions,
): Promise<IndexPlacement> {
  const user = [
    `title: ${proposal.title}`,
    `keywords: ${proposal.keywords.join(", ")}`,
    `claim: ${proposal.claim.slice(0, 600)}`,
    "",
    "DIRECTORIES",
    ...choices.map((n) => renderNode({ path: n.path, entries: n.entries.filter((e) => e.kind === "dir") })),
  ].join("\n");

  const reply = await opts.model.structured<{
    directory: string;
    aliases?: string[];
    related?: string[];
  }>({
    messages: [
      { role: "system", content: INDEXER_SYSTEM },
      { role: "user", content: user },
    ],
    temperature: opts.temperature ?? 0,
    maxTokens: opts.maxTokens ?? 200,
    thinking: opts.thinking ?? false,
    timeoutMs: opts.timeoutMs ?? 120_000,
    schema: PLACEMENT_SCHEMA,
    schemaName: "placement",
  });

  return {
    directory: normalise(reply.value.directory),
    aliases: (reply.value.aliases ?? []).filter((s) => typeof s === "string").slice(0, 4),
    related: (reply.value.related ?? []).filter((s) => typeof s === "string").slice(0, 6),
  };
}

/**
 * A path the model returned, made safe.
 *
 * Not resolved — discarded. `/a/../../etc` means the model is guessing, and a
 * store that quietly resolves a guess into a working path teaches it that
 * guessing works. The caller then checks the result exists or creates exactly
 * one level.
 */
export function normalise(path: string): string {
  const parts = String(path).split("/").filter((s) => s && s !== ".");
  if (parts.some((s) => s === "..")) return "/";
  return "/" + parts.join("/");
}

/**
 * Whether a placement may be created, and where.
 *
 * One new level under an existing parent, and nothing else. A model that
 * answers with a four-deep path it invented would otherwise grow a tree nobody
 * navigates, one note per directory — which is the index failing in the shape
 * that is hardest to notice.
 */
export function placementIsAllowed(
  placement: IndexPlacement,
  existing: ReadonlySet<string>,
): { ok: true; create: string | null } | { ok: false; why: string } {
  const dir = placement.directory;
  if (existing.has(dir)) return { ok: true, create: null };
  const parts = dir.split("/").filter(Boolean);
  const parent = "/" + parts.slice(0, -1).join("/");
  if (parts.length && existing.has(parent === "/" ? "/" : parent))
    return { ok: true, create: dir };
  return {
    ok: false,
    why: `"${dir}" is not an existing directory and its parent does not exist either`,
  };
}
