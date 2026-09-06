/**
 * Promotion between scopes: explicit, justified, recorded, reversible.
 *
 * ```text
 *   FLOW → PROJECT → USER
 *                  → SYSTEM
 * ```
 *
 * ## The model recommends; policy decides
 *
 * A model can say *this constraint applies to every project using the harness*
 * — that is a judgement about meaning and it is the model's to make. Whether
 * the operation is permitted is not: `FLOW → SYSTEM` skips two levels, and a
 * claim that has been true in one flow for an afternoon is not a claim about
 * every project in the organisation. So the allowed transitions are a table in
 * code and the model's recommendation is an input to it.
 *
 * ## No promotion without a record
 *
 * Source scope, target scope, actor, time, and a reason in words. A promotion
 * with an empty reason is refused rather than recorded as "promoted": the whole
 * value of the record is that somebody can later ask *why is this at system
 * level* and get an answer instead of a timestamp.
 *
 * ## Reversible
 *
 * Demotion is the same operation with the scopes swapped and its own record. The
 * promoted note is a new note in the new scope with its own id — the original
 * stays where it was, marked, so demoting is removing the copy and not
 * reconstructing something that was destroyed.
 */
import type { KnowledgeNote, Scope, ScopeType } from "../knowledge/schema.ts";
import type { NoteRepository, WriteContext, PromotionRecord } from "../knowledge/repository.ts";
import type { FileStore } from "../backend/filesystem.ts";
import { scopeDir } from "../knowledge/repository.ts";

/** Where a note may go from where it is. Not derived from an ordering: stated. */
export const ALLOWED: Readonly<Record<ScopeType, readonly ScopeType[]>> = Object.freeze({
  flow: ["project"],
  project: ["user", "system"],
  user: [],
  system: [],
});

export class PromotionRefused extends Error {
  constructor(why: string) {
    super("ai-storage: " + why);
    this.name = "PromotionRefused";
  }
}

export function mayPromote(from: ScopeType, to: ScopeType): boolean {
  return (ALLOWED[from] ?? []).includes(to);
}

export interface PromoteRequest {
  note: KnowledgeNote;
  from: Scope;
  to: Scope;
  /** The model's words. Required. */
  reason: string;
  ctx: WriteContext;
}

/**
 * Copy a note up a level, and record why.
 *
 * The original is not moved. Two reasons: a flow's own record of what it
 * concluded should survive the conclusion being generalised, and demotion then
 * has something to fall back to that was never reconstructed from a diff.
 */
export async function promote(
  store: FileStore,
  repo: NoteRepository,
  req: PromoteRequest,
): Promise<{ record: PromotionRecord; note: KnowledgeNote }> {
  if (!mayPromote(req.from.type, req.to.type))
    throw new PromotionRefused(
      `${req.from.type} → ${req.to.type} is not an allowed promotion. From ${req.from.type} a ` +
        `note may go to: ${(ALLOWED[req.from.type] ?? []).join(", ") || "nowhere"}.`,
    );
  if (!req.reason.trim())
    throw new PromotionRefused(
      "a promotion with no reason is not a promotion, it is a copy. The record exists so " +
        "somebody can later ask why this is at this level and get an answer.",
    );

  const proposal = {
    title: req.note.title,
    type: req.note.type,
    claim: req.note.claim,
    keywords: req.note.keywords,
    source: {
      artifact: req.note.evidence[0]?.path ?? "",
      from: req.note.evidence[0]?.from ?? 0,
      to: req.note.evidence[0]?.to ?? 1,
    },
  };
  // The evidence travels unchanged: the promoted note is the same claim read
  // from the same bytes, and re-deriving a digest here would let a promotion
  // quietly re-verify something that had stopped being verifiable.
  const { note } = await repo.create(req.to, proposal, req.note.evidence, req.ctx);

  const record: PromotionRecord = {
    noteId: req.note.id,
    from: req.from.type,
    to: req.to.type,
    at: req.ctx.at,
    actor: req.ctx.actor,
    reason: req.reason.trim(),
    becomes: note.id,
  };
  await store.transact("note.promote", [promotionPath(req.from, req.note.id)], req.ctx.at, async () => {
    await store.writeJson(promotionPath(req.from, req.note.id), record);
  });
  return { record, note };
}

/**
 * Undo a promotion, leaving both records in place.
 *
 * The promoted copy is withdrawn rather than deleted. `withdrawn` is a state a
 * reader can see and ask about; a missing file is a store that forgot, and this
 * store does not forget.
 */
export async function demote(
  store: FileStore,
  repo: NoteRepository,
  record: PromotionRecord,
  ctx: WriteContext,
  reason: string,
): Promise<void> {
  if (!reason.trim()) throw new PromotionRefused("a demotion needs a reason too");
  const to: Scope = { type: record.to, id: scopeIdOf(record.to) };
  const promoted = await repo.get(to, record.becomes);
  if (!promoted) throw new PromotionRefused(`no promoted note ${record.becomes} in ${record.to}`);
  await repo.revise(to, record.becomes, { state: "withdrawn" }, ctx);
  await store.writeJson(`${scopeDir(to)}/promotions/${record.becomes}.demoted.json`, {
    ...record,
    demotedAt: ctx.at,
    demotedBy: ctx.actor,
    reason: reason.trim(),
  });
}

export function promotionPath(from: Scope, noteId: string): string {
  return `${scopeDir(from)}/promotions/${noteId}.json`;
}

/** Every promotion recorded out of a scope. */
export async function promotionsFrom(store: FileStore, from: Scope): Promise<PromotionRecord[]> {
  const out: PromotionRecord[] = [];
  for (const n of await store.list(`${scopeDir(from)}/promotions`)) {
    if (!n.endsWith(".json") || n.endsWith(".demoted.json")) continue;
    const r = await store.readJson<PromotionRecord>(`${scopeDir(from)}/promotions/${n}`);
    if (r) out.push(r);
  }
  return out.sort((a, b) => (a.at < b.at ? -1 : 1));
}

/**
 * The scope id for a level that has exactly one instance.
 *
 * `system` is a singleton and its directory has no id in it. For `user` and
 * `project` the caller has to supply the real scope; this exists so `demote`
 * can round-trip a system promotion without one.
 */
function scopeIdOf(type: ScopeType): string {
  return type === "system" ? "system" : "";
}
