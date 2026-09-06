/**
 * Notes on disk: create, revise, supersede, promote, and never destroy.
 *
 * ## Every change is a revision
 *
 * ```text
 *   kn_123 rev1  →  kn_123 rev2  →  kn_123 rev3
 * ```
 *
 * The current state is one file; every prior state is another. That is what
 * makes promotion reversible, and it is why nothing here overwrites: a store
 * that forgets cannot answer *what did we think last week and why did we stop
 * thinking it*, which is most of what a project's memory is for.
 *
 * ## Contradictions are kept
 *
 * A note that contradicts another does not overwrite it. It supersedes it, and
 * the superseded note stays readable with a pointer forward. Two notes that
 * genuinely disagree and neither supersedes the other stay as a recorded
 * conflict. Silently resolving either case would be the store deciding meaning,
 * which is the model's job — and neither the model nor the store gets to make
 * an old claim disappear.
 *
 * ## Ids are minted here
 *
 * From the content, so the same note proposed twice gets the same id and a
 * re-run after a crash is idempotent rather than duplicative. Not from a
 * counter, which would make two stores built from the same sources disagree
 * about every id and make the benchmark unrepeatable.
 */
import { createHash } from "node:crypto";
import type { FileStore } from "../backend/filesystem.ts";
import {
  noteFrom,
  type KnowledgeNote,
  type KnowledgeProposal,
  type Minted,
  type NoteState,
  type Scope,
  type ScopeType,
  type EvidenceReference,
} from "./schema.ts";

/** Where a scope's files live. The only place this layout is written down. */
export function scopeDir(scope: Scope): string {
  switch (scope.type) {
    case "system":
      return "system";
    case "user":
      return `users/${scope.id}`;
    case "project":
      return `projects/${scope.id}`;
    case "flow":
      return `flows/${scope.id}`;
  }
}

export const notePath = (scope: Scope, id: string) => `${scopeDir(scope)}/notes/${id}.json`;
export const revisionPath = (scope: Scope, id: string, rev: string) =>
  `${scopeDir(scope)}/notes/${id}.revisions/${rev}.json`;

/**
 * A name a reader can use, and a hash that makes it unique.
 *
 * ```text
 *   kn_deployment-key-rotation-sigma_2dbb6e94
 *      └── from the title ──────────┘ └─ content ─┘
 * ```
 *
 * The first version was `kn_` plus 24 hex characters, and the benchmark found
 * out why that is wrong. An index whose entries are opaque ids cannot be
 * navigated: splitting a full directory by prefix produces meaningless buckets,
 * and the one-line hint beside each entry ends up doing all the work. A
 * navigator then has no way to choose between two hundred notes that all look
 * like `kn_2dbb6e94` except by opening them, which is the scan this component
 * exists to avoid.
 *
 * So the id carries the title's own words. It stays content-addressed — the
 * hash is over scope, type, title, claim and every citation — so proposing the
 * same claim from the same bytes twice still produces the same note and a
 * re-run after a crash is idempotent rather than duplicative. What changes is
 * that the store can be read.
 */
export function mintId(scope: Scope, proposal: KnowledgeProposal, evidence: EvidenceReference[]): string {
  const h = createHash("sha256");
  h.update(scope.type + "\0" + scope.id + "\0");
  h.update(proposal.type + "\0" + proposal.title + "\0" + proposal.claim + "\0");
  for (const e of [...evidence].sort((a, b) => (a.sha256 < b.sha256 ? -1 : 1)))
    h.update(e.sha256 + "\0" + e.from + "\0" + e.to + "\0");
  return `kn_${slugify(proposal.title)}_${h.digest("hex").slice(0, 8)}`;
}

/**
 * A title, as something that can be a filename and read as words.
 *
 * Truncated at a word boundary rather than mid-word: `deployment-key-rot` reads
 * as a typo and costs a navigator a moment of doubt, which is the one thing an
 * index entry must not do.
 */
export function slugify(title: string, max = 44): string {
  const words = title
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .split(/[\s-]+/)
    .filter(Boolean);
  const out: string[] = [];
  let len = 0;
  for (const w of words) {
    if (len && len + 1 + w.length > max) break;
    out.push(w);
    len += (len ? 1 : 0) + w.length;
  }
  return out.join("-") || "note";
}

export interface WriteContext {
  /** ISO 8601. Passed in so the repository has no clock of its own. */
  at: string;
  /** Who or what asked. Recorded on every change; there is no anonymous write. */
  actor: string;
}

export class NoteConflict extends Error {
  constructor(message: string) {
    super("ai-storage: " + message);
    this.name = "NoteConflict";
  }
}

export interface PromotionRecord {
  noteId: string;
  from: ScopeType;
  to: ScopeType;
  at: string;
  actor: string;
  /** Why, in the model's words. A promotion with no reason is refused. */
  reason: string;
  /** The id the note has in its new scope. */
  becomes: string;
}

export class NoteRepository {
  readonly #store: FileStore;

  constructor(store: FileStore) {
    this.#store = store;
  }

  async get(scope: Scope, id: string): Promise<KnowledgeNote | null> {
    return this.#store.readJson<KnowledgeNote>(notePath(scope, id));
  }

  async ids(scope: Scope): Promise<string[]> {
    const names = await this.#store.list(`${scopeDir(scope)}/notes`);
    return names.filter((n) => n.endsWith(".json")).map((n) => n.slice(0, -5));
  }

  async all(scope: Scope): Promise<KnowledgeNote[]> {
    const out: KnowledgeNote[] = [];
    for (const id of await this.ids(scope)) {
      const n = await this.get(scope, id);
      if (n) out.push(n);
    }
    return out;
  }

  /**
   * Persist a proposal as a note.
   *
   * Idempotent by id: creating the same note twice returns the existing one
   * untouched, so a crashed run that is re-run does not double the store. That
   * is the same property `nextGap` gives the archivist, one layer up.
   */
  async create(
    scope: Scope,
    proposal: KnowledgeProposal,
    evidence: EvidenceReference[],
    ctx: WriteContext,
  ): Promise<{ note: KnowledgeNote; created: boolean }> {
    const id = mintId(scope, proposal, evidence);
    const existing = await this.get(scope, id);
    if (existing) return { note: existing, created: false };

    const minted: Minted = { id, scope, evidence, at: ctx.at, revision: "rev1" };
    const note = noteFrom(proposal, minted);
    await this.#store.transact("note.create", [notePath(scope, id)], ctx.at, async () => {
      await this.#store.writeJson(notePath(scope, id), note);
      await this.#store.writeJson(revisionPath(scope, id, "rev1"), {
        ...note,
        changedBy: ctx.actor,
      });
    });
    return { note, created: true };
  }

  /** The next revision label. `rev1`, `rev2`, … — sortable by number, not string. */
  static nextRevision(current: string): string {
    const n = Number(/^rev(\d+)$/.exec(current)?.[1] ?? 0);
    return "rev" + (n + 1);
  }

  /**
   * Change a note, keeping what it used to say.
   *
   * The prior revision is written before the new current state, so a crash in
   * the middle leaves history intact and the journal saying what was being
   * attempted. Losing the past to a crash while changing the present is the one
   * ordering this could get wrong.
   */
  async revise(
    scope: Scope,
    id: string,
    change: Partial<Pick<KnowledgeNote, "claim" | "title" | "keywords" | "state" | "relations">>,
    ctx: WriteContext,
  ): Promise<KnowledgeNote> {
    const current = await this.get(scope, id);
    if (!current) throw new NoteConflict(`no note ${id} in ${scope.type}:${scope.id}`);
    const revision = NoteRepository.nextRevision(current.revision);
    const next: KnowledgeNote = {
      ...current,
      ...change,
      relations: change.relations ?? current.relations,
      updatedAt: ctx.at,
      revision,
    };
    await this.#store.transact(
      "note.revise",
      [notePath(scope, id), revisionPath(scope, id, revision)],
      ctx.at,
      async () => {
        await this.#store.writeJson(revisionPath(scope, id, current.revision), {
          ...current,
          changedBy: ctx.actor,
        });
        await this.#store.writeJson(notePath(scope, id), next);
      },
    );
    return next;
  }

  /**
   * Every state this note has been in, oldest first, including the current one.
   *
   * The current state lives in the note file and the earlier ones in
   * `.revisions/`, so a naive listing of that directory returns the chain
   * minus its last link — which reads as "this note was revised and the new
   * version is missing". Whatever a reader is asking history for, they are
   * asking about the whole chain.
   */
  async history(scope: Scope, id: string): Promise<KnowledgeNote[]> {
    const names = await this.#store.list(`${scopeDir(scope)}/notes/${id}.revisions`);
    const out: KnowledgeNote[] = [];
    for (const n of names.filter((x) => x.endsWith(".json"))) {
      const v = await this.#store.readJson<KnowledgeNote>(
        `${scopeDir(scope)}/notes/${id}.revisions/${n}`,
      );
      if (v) out.push(v);
    }
    const current = await this.get(scope, id);
    if (current && !out.some((v) => v.revision === current.revision)) out.push(current);
    return out.sort(
      (a, b) => Number(/rev(\d+)/.exec(a.revision)?.[1] ?? 0) - Number(/rev(\d+)/.exec(b.revision)?.[1] ?? 0),
    );
  }

  /** Put an earlier revision back as the current state. Itself a revision. */
  async restore(scope: Scope, id: string, revision: string, ctx: WriteContext): Promise<KnowledgeNote> {
    const past = await this.#store.readJson<KnowledgeNote>(revisionPath(scope, id, revision));
    if (!past) throw new NoteConflict(`no revision ${revision} of ${id}`);
    return this.revise(
      scope,
      id,
      { claim: past.claim, title: past.title, keywords: past.keywords, state: past.state },
      ctx,
    );
  }

  /**
   * One note replaces another, and the replaced one stays readable.
   *
   * Both directions are recorded: the old note learns what replaced it, the new
   * one learns what it replaced. A one-way link would leave a reader who found
   * the old note with no way to discover it had been superseded — which is the
   * failure mode of every wiki.
   */
  async supersede(scope: Scope, oldId: string, newId: string, ctx: WriteContext): Promise<void> {
    const older = await this.get(scope, oldId);
    const newer = await this.get(scope, newId);
    if (!older) throw new NoteConflict(`no note ${oldId} to supersede`);
    if (!newer) throw new NoteConflict(`no note ${newId} to supersede with`);
    if (oldId === newId) throw new NoteConflict("a note cannot supersede itself");
    await this.revise(
      scope,
      oldId,
      { state: "superseded" as NoteState, relations: { ...older.relations, supersededBy: newId } },
      ctx,
    );
    await this.revise(
      scope,
      newId,
      {
        relations: {
          ...newer.relations,
          supersedes: [...new Set([...newer.relations.supersedes, oldId])],
        },
      },
      ctx,
    );
  }

  /**
   * Record that two notes disagree, without resolving it.
   *
   * Both stay `active`. The store does not get to decide which of two
   * contradicting claims is right, and neither does the model — what it decided
   * is that they contradict, and that is what gets written down.
   */
  async recordConflict(scope: Scope, a: string, b: string, why: string, ctx: WriteContext): Promise<void> {
    const na = await this.get(scope, a);
    const nb = await this.get(scope, b);
    if (!na || !nb) throw new NoteConflict("both notes must exist to record a conflict between them");
    await this.#store.transact("note.conflict", [`${scopeDir(scope)}/conflicts`], ctx.at, async () => {
      await this.#store.writeJson(`${scopeDir(scope)}/conflicts/${a}--${b}.json`, {
        a,
        b,
        why,
        at: ctx.at,
        actor: ctx.actor,
      });
    });
    for (const [id, other] of [
      [a, b],
      [b, a],
    ] as const) {
      const n = (await this.get(scope, id))!;
      await this.revise(
        scope,
        id,
        { relations: { ...n.relations, related: [...new Set([...n.relations.related, other])] } },
        ctx,
      );
    }
  }

  async conflicts(scope: Scope): Promise<Array<{ a: string; b: string; why: string }>> {
    const out: Array<{ a: string; b: string; why: string }> = [];
    for (const n of await this.#store.list(`${scopeDir(scope)}/conflicts`)) {
      const c = await this.#store.readJson<{ a: string; b: string; why: string }>(
        `${scopeDir(scope)}/conflicts/${n}`,
      );
      if (c) out.push(c);
    }
    return out;
  }
}
