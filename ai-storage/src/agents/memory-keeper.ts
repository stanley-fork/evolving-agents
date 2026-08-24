/**
 * The MemoryKeeper: the one that coordinates and cannot touch anything.
 *
 * ```text
 *   completed flow
 *         │
 *         ▼
 *    MemoryKeeper
 *         │
 *         ├── Archivist   what does this passage say?
 *         ├── Reconciler  does the store already say it?
 *         └── Indexer     where would somebody look for it?
 * ```
 *
 * ## Why it has no capability of its own
 *
 * A coordinator that can also write is a coordinator that will, on the run
 * where a specialist is slow or returns something awkward. Then the guarantees
 * the specialists carry — evidence verified, duplicates reconciled, placement
 * checked — become optional, and nothing in the store records which notes went
 * the long way and which did not.
 *
 * So this file holds no `FileStore` write path of its own. It calls the
 * repository, which mints ids and verifies evidence, and every branch that
 * could skip a specialist is absent rather than guarded.
 *
 * ## Every outcome is recorded, including the ones that did nothing
 *
 * A passage that produced no notes, a proposal the validator threw out, a
 * duplicate, a conflict — all counted separately in the report. A run that says
 * "12 notes written" and nothing else cannot be told apart from a run that
 * dropped forty proposals on the floor.
 */
import type { LocalModel } from "../model/local-model.ts";
import type { TokenCounter } from "../context/budget.ts";
import type { Scope } from "../knowledge/schema.ts";
import { NoteRepository, type WriteContext } from "../knowledge/repository.ts";
import { evidenceFor, type Acl, type SourceReader } from "../provenance/evidence.ts";
import { LexicalIndex } from "../search/lexical.ts";
import { archiveSource, type ArchivistOptions } from "./archivist.ts";
import { reconcile, type ReconcileOptions } from "./reconciler.ts";
import type { Range } from "../provenance/ranges.ts";

export interface KeeperOptions {
  repository: NoteRepository;
  reader: SourceReader;
  scope: Scope;
  model: LocalModel;
  counter: TokenCounter;
  acl?: Acl;
  /** ISO 8601 supplied per call. This component owns no clock. */
  now: () => string;
  actor?: string;
  chunkBytes?: number;
  archivist?: Partial<ArchivistOptions>;
  reconciler?: Partial<ReconcileOptions>;
  /** How many existing notes the Reconciler is shown. Keeps its prompt bounded. */
  candidates?: number;
}

export interface KeeperReport {
  artifact: string;
  /** Byte ranges this run read. Add to the store's cover; do not store a cursor. */
  read: Range[];
  proposed: number;
  /** Rejected by the validator — a schema-valid reply that was not true. */
  rejectedByValidator: number;
  /** Rejected by the provenance pipeline — the bytes did not say that. */
  rejectedByEvidence: number;
  created: number;
  duplicates: number;
  superseded: number;
  conflicts: number;
  /** Passages that produced nothing. A real outcome, not a failure. */
  barren: number;
  errors: string[];
}

export async function keepSource(
  artifact: string,
  text: string,
  covered: readonly Range[],
  opts: KeeperOptions,
): Promise<KeeperReport> {
  const actor = opts.actor ?? "memory-keeper";
  const report: KeeperReport = {
    artifact,
    read: [],
    proposed: 0,
    rejectedByValidator: 0,
    rejectedByEvidence: 0,
    created: 0,
    duplicates: 0,
    superseded: 0,
    conflicts: 0,
    barren: 0,
    errors: [],
  };

  const archivistOpts: ArchivistOptions = {
    model: opts.model,
    counter: opts.counter,
    chunkBytes: opts.chunkBytes ?? 4000,
    ...opts.archivist,
  };
  const reconcilerOpts: ReconcileOptions = { model: opts.model, ...opts.reconciler };

  for await (const outcome of archiveSource(artifact, text, covered, archivistOpts)) {
    report.read.push(outcome.range);
    report.rejectedByValidator += outcome.rejected.length;
    if (outcome.error) report.errors.push(outcome.error);
    if (!outcome.proposals.length) {
      report.barren += 1;
      continue;
    }

    for (const proposal of outcome.proposals) {
      report.proposed += 1;
      const ctx: WriteContext = { at: opts.now(), actor };

      // Provenance before meaning. A proposal whose bytes do not say what it
      // claims never reaches the Reconciler, so a hallucinated citation cannot
      // supersede a real note on its way to being rejected.
      let evidence;
      try {
        evidence = [await evidenceFor(proposal, opts.reader, { acl: opts.acl })];
      } catch (err) {
        report.rejectedByEvidence += 1;
        report.errors.push((err as Error).message);
        continue;
      }

      const existing = await opts.repository.all(opts.scope);
      const candidates = shortlist(existing, proposal.title + " " + proposal.claim, opts.candidates ?? 5);
      let verdict;
      try {
        verdict = await reconcile(proposal, candidates, reconcilerOpts);
      } catch (err) {
        report.errors.push(`reconciler: ${(err as Error).message}`);
        verdict = { action: "new" as const };
      }

      if (verdict.action === "same") {
        report.duplicates += 1;
        continue;
      }

      const { note, created } = await opts.repository.create(opts.scope, proposal, evidence, ctx);
      if (created) report.created += 1;
      else report.duplicates += 1;

      if (verdict.action === "supersedes") {
        await opts.repository.supersede(opts.scope, verdict.existing, note.id, ctx);
        report.superseded += 1;
      } else if (verdict.action === "conflict") {
        await opts.repository.recordConflict(
          opts.scope,
          verdict.existing,
          note.id,
          verdict.explanation,
          ctx,
        );
        report.conflicts += 1;
      }
    }
  }

  return report;
}

/**
 * The few notes worth showing the Reconciler.
 *
 * Lexical, not semantic, and bounded: the Reconciler has a context too, and
 * handing it every note in the scope would make its prompt grow with the store
 * — which is the failure this whole component exists to avoid, reappearing
 * inside one of its own agents.
 */
export function shortlist(
  notes: readonly import("../knowledge/schema.ts").KnowledgeNote[],
  query: string,
  limit: number,
): import("../knowledge/schema.ts").KnowledgeNote[] {
  if (!notes.length) return [];
  const ix = LexicalIndex.of(notes);
  const byId = new Map(notes.map((n) => [n.id, n]));
  return ix
    .search(query, limit)
    .map((h) => byId.get(h.id))
    .filter((n): n is import("../knowledge/schema.ts").KnowledgeNote => Boolean(n));
}
