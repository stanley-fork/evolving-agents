/**
 * Who may read what, enforced below the model.
 *
 * ## Two rules, and the second is the one that gets forgotten
 *
 * **A scope may read itself and outward, never inward.** A flow reads its own
 * notes, its project's, its user's and the system's — that is the retrieval
 * order the whole store is built on. It does not read *another* flow's, or
 * another project's. Knowledge flows outward through promotion, which is
 * explicit, recorded and reversible; anything else would make promotion
 * pointless because everything would already be visible.
 *
 * **A source is not a note.** A note is a claim somebody decided to keep; the
 * bytes it came from are whatever happened to be in the flow — a transcript, a
 * key, a customer's file. So `memory_source` is checked separately and more
 * tightly than `memory_open`, and a scope that can read a promoted note cannot
 * necessarily read the artifact it was promoted from.
 *
 * That second rule is the one a system gets wrong quietly: a note is promoted
 * to system level for a good reason, and its evidence pointer quietly grants
 * everyone a path into the flow it came from.
 */
import type { Scope, ScopeType } from "../knowledge/schema.ts";
import type { Acl } from "../provenance/evidence.ts";
import { scopeDir } from "../knowledge/repository.ts";

/** Outward, in order. A scope may read itself and everything after it. */
export const OUTWARD: Readonly<Record<ScopeType, readonly ScopeType[]>> = Object.freeze({
  flow: ["flow", "project", "user", "system"],
  project: ["project", "user", "system"],
  user: ["user", "system"],
  system: ["system"],
});

export interface Reader {
  /** The scope doing the reading. */
  scope: Scope;
  /**
   * Whether this reader may follow evidence to the bytes behind a note.
   *
   * Off by default. A note is a claim somebody chose to keep; a source is
   * whatever was lying in the flow, which is a different thing to be allowed to
   * see and needs to be granted on purpose.
   */
  mayReadSources?: boolean;
}

export class Denied extends Error {
  constructor(what: string, why: string) {
    super(`ai-storage: ${what} — ${why}`);
    this.name = "Denied";
  }
}

/** May this reader see notes in that scope? */
export function mayReadScope(reader: Reader, target: Scope): boolean {
  if (!(OUTWARD[reader.scope.type] ?? []).includes(target.type)) return false;
  // Same level means the same instance. Two flows are both `flow`, and neither
  // reads the other — which is the case a level check alone gets wrong.
  if (target.type === reader.scope.type) return target.id === reader.scope.id;
  return true;
}

/**
 * The ACL a reader gets over the store's paths.
 *
 * Built from the reader rather than configured, so there is one place where
 * "what may this agent touch" is decided and it is the same place for the
 * provenance pipeline and for the tools.
 */
export function aclFor(reader: Reader, visible: readonly Scope[]): Acl {
  const allowedDirs = visible.filter((s) => mayReadScope(reader, s)).map(scopeDir);
  return {
    denies(artifact: string): string | null {
      if (!reader.mayReadSources)
        return (
          "this agent may read notes but not the artifacts behind them. A note is a claim " +
          "somebody kept; a source is whatever was in the flow."
        );
      if (artifact.includes("\0") || artifact.startsWith("/") || /(^|[\\/])\.\.([\\/]|$)/.test(artifact))
        return "the path leaves the store";
      const ok = allowedDirs.some((d) => artifact === d || artifact.startsWith(d + "/"));
      return ok
        ? null
        : `${reader.scope.type}:${reader.scope.id} may read ${allowedDirs.join(", ") || "nothing"}`;
    },
  };
}

/**
 * Filter what a reader is allowed to see, before anything is rendered.
 *
 * Filtering at the edge — hiding a note the model was already shown — is not
 * enforcement, it is redaction, and it leaks through counts, rankings and
 * "no results" answers. So the reader's scope list is narrowed first and the
 * tools never see the rest.
 */
export function visibleTo(reader: Reader, all: readonly Scope[]): Scope[] {
  return all.filter((s) => mayReadScope(reader, s));
}
