/**
 * The four read tools, and the shapes they answer in.
 *
 * ## Answers are terse on purpose
 *
 * Every character a tool returns is charged to a lane a model with 8K has very
 * little of. So `memory_index` returns a directory listing and not a
 * description; `memory_find` returns ids and titles and not claims. The model
 * asks for the claim by opening the note, which is one more call and a decision
 * it made — rather than 2,000 tokens it did not ask for and cannot give back.
 *
 * ## Nothing here can write
 *
 * Not by convention: there is no write in this file. See `registry.ts`.
 */
import type { ScopeType } from "../knowledge/schema.ts";
import {
  DEFAULT_LIMITS,
  assertWithinBudget,
  renderNode,
  type IndexLimits,
  type IndexNode,
} from "../index/tree.ts";
import type { LexicalIndex } from "../search/lexical.ts";
import type { KnowledgeNote } from "../knowledge/schema.ts";
import type { ToolImpl } from "./registry.ts";

/**
 * What the tools read. An interface, so the benchmark can build a synthetic
 * store of 50,000 notes without a filesystem, and the real one can be the
 * filesystem, and neither knows which it is talking to.
 */
export interface MemoryView {
  /** The index node at a path, or null when there is none. */
  nodeAt(path: string): Promise<IndexNode | null>;
  /** A note by id, searched nearest-scope-first. */
  noteById(id: string): Promise<{ note: KnowledgeNote; scope: ScopeType } | null>;
  /** The lexical index, already built. */
  index(): Promise<LexicalIndex>;
  /** The bytes a note cited, for `memory_source`. */
  sourceSlice(note: KnowledgeNote, evidenceIndex: number): Promise<string | null>;
}

const str = (v: unknown, fallback = ""): string => (typeof v === "string" ? v : fallback);
const num = (v: unknown, fallback: number): number =>
  typeof v === "number" && Number.isFinite(v) ? v : fallback;

/**
 * Render a note for the model.
 *
 * The evidence is listed as addresses, not contents: the model gets to decide
 * whether reading the underlying bytes is worth the tokens, and most of the
 * time it is not — the claim is what it came for. This is the same discipline
 * the user interface runs on, where a citation is a button rather than an
 * expansion.
 */
export function renderNote(note: KnowledgeNote, scope: ScopeType): string {
  const lines = [
    `${note.id}  [${scope}]  ${note.state}`,
    `type: ${note.type}`,
    `title: ${note.title}`,
    `keywords: ${note.keywords.join(", ") || "(none)"}`,
    "",
    note.claim,
  ];
  if (note.evidence.length) {
    lines.push("", "evidence:");
    note.evidence.forEach((e, i) => lines.push(`  [${i}] ${e.path} bytes ${e.from}-${e.to}`));
  }
  if (note.relations.supersededBy)
    lines.push("", `superseded by ${note.relations.supersededBy}`);
  if (note.relations.supersedes.length)
    lines.push("", `supersedes ${note.relations.supersedes.join(", ")}`);
  return lines.join("\n");
}

export function memoryIndex(view: MemoryView, limits: IndexLimits = DEFAULT_LIMITS): ToolImpl {
  return {
    lane: "navigation",
    spec: {
      name: "memory_index",
      description:
        "List what is at a path in the knowledge index. Start at '/'. Names ending in '/' " +
        "are directories to descend into; the rest are notes you can open. This is how you " +
        "narrow down — it does not tell you what the project is about.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: { path: { type: "string", description: "Store path, e.g. '/architecture'" } },
      },
    },
    async run(args) {
      const path = normalisePath(str(args["path"], "/"));
      const node = await view.nodeAt(path);
      if (!node)
        return (
          `NO_SUCH_PATH: "${path}". Nothing is stored there. Go back to '/' and descend ` +
          `from a name you have actually seen.`
        );
      // Refuses rather than trims: a node over budget is a store that has not
      // been split, and showing half of it would let the model conclude a note
      // is absent when it is merely below the fold.
      assertWithinBudget(node, { describe: "n/a", count: () => 0 }, limits);
      return renderNode(node);
    },
  };
}

export function memoryOpen(view: MemoryView): ToolImpl {
  return {
    lane: "memory",
    spec: {
      name: "memory_open",
      description: "Read one note in full, by its id. Ids come from memory_index or memory_find.",
      parameters: {
        type: "object",
        additionalProperties: false,
        required: ["id"],
        properties: { id: { type: "string" } },
      },
    },
    async run(args) {
      const id = str(args["id"]).trim();
      if (!id) return "BAD_ARGUMENTS: memory_open needs an id.";
      const hit = await view.noteById(id);
      if (!hit)
        return `NO_SUCH_NOTE: "${id}". Ids look like kn_ followed by hex. Use memory_find.`;
      return renderNote(hit.note, hit.scope);
    },
  };
}

export function memoryFind(view: MemoryView): ToolImpl {
  return {
    lane: "navigation",
    spec: {
      name: "memory_find_exact",
      description:
        "Find notes containing exact words. Returns ids and titles only — open a note to " +
        "read it. Exact word match, no synonyms and no stemming, so use words that would " +
        "literally appear.",
      parameters: {
        type: "object",
        additionalProperties: false,
        required: ["query"],
        properties: {
          query: { type: "string" },
          limit: { type: "integer", minimum: 1, maximum: 20 },
        },
      },
    },
    async run(args) {
      const query = str(args["query"]).trim();
      if (!query) return "BAD_ARGUMENTS: memory_find_exact needs a query.";
      const ix = await view.index();
      const hits = ix.search(query, Math.min(20, Math.max(1, num(args["limit"], 8))));
      if (!hits.length)
        return (
          `NO_MATCHES for "${query}" across ${ix.size} notes. The match is on exact words. ` +
          `Try a word that would literally appear in the text.`
        );
      const lines: string[] = [];
      for (const h of hits) {
        const got = await view.noteById(h.id);
        // The counts go with each hit: "matched 2/3" is checkable, a score is
        // not, and a model choosing what to open should see why something ranked.
        lines.push(`${h.id}  matched ${h.matched}  ${got ? got.note.title : "(missing)"}`);
      }
      return lines.join("\n");
    },
  };
}

export function memorySource(view: MemoryView): ToolImpl {
  return {
    lane: "memory",
    spec: {
      name: "memory_source",
      description:
        "Read the original bytes a note was written from. Expensive — only when the note " +
        "itself is not enough and you need the wording of the source.",
      parameters: {
        type: "object",
        additionalProperties: false,
        required: ["id"],
        properties: { id: { type: "string" }, evidence: { type: "integer", minimum: 0 } },
      },
    },
    async run(args) {
      const id = str(args["id"]).trim();
      const hit = await view.noteById(id);
      if (!hit) return `NO_SUCH_NOTE: "${id}".`;
      const i = num(args["evidence"], 0);
      const text = await view.sourceSlice(hit.note, i);
      if (text === null)
        return (
          `SOURCE_UNAVAILABLE for ${id} evidence ${i}. The note records where it was read; ` +
          `those bytes are no longer there. That is not a reason to doubt the claim and it ` +
          `is a reason not to treat this as verified.`
        );
      return text;
    },
  };
}

export function memoryDone(): ToolImpl {
  return {
    lane: "generation",
    spec: {
      name: "memory_done",
      description:
        "Stop. Call this the moment you have enough to answer, with the note ids you used " +
        "and the answer itself. Reading more after this point costs the answer nothing.",
      parameters: {
        type: "object",
        additionalProperties: false,
        required: ["answer", "cites"],
        properties: {
          answer: { type: "string" },
          cites: { type: "array", items: { type: "string" } },
        },
      },
    },
    async run() {
      // The loop reads the arguments; the result is a stop signal, not content.
      return "DONE";
    },
  };
}

function normalisePath(p: string): string {
  const parts = p.split("/").filter((s) => s && s !== ".");
  // A traversal is not resolved, it is discarded: '/a/../b' means the model is
  // guessing, and letting it work would teach it that guessing works.
  if (parts.some((s) => s === "..")) return " invalid";
  return "/" + parts.join("/");
}
