/**
 * The two things ai-storage has to beat, and the ceiling it is measured against.
 *
 * A candidate with no baseline is a demo. These are the baselines from the
 * specification, built so that the *only* difference between them and the
 * candidate is the retrieval mechanism — same model, same budget, same lanes,
 * same loop, same question, same corpus. Anything else that differs would show
 * up in the result and be read as a property of hierarchies.
 *
 * **A — flat.** One `MEMORY.md`: every note concatenated, handed over in one
 * `memory_all` call. This is what upstream memory is, and it is the baseline
 * because it is what people actually do. It fails in a way worth watching: past
 * a few hundred notes the file does not fit, and the failure is a refusal
 * rather than a wrong answer — which is the honest version of what happens
 * today, where the file is silently truncated to the last 300 bullets.
 *
 * **B — search.** No index, one tool: exact lexical search over the same notes.
 * Much stronger than A and the one that matters, because if hierarchy does not
 * beat grep then hierarchy is decoration. The predecessor project's flat result
 * is the reason to expect it might not.
 *
 * **C — the candidate.** `librarianTools`: index, open, find, source.
 *
 * The oracle runs against all three, which is the point of having it: it gives
 * each arm its own ceiling, so a model's score can be read as *how much of what
 * was available did it get* rather than as a bare number.
 */
import type { ToolImpl } from "../tools/registry.ts";
import { librarianTools } from "../agents/librarian.ts";
import { memoryFind, memoryOpen, memorySource, type MemoryView } from "../tools/memory.ts";
import { noteText, type Corpus } from "./corpus.ts";
import { renderNote } from "../tools/memory.ts";

export type Arm = "flat" | "search" | "storage";

export const ARMS: readonly Arm[] = ["flat", "search", "storage"];

/**
 * Baseline A: the whole store, in one call.
 *
 * The refusal is the interesting part. `memory_all` renders every note and asks
 * the budget for the tokens; past a few hundred notes the memory lane says no,
 * and the run ends as `context_limit` with the numbers that prove it — which is
 * the measurement, not a failure of the harness.
 */
export function memoryAll(corpus: Corpus): ToolImpl {
  return {
    lane: "memory",
    spec: {
      name: "memory_all",
      description:
        "Read the entire memory file. There is no index and no search — this is everything " +
        "that has been written down, in one piece.",
      parameters: { type: "object", additionalProperties: false, properties: {} },
    },
    async run() {
      return corpus.notes.map((n) => `## ${n.title}\n${n.claim}`).join("\n\n");
    },
  };
}

export function flatTools(corpus: Corpus): ToolImpl[] {
  return [memoryAll(corpus)];
}

export function searchTools(view: MemoryView): ToolImpl[] {
  return [memoryFind(view), memoryOpen(view), memorySource(view)];
}

/** How many tokens arm A would need if nothing refused it. */
export function flatCost(corpus: Corpus, count: (t: string) => number): number {
  return corpus.notes.reduce((n, x) => n + count(noteText(x)), 0);
}

/**
 * The system prompt each arm gets.
 *
 * Different, because the tools are different and a prompt describing tools that
 * do not exist would handicap the arm rather than describe it. What is held
 * constant is the *task*: find the answer, cite what you read, stop.
 */
export function systemFor(arm: Arm): string {
  const common = [
    "",
    "Stop as soon as you can answer. Call memory_done with the answer and the ids of the",
    "notes you used. Do not keep looking to be sure.",
    "",
    "If a tool returns an error, read it: it tells you what to do differently. Do not repeat",
    "a call that already failed the same way.",
  ];
  if (arm === "flat")
    return [
      "You answer questions from a project's memory. You have one tool: memory_all, which",
      "returns everything that has been written down. Read it and answer.",
      ...common,
    ].join("\n");
  if (arm === "search")
    return [
      "You answer questions from a project's memory. You have memory_find_exact, which",
      "matches whole words literally — no synonyms, no stemming — and memory_open, which",
      "reads one note. There is no index: search is how you find things.",
      "",
      "Every note you open costs you context you will need for the answer.",
      ...common,
    ].join("\n");
  return [
    "You find knowledge in a store you cannot see all of. You have four tools and no others.",
    "",
    "Navigate before you read. Start at memory_index('/'), descend into the directory whose",
    "name best matches the question, and only open a note once its name suggests it answers",
    "the question. memory_find_exact matches whole words literally — no synonyms.",
    "",
    "Read as little as necessary. Every note you open costs you context you will need for",
    "the answer. Two notes that answer the question beat six that surround it.",
    ...common,
  ].join("\n");
}

export function toolsFor(arm: Arm, corpus: Corpus, view: MemoryView): ToolImpl[] {
  if (arm === "flat") return flatTools(corpus);
  if (arm === "search") return searchTools(view);
  // The candidate is the real Librarian's capability list, imported rather than
  // restated: a benchmark arm that drifts from the thing it is measuring is a
  // benchmark of nothing.
  return librarianTools(view);
}

/** Re-export so a runner can render a note the same way every arm does. */
export { renderNote };
