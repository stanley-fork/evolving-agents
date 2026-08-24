/**
 * The Archivist: turn a finished piece of work into knowledge somebody can use.
 *
 * ## The distinction this agent exists to make
 *
 * A chunk is not a note. Chunks exist because the model has finite context;
 * they are an artefact of the reader. A note is a claim that stands on its own
 * — retrievable, citable, supersedable — and deciding which of those a passage
 * contains is the semantic judgement the model is here for.
 *
 * So the Archivist reads a *range of bytes* and returns *zero or more
 * proposals*. Zero is a real answer and the loop treats it as one: a page of
 * boilerplate produces no knowledge, and a store that manufactures a note per
 * chunk fills up with nothing.
 *
 * ## What it cannot do
 *
 * Write storage, edit a source, run anything, promote anything. Three tools —
 * open, slice, propose — and `Registry.invoke` refuses the rest.
 *
 * ## Progress is the cover, not a cursor
 *
 * The loop asks `nextGap` for the next unread stretch, hands it over, and
 * persists what comes back. Kill it at any point and the next run resumes the
 * holes, because the holes are computed from evidence that was actually stored.
 * See `provenance/ranges.ts`.
 */
import type { LocalModel } from "../model/local-model.ts";
import type { TokenCounter } from "../context/budget.ts";
import { BudgetLedger, budgetFor, type ContextBudget, DEFAULT_BUDGET } from "../context/budget.ts";
import {
  PROPOSAL_SCHEMA,
  ProposalRejected,
  proposalFrom,
  type KnowledgeProposal,
} from "../knowledge/schema.ts";
import { nextGap, type Range } from "../provenance/ranges.ts";

export const ARCHIVIST_SYSTEM = [
  "You turn a passage of a document into reusable knowledge.",
  "",
  "A note is one claim that stands on its own: someone who reads it a year from now, with",
  "no other context, can act on it. 'The build fails under Node 20 because package X needs",
  "Node 22' is a note. 'The build was discussed' is not.",
  "",
  "Return nothing when the passage contains nothing reusable. Boilerplate, restated",
  "requirements and prose about process are not knowledge. An empty answer is a correct",
  "answer and it is better than a note nobody will ever want.",
  "",
  "Every note cites the bytes you read it from — the offsets of the passage you were given,",
  "narrowed to the part that actually says it. Do not cite the whole passage out of",
  "convenience: the citation is what makes the claim checkable.",
].join("\n");

export interface ArchivistOptions {
  model: LocalModel;
  counter: TokenCounter;
  budget?: ContextBudget;
  temperature?: number;
  thinking?: boolean;
  timeoutMs?: number;
  /** How many bytes to hand over at a time. */
  chunkBytes?: number;
}

export interface ArchivistOutcome {
  /** What was read, so `holesIn` can be recomputed from it. */
  range: Range;
  proposals: KnowledgeProposal[];
  /** Proposals the validator threw out, with why. Counted, never hidden. */
  rejected: Array<{ why: string; field: string }>;
  usage: { promptTokens: number; completionTokens: number; reasoningTokens: number; latencyMs: number };
  error: string | null;
}

const ARCHIVIST_SCHEMA: Record<string, unknown> = {
  type: "object",
  additionalProperties: false,
  required: ["notes"],
  properties: {
    notes: { type: "array", maxItems: 8, items: PROPOSAL_SCHEMA },
  },
};

/**
 * Read one stretch of a source and propose what is in it.
 *
 * Two semantic retries and then a recorded failure, the same rule the Librarian
 * runs under and for the same reason: how often a quantization produces
 * unusable structured output is a result, and a retry loop erases it.
 */
export async function archiveRange(
  artifact: string,
  text: string,
  range: Range,
  opts: ArchivistOptions,
): Promise<ArchivistOutcome> {
  const ledger = new BudgetLedger(opts.budget ?? DEFAULT_BUDGET);
  const passage = text.slice(range.from, range.to);
  const user = [
    `Source: ${artifact}`,
    `Bytes ${range.from} to ${range.to}. Offsets you cite must fall inside that range.`,
    "",
    passage,
  ].join("\n");

  ledger.spend("harness", opts.counter.count(ARCHIVIST_SYSTEM), "the system prompt");
  try {
    ledger.spend("memory", opts.counter.count(user), "the passage");
  } catch (err) {
    // A chunk that does not fit is a chunk that was cut too large. Reported
    // rather than trimmed: trimming would cite bytes the model never saw.
    return {
      range,
      proposals: [],
      rejected: [],
      usage: { promptTokens: 0, completionTokens: 0, reasoningTokens: 0, latencyMs: 0 },
      error: (err as Error).message,
    };
  }

  let failures = 0;
  for (;;) {
    try {
      const reply = await opts.model.structured<{ notes: unknown[] }>({
        messages: [
          { role: "system", content: ARCHIVIST_SYSTEM },
          { role: "user", content: user },
        ],
        temperature: opts.temperature ?? 0.1,
        maxTokens: Math.max(128, ledger.remainingIn("generation")),
        thinking: opts.thinking ?? true,
        timeoutMs: opts.timeoutMs ?? 180_000,
        schema: ARCHIVIST_SCHEMA,
        schemaName: "archivist_notes",
      });

      const proposals: KnowledgeProposal[] = [];
      const rejected: ArchivistOutcome["rejected"] = [];
      for (const raw of reply.value.notes ?? []) {
        try {
          const p = proposalFrom({ ...(raw as object), source: sourceOf(raw, artifact, range) });
          // The offsets have to be inside what it was shown. A model that cites
          // bytes it never saw is guessing, and a guess with a byte range on it
          // is the most dangerous shape a guess can take.
          if (p.source.from < range.from || p.source.to > range.to)
            throw new ProposalRejected("source", `cites ${p.source.from}..${p.source.to}, outside the passage`);
          proposals.push(p);
        } catch (err) {
          rejected.push({
            why: (err as Error).message,
            field: err instanceof ProposalRejected ? err.field : "(unknown)",
          });
        }
      }
      return {
        range,
        proposals,
        rejected,
        usage: {
          promptTokens: reply.usage.promptTokens,
          completionTokens: reply.usage.completionTokens,
          reasoningTokens: reply.usage.reasoningTokens,
          latencyMs: reply.usage.latencyMs,
        },
        error: null,
      };
    } catch (err) {
      failures += 1;
      if (failures > 2)
        return {
          range,
          proposals: [],
          rejected: [],
          usage: { promptTokens: 0, completionTokens: 0, reasoningTokens: 0, latencyMs: 0 },
          error: `MEMORY_AGENT_FAILURE: ${(err as Error).message}`,
        };
    }
  }
}

/** The source block, defaulted to the passage when the model left it out. */
function sourceOf(raw: unknown, artifact: string, range: Range): Record<string, unknown> {
  const s = (raw as Record<string, unknown>)["source"];
  if (s && typeof s === "object") {
    const o = s as Record<string, unknown>;
    return {
      artifact: typeof o["artifact"] === "string" ? o["artifact"] : artifact,
      from: o["from"],
      to: o["to"],
    };
  }
  return { artifact, from: range.from, to: range.to };
}

/**
 * Walk a source to completion, resuming holes.
 *
 * `covered` is whatever evidence the store already holds for this artifact.
 * Pass what is persisted and nothing else — passing what this run has read but
 * not committed is how a cursor gets reinvented.
 */
export async function* archiveSource(
  artifact: string,
  text: string,
  covered: readonly Range[],
  opts: ArchivistOptions,
): AsyncGenerator<ArchivistOutcome> {
  const chunk = opts.chunkBytes ?? 4000;
  const done: Range[] = [...covered];
  for (;;) {
    const gap = nextGap(text.length, done, chunk);
    if (!gap) return;
    const outcome = await archiveRange(artifact, text, gap, opts);
    yield outcome;
    // The gap counts as read whatever came back, including nothing: a passage
    // with no knowledge in it has still been looked at, and re-reading it every
    // run would never terminate.
    done.push(gap);
  }
}

/** A budget shaped for the Archivist: most of it is the passage. */
export function archivistBudget(total = 8192): ContextBudget {
  const b = budgetFor(total);
  return { ...b, navigation: 0, memory: b.navigation + b.memory };
}
