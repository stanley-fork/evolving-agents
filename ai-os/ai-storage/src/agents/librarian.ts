/**
 * The Librarian: find the least knowledge that answers the question.
 *
 * This is the agent the whole component is a bet on. If a 27B model at 8K can
 * walk an index down to two or three notes and answer from them, the hypothesis
 * holds and everything else is worth building. If it cannot, no amount of
 * Archivist or Reconciler saves it.
 *
 * ## What it is allowed to do
 *
 * Four tools, none of which writes. No shell, no filesystem, no code
 * execution — and again, not by instruction: `LIBRARIAN_TOOLS` is a list with
 * four entries and `Registry.invoke` refuses anything else.
 *
 * ## The loop, and every way it ends
 *
 * ```text
 *   ask → tool calls → results → ask → … → memory_done
 * ```
 *
 * and it ends when:
 *
 * - the model calls `memory_done` — the only good ending;
 * - the step cap is reached — `STEP_CAP`;
 * - a lane runs out — `MEMORY_CONTEXT_LIMIT`, which the model saw as a result
 *   first and had a chance to react to;
 * - the model answers in prose without calling `memory_done` — accepted, with
 *   `viaProse` set, because refusing an answer that is there would measure
 *   instruction-following rather than navigation, and the two have to be
 *   separable in the results;
 * - two schema failures — `MEMORY_AGENT_FAILURE`.
 *
 * Every one of those is recorded distinctly. A benchmark that reports "failed"
 * cannot tell a model that got lost from a model that ran out of context from a
 * model that answered fine but ignored the protocol, and those three call for
 * three different fixes.
 *
 * ## What it never does
 *
 * Sample until something validates. Two semantic retries, then a recorded
 * failure — because how often a quantization produces a malformed call is one
 * of the numbers being measured, and a retry loop erases it.
 */
import { BudgetLedger, type ContextBudget, DEFAULT_BUDGET, type TokenCounter } from "../context/budget.ts";
import type { LocalModel, Message, ToolCall } from "../model/local-model.ts";
import { Registry, type ToolImpl } from "../tools/registry.ts";
import {
  memoryDone,
  memoryFind,
  memoryIndex,
  memoryOpen,
  memorySource,
  type MemoryView,
} from "../tools/memory.ts";

export const LIBRARIAN_SYSTEM = [
  "You find knowledge in a store you cannot see all of. You have four tools and no others.",
  "",
  "Navigate before you read. Start at memory_index('/'), descend into the directory whose",
  "name best matches the question, and only open a note once its name suggests it answers",
  "the question. memory_find_exact matches whole words literally — no synonyms.",
  "",
  "Read as little as necessary. Every note you open costs you context you will need for",
  "the answer. Two notes that answer the question beat six that surround it.",
  "",
  "Stop as soon as you can answer. Call memory_done with the answer and the ids of the",
  "notes you used. Do not keep looking to be sure.",
  "",
  "If a tool returns an error, read it: it tells you what to do differently. Do not repeat",
  "a call that already failed the same way.",
].join("\n");

export interface LibrarianOptions {
  model: LocalModel;
  view: MemoryView;
  counter: TokenCounter;
  budget?: ContextBudget;
  maxSteps?: number;
  temperature?: number;
  thinking?: boolean;
  timeoutMs?: number;
  /** Extra tools. The benchmark's baselines use this to swap navigation out. */
  tools?: readonly ToolImpl[];
  /** The system prompt. Each benchmark arm needs its own: a prompt describing
   *  tools an arm does not have would handicap it rather than describe it. */
  system?: string;
}

export type LibrarianEnding =
  | "done"
  | "prose"
  | "step_cap"
  | "context_limit"
  | "agent_failure"
  | "loop";

export interface LibrarianStep {
  n: number;
  calls: Array<{ name: string; args: Record<string, unknown>; ok: boolean; tokens: number }>;
  promptTokens: number;
  completionTokens: number;
  reasoningTokens: number;
  latencyMs: number;
}

export interface LibrarianResult {
  ending: LibrarianEnding;
  /** The answer, or null when there is none. */
  answer: string | null;
  /** Note ids the model said it used. Empty when it did not say. */
  cites: string[];
  /** Note ids it actually opened, which is not always the same list. */
  opened: string[];
  steps: LibrarianStep[];
  /** Everything charged, by lane. The benchmark's denominator. */
  spent: Record<string, number>;
  /** What went wrong, when something did. */
  error: string | null;
}

export function librarianTools(view: MemoryView, extra: readonly ToolImpl[] = []): ToolImpl[] {
  return [
    memoryIndex(view),
    memoryOpen(view),
    memoryFind(view),
    memorySource(view),
    memoryDone(),
    ...extra,
  ];
}

export async function runLibrarian(
  question: string,
  opts: LibrarianOptions,
): Promise<LibrarianResult> {
  const ledger = new BudgetLedger(opts.budget ?? DEFAULT_BUDGET);
  const registry = new Registry(opts.tools ? [...opts.tools, memoryDone()] : librarianTools(opts.view));
  const ctx = { ledger, counter: opts.counter };
  const maxSteps = opts.maxSteps ?? 12;

  const system = opts.system ?? LIBRARIAN_SYSTEM;
  const messages: Message[] = [
    { role: "system", content: system },
    { role: "user", content: question },
  ];
  ledger.spend("harness", opts.counter.count(system), "the system prompt");
  ledger.spend("task", opts.counter.count(question), "the question");

  const steps: LibrarianStep[] = [];
  const opened: string[] = [];
  let failures = 0;

  const out = (
    ending: LibrarianEnding,
    answer: string | null,
    cites: string[],
    error: string | null,
  ): LibrarianResult => ({
    ending,
    answer,
    cites,
    opened,
    steps,
    spent: ledger.snapshot(),
    error,
  });

  for (let n = 1; n <= maxSteps; n += 1) {
    let reply;
    try {
      reply = await opts.model.tools({
        messages,
        tools: registry.specs(),
        temperature: opts.temperature ?? 0,
        maxTokens: Math.max(64, ledger.remainingIn("generation")),
        thinking: opts.thinking ?? false,
        timeoutMs: opts.timeoutMs ?? 120_000,
      });
    } catch (err) {
      failures += 1;
      // Two semantic retries, then stop. Not "keep sampling until it parses".
      if (failures > 2)
        return out("agent_failure", null, [], `MEMORY_AGENT_FAILURE: ${(err as Error).message}`);
      messages.push({
        role: "user",
        content: `Your last reply could not be read: ${(err as Error).message}. Call one tool.`,
      });
      continue;
    }

    const step: LibrarianStep = {
      n,
      calls: [],
      promptTokens: reply.usage.promptTokens,
      completionTokens: reply.usage.completionTokens,
      reasoningTokens: reply.usage.reasoningTokens,
      latencyMs: reply.usage.latencyMs,
    };
    steps.push(step);

    if (!reply.toolCalls.length) {
      // It answered in prose. Accepted and marked, so the results can separate
      // "could not navigate" from "navigated and ignored the protocol".
      const text = reply.text.trim();
      if (text) return out("prose", text, [], null);
      messages.push({
        role: "user",
        content: "You returned nothing. Call memory_index('/') to start, or memory_done to stop.",
      });
      continue;
    }

    messages.push({ role: "assistant", content: reply.text, toolCalls: reply.toolCalls });

    for (const call of reply.toolCalls) {
      if (call.name === "memory_done") {
        const { answer, cites } = readDone(call);
        step.calls.push({ name: call.name, args: call.arguments, ok: true, tokens: 0 });
        return out("done", answer, cites, null);
      }
      const res = await registry.invoke(call.name, call.arguments, ctx);
      step.calls.push({ name: call.name, args: call.arguments, ok: res.ok, tokens: res.tokens });
      if (call.name === "memory_open" && res.ok && typeof call.arguments["id"] === "string")
        opened.push(call.arguments["id"] as string);
      messages.push({ role: "tool", content: res.text, toolCallId: call.id });

      /**
       * End when the lane that refused has nothing left at all.
       *
       * A refusal is a result the model can react to — ask for fewer notes, a
       * narrower listing — so one is not the end of the run. But when the lane
       * is at zero, no smaller request would succeed either, and continuing
       * just burns steps producing the same refusal. The first version checked
       * the *memory* lane whatever lane had actually refused, so a run that ran
       * out of navigation kept going until the step cap and was recorded as
       * lost rather than as out of context.
       */
      const refused = laneOf(res.text);
      if (refused && (ledger.remainingIn(refused.lane) <= 0 || refused.requested > ledger.budget[refused.lane]))
        return out("context_limit", null, [], res.text);
      if (res.text.startsWith("REPEATED_TOOL_LOOP") && step.calls.filter((c) => !c.ok).length >= 2)
        return out("loop", null, [], res.text);
    }
  }

  return out("step_cap", null, [], `STEP_CAP: ${maxSteps} steps without an answer`);
}

function readDone(call: ToolCall): { answer: string | null; cites: string[] } {
  const a = call.arguments["answer"];
  const c = call.arguments["cites"];
  return {
    answer: typeof a === "string" && a.trim() ? a.trim() : null,
    cites: Array.isArray(c) ? c.filter((x): x is string => typeof x === "string") : [],
  };
}

/**
 * What a MEMORY_CONTEXT_LIMIT result was about, or null.
 *
 * Both numbers matter, and the first version only read one. A lane at zero
 * cannot serve any request — but a request *larger than the lane's whole
 * capacity* cannot be served either, however empty the lane is, and that is the
 * flat baseline's entire story: a memory file of two hundred notes does not fit
 * an 8K window and never will. Reading only "is the lane empty" recorded that
 * as a model answering nothing rather than as a store that does not fit.
 */
function laneOf(
  text: string,
): { lane: "harness" | "task" | "navigation" | "memory" | "generation"; requested: number } | null {
  if (!text.includes("MEMORY_CONTEXT_LIMIT")) return null;
  try {
    const d = JSON.parse(text) as { lane?: string; requestedTokens?: number };
    const lane = d.lane;
    if (
      lane !== "harness" && lane !== "task" && lane !== "navigation" && lane !== "memory" &&
      lane !== "generation"
    )
      return null;
    return { lane, requested: typeof d.requestedTokens === "number" ? d.requestedTokens : 0 };
  } catch {
    return null;
  }
}
