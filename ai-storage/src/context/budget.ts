/**
 * The context budget, and the invariant the whole component rests on.
 *
 * ## Why this file is small and load-bearing
 *
 * ai-storage exists to answer one question: how much of a project can a model
 * with 8K of working context operate on? Every number that answers it is a
 * ratio whose denominator is *tokens the model actually had to see*. If
 * anything anywhere silently trims a read to make it fit, that denominator is
 * a fiction and the headline result is a fiction with it.
 *
 * So the rule is absolute:
 *
 * > **A read that does not fit is refused. It is never truncated.**
 *
 * A refusal is an event the harness can see, count, and act on — narrow the
 * query, open fewer notes, split the index node. A truncation is invisible, and
 * an invisible truncation is a model answering from half a note while the run
 * record says it read the whole one.
 *
 * ## The lanes
 *
 * ```text
 *   System / harness           1,500
 *   Current task                 600
 *   Navigation                 2,000
 *   Retrieved knowledge        2,300
 *   Reasoning + output         1,792
 *                            -------
 *                              8,192
 * ```
 *
 * Lanes rather than one pool, because the failure they prevent is the one that
 * looks like success: a Librarian that spends 6,000 tokens walking the index
 * and then has no room to read the note it found has navigated beautifully and
 * answered nothing.
 *
 * These five numbers are configuration, not physics. They are a starting point
 * to be moved by benchmark results, and `bench/context-budget` exists to move
 * them.
 *
 * Pure. No fetch, no clock, no DOM, and no tokenizer — see `TokenCounter`.
 */

export type Lane = "harness" | "task" | "navigation" | "memory" | "generation";

export interface ContextBudget {
  /** The effective context. Never the model's physical maximum. */
  total: number;
  harness: number;
  task: number;
  navigation: number;
  memory: number;
  generation: number;
}

/** The starting point from the specification, at the 8K effective context. */
export const DEFAULT_BUDGET: Readonly<ContextBudget> = Object.freeze({
  total: 8192,
  harness: 1500,
  task: 600,
  navigation: 2000,
  memory: 2300,
  generation: 1792,
});

const LANES: readonly Lane[] = ["harness", "task", "navigation", "memory", "generation"];

/**
 * A budget whose lanes do not fit inside its total is a bug, not a preference.
 *
 * Checked when the budget is built rather than when a lane overflows, because
 * the second one only shows up on the runs that were going to be interesting.
 */
export function assertBudget(b: ContextBudget): ContextBudget {
  for (const lane of LANES)
    if (!Number.isInteger(b[lane]) || b[lane] < 0)
      throw new Error(`ai-storage: budget lane "${lane}" must be a non-negative integer`);
  if (!Number.isInteger(b.total) || b.total <= 0)
    throw new Error("ai-storage: budget total must be a positive integer");
  const sum = LANES.reduce((n, lane) => n + b[lane], 0);
  if (sum > b.total)
    throw new Error(
      `ai-storage: the budget lanes sum to ${sum}, which is more than the ${b.total} ` +
        `available. A budget that does not fit is a budget that will truncate something.`,
    );
  return b;
}

/** Scale the default lanes to a different effective context, proportionally. */
export function budgetFor(total: number): ContextBudget {
  if (!Number.isInteger(total) || total <= 0)
    throw new Error("ai-storage: an effective context must be a positive integer");
  const k = total / DEFAULT_BUDGET.total;
  const scaled = {
    total,
    harness: Math.floor(DEFAULT_BUDGET.harness * k),
    task: Math.floor(DEFAULT_BUDGET.task * k),
    navigation: Math.floor(DEFAULT_BUDGET.navigation * k),
    memory: Math.floor(DEFAULT_BUDGET.memory * k),
    generation: 0,
  };
  // Whatever the floors left over goes to generation, which is the lane that
  // can use an odd number of tokens without anything downstream caring.
  scaled.generation = total - (scaled.harness + scaled.task + scaled.navigation + scaled.memory);
  return assertBudget(scaled);
}

export interface OverflowError {
  error: "MEMORY_CONTEXT_LIMIT";
  lane: Lane;
  requestedTokens: number;
  availableTokens: number;
  /** What the caller can do about it, in the caller's own vocabulary. */
  hint: string;
}

export class ContextLimitExceeded extends Error {
  readonly detail: OverflowError;
  constructor(detail: OverflowError) {
    super(
      `MEMORY_CONTEXT_LIMIT: lane "${detail.lane}" was asked for ${detail.requestedTokens} ` +
        `tokens and has ${detail.availableTokens}. Refused rather than truncated.`,
    );
    this.name = "ContextLimitExceeded";
    this.detail = detail;
  }
}

/**
 * A budget with money already spent out of it.
 *
 * Deliberately mutable and deliberately not clever: one object per model turn,
 * every read charged to a lane, and a running total that a run record can
 * print. The interesting property is that `spend` is the only way to consume,
 * so there is exactly one place where a read could have been truncated and it
 * is the place that refuses to.
 */
export class BudgetLedger {
  readonly budget: ContextBudget;
  readonly #spent: Record<Lane, number> = {
    harness: 0,
    task: 0,
    navigation: 0,
    memory: 0,
    generation: 0,
  };

  constructor(budget: ContextBudget = DEFAULT_BUDGET) {
    this.budget = assertBudget({ ...budget });
  }

  spentIn(lane: Lane): number {
    return this.#spent[lane];
  }

  remainingIn(lane: Lane): number {
    return this.budget[lane] - this.#spent[lane];
  }

  /** Total charged across every lane, which is what the run record reports. */
  spent(): number {
    return LANES.reduce((n, lane) => n + this.#spent[lane], 0);
  }

  /** Would this fit? Asked before assembling, so nothing is built to be thrown away. */
  fits(lane: Lane, tokens: number): boolean {
    return tokens <= this.remainingIn(lane);
  }

  /**
   * Charge a lane, or refuse.
   *
   * Never returns a smaller number than it was asked for. There is no partial
   * success here: the caller either gets the tokens it asked for or gets an
   * error naming what it asked for and what was left, which is the information
   * it needs to ask for less.
   */
  spend(lane: Lane, tokens: number, hint: string): void {
    if (!Number.isInteger(tokens) || tokens < 0)
      throw new Error(`ai-storage: cannot spend ${tokens} tokens from "${lane}"`);
    const available = this.remainingIn(lane);
    if (tokens > available)
      throw new ContextLimitExceeded({
        error: "MEMORY_CONTEXT_LIMIT",
        lane,
        requestedTokens: tokens,
        availableTokens: available,
        hint,
      });
    this.#spent[lane] += tokens;
  }

  /** What the ledger looks like, for a run record. */
  snapshot(): Record<Lane | "total", number> {
    return {
      harness: this.#spent.harness,
      task: this.#spent.task,
      navigation: this.#spent.navigation,
      memory: this.#spent.memory,
      generation: this.#spent.generation,
      total: this.spent(),
    };
  }
}

/**
 * How tokens are counted, and why it is an interface.
 *
 * The only honest counter is the model's own tokenizer, and it lives in the
 * engine's process. Everything else is an estimate. So counting is injected:
 * the benchmark uses the real one, unit tests use a deterministic fake, and no
 * code path anywhere gets to quietly substitute one for the other.
 *
 * A wrong count in the *safe* direction is still wrong. Over-counting means
 * refusing reads that would have fit, which shows up as a worse navigation
 * ratio — the number this whole component is trying to make large. Do not
 * "add 10% to be safe".
 */
export interface TokenCounter {
  /** How this counter was obtained, for the run record. */
  readonly describe: string;
  count(text: string): number;
}

/**
 * A counter for tests and for cost estimates before a server exists.
 *
 * Four characters per token is the usual rough figure for English prose. It is
 * an estimate and it says so in `describe`, which is the point: a run record
 * that carries `chars/4` instead of a tokenizer name is a run record whose
 * ratios are approximate, and the reader can see that without asking.
 */
export const approxCounter: TokenCounter = {
  describe: "estimate: chars/4 — not a tokenizer",
  count: (text: string) => Math.ceil(text.length / 4),
};
