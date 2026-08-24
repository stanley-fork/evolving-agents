/**
 * The index Qwen navigates, and the budget that keeps it navigable.
 *
 * ## The index answers one question
 *
 * > Where should I look?
 *
 * Not *what is this project about*. That distinction is the whole design. An
 * index written as prose —
 *
 * ```text
 *   Architecture
 *   This project has evolved over several months and uses a
 *   sophisticated architecture...
 * ```
 *
 * — costs six hundred tokens and narrows nothing. An index written as a
 * directory —
 *
 * ```text
 *   architecture/
 *     agent-capabilities
 *     model-routing
 *     storage-boundaries
 * ```
 *
 * — costs forty and eliminates nine tenths of the store. The reader is a model
 * with two thousand navigation tokens for the whole descent, and prose spends
 * them describing rather than dividing.
 *
 * ## Why a node has a hard budget
 *
 * One `INDEX.md` that grows without limit is the failure mode this component
 * exists to prevent, arriving through the back door. At ten thousand notes it
 * is larger than the context it is supposed to make unnecessary.
 *
 * So every node has a token budget, and a node that exceeds it **splits**. Not
 * "should be split" — `planSplit` computes the split and `assertWithinBudget`
 * refuses to render a node that has not been split yet. A budget nothing
 * enforces is a comment.
 *
 * ## What splitting must never do
 *
 * Lose a note. After any split, every note that was reachable is still
 * reachable, and `test/tree.test.ts` asserts exactly that over generated trees
 * rather than over one example.
 *
 * Pure. No fetch, no clock, no DOM. Token counting is injected — see
 * `context/budget.ts` for why estimating it here would be a lie in the
 * benchmark's denominator.
 */
import type { TokenCounter } from "../context/budget.ts";

export interface IndexEntry {
  /** The note's id, or a child directory's name when `kind` is `dir`. */
  name: string;
  kind: "note" | "dir";
  /** Shown beside the name. One short clause, or nothing. */
  hint?: string;
}

export interface IndexNode {
  /** Store-absolute, e.g. `/architecture/permissions`. Always starts with `/`. */
  path: string;
  entries: IndexEntry[];
}

export interface IndexLimits {
  rootMaxTokens: number;
  nodeMaxTokens: number;
  maximumDepth: number;
}

/** The starting point from the specification. Configuration, not physics. */
export const DEFAULT_LIMITS: Readonly<IndexLimits> = Object.freeze({
  rootMaxTokens: 1200,
  nodeMaxTokens: 1400,
  maximumDepth: 8,
});

export class IndexTooLarge extends Error {
  readonly detail: { path: string; tokens: number; budget: number; entries: number };
  constructor(detail: { path: string; tokens: number; budget: number; entries: number }) {
    super(
      `ai-storage: index node "${detail.path}" renders to ${detail.tokens} tokens against a ` +
        `budget of ${detail.budget} (${detail.entries} entries). It must be split before it ` +
        `can be shown to a model.`,
    );
    this.name = "IndexTooLarge";
    this.detail = detail;
  }
}

export class IndexTooDeep extends Error {
  constructor(path: string, depth: number, max: number) {
    super(`ai-storage: index path "${path}" is ${depth} deep against a maximum of ${max}`);
    this.name = "IndexTooDeep";
  }
}

export function depthOf(path: string): number {
  return path.split("/").filter(Boolean).length;
}

/**
 * Render a node the way the model will see it.
 *
 * This exact text is what gets counted and what gets sent. Rendering one string
 * for the budget and a different one for the prompt is how a budget stops being
 * a budget, so there is one function.
 *
 * Directories first, then notes, each group alphabetical. Directories first
 * because the descent is the point: a model reading top to bottom meets the
 * ways to narrow before it meets the things to read.
 */
export function renderNode(node: IndexNode): string {
  const dirs = node.entries.filter((e) => e.kind === "dir").sort(cmp);
  const notes = node.entries.filter((e) => e.kind === "note").sort(cmp);
  const line = (e: IndexEntry) =>
    e.kind === "dir"
      ? `  ${e.name}/${e.hint ? "  " + e.hint : ""}`
      : `  ${e.name}${e.hint ? "  " + e.hint : ""}`;
  return [node.path, ...dirs.map(line), ...notes.map(line)].join("\n");
}

function cmp(a: IndexEntry, b: IndexEntry): number {
  return a.name < b.name ? -1 : a.name > b.name ? 1 : 0;
}

export function budgetForNode(node: IndexNode, limits: IndexLimits = DEFAULT_LIMITS): number {
  return node.path === "/" ? limits.rootMaxTokens : limits.nodeMaxTokens;
}

/** Refuse to show a node that has outgrown its budget. */
export function assertWithinBudget(
  node: IndexNode,
  counter: TokenCounter,
  limits: IndexLimits = DEFAULT_LIMITS,
): IndexNode {
  const depth = depthOf(node.path);
  if (depth > limits.maximumDepth) throw new IndexTooDeep(node.path, depth, limits.maximumDepth);
  const budget = budgetForNode(node, limits);
  const tokens = counter.count(renderNode(node));
  if (tokens > budget)
    throw new IndexTooLarge({ path: node.path, tokens, budget, entries: node.entries.length });
  return node;
}

export interface SplitPlan {
  /** The node as it will look afterwards: its notes replaced by directories. */
  parent: IndexNode;
  /** The new children, each holding a share of what the parent had. */
  children: IndexNode[];
}

/**
 * How a node is split, and why by prefix.
 *
 * Grouped by the first path-ish segment of an entry's name, falling back to its
 * first character. Prefix rather than semantics, because splitting is
 * *mechanics* and mechanics are code's job — a split that asks a model where
 * things belong is a split whose result differs between two runs on the same
 * data, which makes the benchmark unrepeatable. Where a *new* note belongs is a
 * semantic question and it goes to the Indexer; where an existing overfull
 * directory divides is not.
 *
 * When the prefixes do not divide the node — one group holding nearly
 * everything — it falls back to fixed-size buckets named by their range, e.g.
 * `aa-fm`. Ugly and honest: the alternative is a plan that returns a child
 * still over budget, and a split that does not split is worse than no split
 * because the caller believes it worked.
 */
export function planSplit(
  node: IndexNode,
  counter: TokenCounter,
  limits: IndexLimits = DEFAULT_LIMITS,
): SplitPlan | null {
  const budget = budgetForNode(node, limits);
  if (counter.count(renderNode(node)) <= budget) return null;
  if (depthOf(node.path) >= limits.maximumDepth)
    throw new IndexTooDeep(node.path, depthOf(node.path) + 1, limits.maximumDepth);

  const notes = node.entries.filter((e) => e.kind === "note");
  const dirs = node.entries.filter((e) => e.kind === "dir");
  if (notes.length < 2) return null;

  const base = node.path === "/" ? "" : node.path;
  const build = (chosen: Map<string, IndexEntry[]>): SplitPlan => {
    const children: IndexNode[] = [];
    const parentEntries: IndexEntry[] = [...dirs];
    for (const [name, group] of [...chosen.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
      children.push({ path: `${base}/${name}`, entries: group });
      parentEntries.push({ kind: "dir", name, hint: `${group.length}` });
    }
    return { parent: { path: node.path, entries: parentEntries }, children };
  };
  const parentFits = (plan: SplitPlan) => counter.count(renderNode(plan.parent)) <= budget;

  /**
   * The split has to fit the *parent* too, and the first version did not check.
   *
   * Grouping by prefix can produce as many groups as there were notes — a store
   * of `note0000 … note0399` has four hundred distinct prefixes — and the
   * parent then lists four hundred directories, which is the node that was over
   * budget with an extra level of indirection in front of it. So a candidate
   * split is only accepted once the parent it produces has been measured.
   */
  const byPrefix = groupByPrefix(notes);
  const biggest = Math.max(...[...byPrefix.values()].map((g) => g.length));
  if (byPrefix.size >= 2 && biggest <= notes.length * 0.8) {
    const plan = build(byPrefix);
    if (parentFits(plan)) return plan;
  }

  // Halve the number of buckets until the parent fits. Two is the floor: a
  // one-bucket split is not a split, and returning one anyway would loop.
  let buckets = Math.max(2, Math.min(notes.length, byPrefix.size || notes.length));
  for (;;) {
    const plan = build(bucketise(notes, Math.ceil(notes.length / buckets)));
    if (parentFits(plan)) return plan;
    if (buckets <= 2)
      // Not even two directory lines fit beside what this node already holds.
      // Reported rather than papered over: the caller gets the node's real
      // numbers instead of a plan that does not help.
      throw new IndexTooLarge({
        path: node.path,
        tokens: counter.count(renderNode(plan.parent)),
        budget,
        entries: plan.parent.entries.length,
      });
    buckets = Math.max(2, Math.floor(buckets / 2));
  }
}

function groupByPrefix(entries: IndexEntry[]): Map<string, IndexEntry[]> {
  const out = new Map<string, IndexEntry[]>();
  for (const e of entries) {
    const head = e.name.split(/[-_/.]/)[0] || e.name[0] || "_";
    const key = head.toLowerCase();
    const list = out.get(key);
    if (list) list.push(e);
    else out.set(key, [e]);
  }
  return out;
}

function bucketise(entries: IndexEntry[], per: number): Map<string, IndexEntry[]> {
  const sorted = [...entries].sort(cmp);
  const out = new Map<string, IndexEntry[]>();
  for (let i = 0; i < sorted.length; i += per) {
    const slice = sorted.slice(i, i + per);
    const lo = (slice[0]?.name ?? "").slice(0, 2).toLowerCase() || "aa";
    const hi = (slice[slice.length - 1]?.name ?? "").slice(0, 2).toLowerCase() || "zz";
    let name = lo === hi ? lo : `${lo}-${hi}`;
    // Two buckets can want the same name when many entries share two letters.
    // Suffixed rather than merged: merging them would put the node back over
    // budget, which is the thing being fixed.
    let n = 2;
    while (out.has(name)) name = `${lo}-${hi}-${n++}`;
    out.set(name, slice);
  }
  return out;
}

/**
 * Split until nothing is over budget, and say what it took.
 *
 * Returns every node of the resulting subtree. The caller writes them; this
 * function decides. `rounds` is reported because a store that needs eleven
 * rounds to settle is telling you something about its naming, and a silent
 * loop would swallow it.
 */
export function splitUntilFits(
  node: IndexNode,
  counter: TokenCounter,
  limits: IndexLimits = DEFAULT_LIMITS,
): { nodes: IndexNode[]; rounds: number } {
  const out: IndexNode[] = [];
  const queue: IndexNode[] = [node];
  let rounds = 0;
  while (queue.length) {
    const next = queue.shift()!;
    const plan = planSplit(next, counter, limits);
    if (!plan) {
      // Still over budget with nothing left to split: refuse rather than
      // pretend. A single note whose rendered line exceeds a node budget is a
      // naming problem, and the error names it.
      assertWithinBudget(next, counter, limits);
      out.push(next);
      continue;
    }
    rounds += 1;
    out.push(plan.parent);
    queue.push(...plan.children);
  }
  return { nodes: out, rounds };
}

/** Every note id reachable from a set of nodes, for the no-loss property. */
export function reachableNotes(nodes: readonly IndexNode[]): Set<string> {
  const out = new Set<string>();
  for (const n of nodes) for (const e of n.entries) if (e.kind === "note") out.add(e.name);
  return out;
}
