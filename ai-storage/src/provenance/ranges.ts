/**
 * Progress, derived rather than stored.
 *
 * ## The bug this file exists to make impossible
 *
 * Write `processedUntil = 15629` and you have created a number that can be true
 * of nothing. The archivist crashes after reading to 15,629 and before the
 * write commits: the cursor says the work is done and no note exists. Or it
 * reads out of order, or it skips a region it could not parse, and the cursor
 * says everything behind it is covered when a hole is sitting in the middle.
 *
 * The largest offset anything reached is not progress. **Progress is the union
 * of the ranges that evidence actually covers**, computed from the notes that
 * were actually persisted, every time it is asked for.
 *
 * ```text
 *   source
 *   0 -------------------------------- 10000
 *   notes:
 *   0------1100
 *          1100------2450
 *                      3000----3800
 *   missing:
 *                 2450--3000
 *                          3800------- 10000
 * ```
 *
 * A restart resumes the holes. Interrupt indexing at any point, run it again,
 * and the result is the same as if it had never been interrupted — which is a
 * property a test can assert, and `test/ranges.test.ts` does.
 *
 * Ranges are half-open: `[from, to)`. Half-open because adjacency then means
 * `a.to === b.from` with no off-by-one, and because a zero-width range is
 * exactly `from === to`, which is a citation that points at nothing and is
 * refused everywhere.
 *
 * Pure. No fetch, no clock, no DOM.
 */

export interface Range {
  from: number;
  to: number;
}

export class BadRange extends Error {
  constructor(r: Range, why: string) {
    super(`ai-storage: range [${r.from}, ${r.to}) ${why}`);
    this.name = "BadRange";
  }
}

export function assertRange(r: Range): Range {
  if (!Number.isInteger(r.from) || !Number.isInteger(r.to))
    throw new BadRange(r, "has a non-integer bound");
  if (r.from < 0) throw new BadRange(r, "starts before zero");
  if (r.to <= r.from) throw new BadRange(r, "has no width — a citation with no width cites nothing");
  return r;
}

/**
 * Merge overlapping and touching ranges into a canonical cover.
 *
 * Touching counts: `[0,10)` and `[10,20)` become `[0,20)`, because between them
 * there is no byte that is uncovered, and a hole with no bytes in it is not a
 * hole. Getting this wrong produces phantom work — an indexer that re-reads a
 * zero-width gap forever.
 */
export function coverOf(ranges: readonly Range[]): Range[] {
  const sorted = ranges.map(assertRange).sort((a, b) => a.from - b.from || a.to - b.to);
  const out: Range[] = [];
  for (const r of sorted) {
    const last = out[out.length - 1];
    if (last && r.from <= last.to) {
      if (r.to > last.to) last.to = r.to;
      continue;
    }
    out.push({ from: r.from, to: r.to });
  }
  return out;
}

/**
 * What is left, given what is covered.
 *
 * `total` is the length of the source in bytes. It is a required argument and
 * not inferred from the highest covered offset, because inferring it is the
 * original bug in a new hat: a source whose last region was never read would
 * report itself complete.
 */
export function holesIn(total: number, ranges: readonly Range[]): Range[] {
  if (!Number.isInteger(total) || total < 0)
    throw new Error(`ai-storage: a source length of ${total} is not a length`);
  const cover = coverOf(ranges);
  const out: Range[] = [];
  let at = 0;
  for (const r of cover) {
    if (r.from > total) break;
    if (r.from > at) out.push({ from: at, to: Math.min(r.from, total) });
    at = Math.max(at, Math.min(r.to, total));
  }
  if (at < total) out.push({ from: at, to: total });
  return out;
}

/** How many bytes of a source are covered. */
export function coveredBytes(total: number, ranges: readonly Range[]): number {
  return coverOf(ranges).reduce(
    (n, r) => n + Math.max(0, Math.min(r.to, total) - Math.min(r.from, total)),
    0,
  );
}

/**
 * The fraction read, as a number and as the two counts behind it.
 *
 * Both counts travel with the fraction on purpose. `0.5` is a number that
 * cannot be checked; `{ covered: 5000, total: 10000 }` is one that can, and a
 * progress display that shows the fraction without the counts is the kind of
 * summary this repository keeps finding to be wrong.
 */
export function progressOf(
  total: number,
  ranges: readonly Range[],
): { fraction: number; covered: number; total: number; holes: number } {
  const covered = coveredBytes(total, ranges);
  return {
    fraction: total === 0 ? 1 : covered / total,
    covered,
    total,
    holes: holesIn(total, ranges).length,
  };
}

/**
 * The next piece of work, sized to fit.
 *
 * Returns the first hole, clipped to `maxBytes`. Front to back rather than
 * largest-first: a source read in order produces notes whose claims can refer
 * to what came before them, and the archivist's context is small enough that
 * order is the only cheap way to give it any.
 *
 * `null` when there is nothing left, which is the only signal that indexing is
 * complete. There is no separate `done` flag to disagree with it.
 */
export function nextGap(
  total: number,
  ranges: readonly Range[],
  maxBytes: number,
): Range | null {
  if (!(maxBytes > 0)) throw new Error("ai-storage: a chunk of no bytes is not a chunk");
  const holes = holesIn(total, ranges);
  const first = holes[0];
  if (!first) return null;
  return { from: first.from, to: Math.min(first.to, first.from + maxBytes) };
}
