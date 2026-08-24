/**
 * Progress, and the property that a crash cannot corrupt it.
 *
 * The interesting test here is the last one: indexing interrupted at every
 * possible point and resumed must produce the same cover as indexing that ran
 * straight through. That is the property `processedUntil = 15629` cannot have,
 * and it is why this module exists.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  BadRange,
  assertRange,
  coverOf,
  coveredBytes,
  holesIn,
  nextGap,
  progressOf,
  type Range,
} from "../src/provenance/ranges.ts";

test("a range with no width is refused — a citation that points at nothing", () => {
  assert.throws(() => assertRange({ from: 5, to: 5 }), BadRange);
  assert.throws(() => assertRange({ from: 5, to: 4 }), BadRange);
  assert.throws(() => assertRange({ from: -1, to: 4 }), BadRange);
  assert.throws(() => assertRange({ from: 0, to: 1.5 }), BadRange);
  assert.deepEqual(assertRange({ from: 0, to: 1 }), { from: 0, to: 1 });
});

test("touching ranges merge, because a gap with no bytes in it is not a gap", () => {
  assert.deepEqual(coverOf([{ from: 0, to: 10 }, { from: 10, to: 20 }]), [{ from: 0, to: 20 }]);
  assert.deepEqual(coverOf([{ from: 10, to: 20 }, { from: 0, to: 10 }]), [{ from: 0, to: 20 }]);
  // One byte apart is a gap.
  assert.deepEqual(coverOf([{ from: 0, to: 10 }, { from: 11, to: 20 }]), [
    { from: 0, to: 10 },
    { from: 11, to: 20 },
  ]);
});

test("a range wholly inside another disappears into it", () => {
  assert.deepEqual(coverOf([{ from: 0, to: 100 }, { from: 20, to: 30 }]), [{ from: 0, to: 100 }]);
});

test("the holes are what the specification draws", () => {
  const notes: Range[] = [
    { from: 0, to: 1100 },
    { from: 1100, to: 2450 },
    { from: 3000, to: 3800 },
  ];
  assert.deepEqual(holesIn(10000, notes), [
    { from: 2450, to: 3000 },
    { from: 3800, to: 10000 },
  ]);
  assert.equal(coveredBytes(10000, notes), 1100 + 1350 + 800);
});

test("a source nothing has read is one hole the size of the source", () => {
  assert.deepEqual(holesIn(500, []), [{ from: 0, to: 500 }]);
  assert.equal(coveredBytes(500, []), 0);
});

test("the total is given, never inferred from the highest offset", () => {
  // This is the original bug wearing a new hat: infer the length from what has
  // been read and a source whose tail was never opened reports itself complete.
  const read: Range[] = [{ from: 0, to: 100 }];
  assert.deepEqual(holesIn(100, read), []);
  assert.deepEqual(holesIn(4000, read), [{ from: 100, to: 4000 }]);
});

test("evidence past the end of the source does not invent coverage", () => {
  // A stale note citing bytes that no longer exist must not make the file look
  // more covered than it is.
  const stale: Range[] = [{ from: 0, to: 50 }, { from: 900, to: 1000 }];
  assert.equal(coveredBytes(100, stale), 50);
  assert.deepEqual(holesIn(100, stale), [{ from: 50, to: 100 }]);
});

test("progress carries the counts behind the fraction", () => {
  const p = progressOf(1000, [{ from: 0, to: 250 }]);
  assert.equal(p.fraction, 0.25);
  assert.equal(p.covered, 250);
  assert.equal(p.total, 1000);
  assert.equal(p.holes, 1);
  // An empty source is complete rather than a division by zero.
  assert.equal(progressOf(0, []).fraction, 1);
});

test("the next gap is the first one, clipped to the chunk size", () => {
  const notes: Range[] = [{ from: 0, to: 1000 }, { from: 3000, to: 4000 }];
  assert.deepEqual(nextGap(10000, notes, 500), { from: 1000, to: 1500 });
  assert.deepEqual(nextGap(10000, notes, 5000), { from: 1000, to: 3000 });
  // Nothing left only when the cover really is complete — the hole at
  // [1000, 3000) above is real, and a shorter total does not close it.
  assert.deepEqual(nextGap(4000, notes, 500), { from: 1000, to: 1500 });
  assert.equal(
    nextGap(1000, [{ from: 0, to: 1000 }], 500),
    null,
    "nothing left is the only completion signal",
  );
  assert.throws(() => nextGap(10, [], 0), /not a chunk/);
});

test("interruption at any point and resume gives the same cover as one clean run", () => {
  // The property `processedUntil` cannot have. Index a source in chunks, kill
  // the process after every chunk in turn, resume from the persisted evidence
  // only, and compare.
  const TOTAL = 4321;
  const CHUNK = 300;

  const runToCompletion = (): Range[] => {
    const done: Range[] = [];
    for (;;) {
      const gap = nextGap(TOTAL, done, CHUNK);
      if (!gap) break;
      done.push(gap);
    }
    return coverOf(done);
  };
  const clean = runToCompletion();
  assert.deepEqual(clean, [{ from: 0, to: TOTAL }]);

  for (let killAfter = 0; killAfter < 20; killAfter += 1) {
    const persisted: Range[] = [];
    // First run: do `killAfter` chunks, then die. A chunk that was read but
    // whose write did not commit leaves nothing behind — which is the case that
    // a cursor gets wrong and derived progress gets right.
    for (let i = 0; i < killAfter; i += 1) {
      const gap = nextGap(TOTAL, persisted, CHUNK);
      if (!gap) break;
      persisted.push(gap);
    }
    // Second run: resume from what was persisted, nothing else.
    for (;;) {
      const gap = nextGap(TOTAL, persisted, CHUNK);
      if (!gap) break;
      persisted.push(gap);
    }
    assert.deepEqual(
      coverOf(persisted),
      clean,
      `resuming after ${killAfter} committed chunks must reach the same cover`,
    );
  }
});

test("chunks arriving out of order still produce the right holes", () => {
  // The archivist is allowed to finish notes in any order; only what committed
  // counts, and coverage is order-independent by construction.
  const out: Range[] = [
    { from: 600, to: 900 },
    { from: 0, to: 300 },
    { from: 300, to: 600 },
  ];
  assert.deepEqual(coverOf(out), [{ from: 0, to: 900 }]);
  assert.deepEqual(holesIn(1200, out), [{ from: 900, to: 1200 }]);
});
