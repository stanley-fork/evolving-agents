/**
 * What the activity canvas may and may not claim.
 *
 * These run against the same text the browser gets: `grid.ts` ships `GRID_JS`
 * as a source string and its exports are a thin wrapper around it, so a test
 * that passes here is a statement about the page and not about a paraphrase of
 * the page.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { cellsOf, rowsOf, assertCitedCell, bucketFor, CELL_RANK } from "../src/grid.ts";
import { threadsOf } from "../src/threads.ts";
import { traceOf } from "../src/trace.ts";
import { agentOfIntent } from "../src/server.ts";
import { DEMO_AT } from "../src/simulate.ts";
import { cochleaFlows } from "../src/cochlea-demo.ts";
import { cochleaProjectFlows } from "../src/cochlea-project.ts";
import { hemoFlows } from "../src/hemo-demo.ts";
import { memoryFlows } from "../src/memory-demo.ts";

const docsOf = (raw: Array<Record<string, unknown>>) =>
  raw.map((d) => ({
    id: d["id"] as string,
    title: d["title"] as string,
    state: d["state"] as string,
    trace: traceOf(d["steps"] as never, agentOfIntent),
  }));

const coclea = () =>
  threadsOf(docsOf([...cochleaFlows(DEMO_AT), ...cochleaProjectFlows(DEMO_AT)] as never));
const hemo = () => threadsOf(docsOf(hemoFlows(DEMO_AT) as never));
const memory = () => threadsOf(docsOf(memoryFlows(DEMO_AT) as never));

const whole = (w: ReturnType<typeof coclea>, bucket: number) =>
  cellsOf(w, { t0: 0, t1: w.span * 1.02, bucket });

test("a bucket with nothing recorded in it produces no square", () => {
  const w = coclea();
  const bucket = 10;
  const cells = whole(w, bucket);
  const filled = new Set(cells.map((c) => c.flowId + "|" + c.col));

  // Pick a flow and a bucket that no rest and no cross touches, then assert
  // that the grid emitted nothing there rather than an empty-looking square.
  const th = w.threads[0]!;
  const touched = (t0: number, t1: number) =>
    th.rests.some((r) => r.x1 > t0 && r.x0 < t1) ||
    th.crosses.some((c) => c.x1 > t0 && c.x0 < t1);

  let checked = 0;
  const nCols = Math.ceil((w.span * 1.02) / bucket);
  for (let c = 0; c < nCols; c += 1) {
    if (touched(c * bucket, (c + 1) * bucket)) continue;
    assert.equal(filled.has(th.flowId + "|" + c), false,
      "an untouched bucket must not become a square");
    checked += 1;
  }
  assert.ok(checked > 0, "the fixture must contain at least one empty bucket to test");
});

test("an unrecorded handoff is a hole, not a faint square", () => {
  const w = coclea();
  const gaps = w.threads.flatMap((t) => t.crosses.filter((c) => c.state === "unknown"));
  assert.ok(gaps.length > 0, "the coclea scope must contain an unrecorded hop");
  const cells = whole(w, 5);
  for (const g of gaps) {
    // The hop itself contributes nothing. Any square in its columns must come
    // from a rest or another hop that really was recorded.
    const cols = cells.filter(
      (c) => c.flowId === g.flowId && c.t0 < g.x1 && c.t1 > g.x0,
    );
    for (const c of cols) assert.ok(c.count > 0);
    const fromHop = cols.filter((c) => c.steps.length === 1 && c.steps[0] === g.toIndex
      && c.holders.length === 1 && c.holders[0] === g.to && c.count === 1);
    assert.equal(fromHop.length, 0, "an unknown hop must not be the only reason for a square");
  }
});

test("a step that lasted longer than a bucket lights every bucket it covers", () => {
  const w = coclea();
  const bucket = 5;
  const cells = whole(w, bucket);
  const long = w.threads
    .flatMap((t) => t.rests.map((r) => ({ t, r })))
    .filter(({ r }) => r.x1 - r.x0 > bucket * 2.5)[0];
  assert.ok(long, "the fixture must contain a step longer than two buckets");
  const spanned = cells.filter(
    (c) => c.flowId === long.t.flowId && c.t0 < long.r.x1 && c.t1 > long.r.x0,
  );
  const want = Math.floor(long.r.x1 / bucket) - Math.floor(long.r.x0 / bucket) + 1;
  assert.equal(spanned.length, want, "duration is columns, not one square");
});

test("a square reports the most consequential thing in it, never an average", () => {
  const w = hemo();
  const cells = whole(w, 30);
  for (const c of cells) {
    // Every state present in the bucket must rank at or below the one reported.
    const rests = w.threads
      .find((t) => t.flowId === c.flowId)!
      .rests.filter((r) => r.x0 < c.t1 && r.x1 > c.t0);
    const crosses = w.threads
      .find((t) => t.flowId === c.flowId)!
      .crosses.filter((x) => x.state !== "unknown" && x.x0 < c.t1 && x.x1 > c.t0);
    const present = [
      ...rests.map((r) =>
        r.state === "running" ? "running"
          : r.state === "failed" ? "blocked"
            : r.state === "blocked" ? "open"
              : r.state === "pending" || r.state === "waiting" || r.state === "draft" ? "pending"
                : "carried"),
      ...crosses.map((x) => x.state),
    ] as Array<keyof typeof CELL_RANK>;
    for (const p of present) {
      assert.ok(CELL_RANK[p] <= CELL_RANK[c.state],
        c.flowId + " column " + c.col + ": " + p + " outranks the reported " + c.state);
    }
  }
});

test("blocked survives standing next to work that went fine", () => {
  const w = coclea();
  const blocked = w.threads.flatMap((t) => t.crosses).filter((c) => c.state === "blocked");
  assert.ok(blocked.length > 0, "the coclea scope must contain a blocked handoff");
  // At a bucket wide enough to swallow the whole scope, the square covering a
  // blocked hop still says blocked.
  const cells = cellsOf(w, { t0: 0, t1: w.span * 1.02, bucket: w.span * 1.02 });
  for (const b of blocked) {
    const c = cells.find((x) => x.flowId === b.flowId)!;
    assert.equal(c.state, "blocked");
  }
});

test("a square's holder is one of the holders it recorded", () => {
  for (const w of [coclea(), hemo(), memory()]) {
    for (const c of whole(w, 15)) {
      assert.ok(c.holders.length > 0);
      assert.ok(c.holders.includes(c.holder));
      const known = new Set(w.lanes.map((l) => l.id));
      for (const h of c.holders) assert.ok(known.has(h), h + " is not a lane in this world");
    }
  }
});

test("a square that says what was observed says where it is written", () => {
  for (const w of [coclea(), hemo(), memory()]) {
    for (const c of whole(w, 15)) assertCitedCell(c);
  }
});

test("assertCitedCell throws on an observation with no address", () => {
  const c = whole(coclea(), 15).find((x) => x.digest !== null);
  assert.ok(c, "the fixture must contain a square that reports an observation");
  assert.throws(() => assertCitedCell({ ...c, source: null }), /cite/);
});

test("every square's steps are steps of its own flow", () => {
  const w = coclea();
  for (const c of whole(w, 15)) {
    const th = w.threads.find((t) => t.flowId === c.flowId)!;
    const known = new Set<number>([
      ...th.rests.map((r) => r.index),
      ...th.crosses.map((x) => x.toIndex),
    ]);
    for (const s of c.steps) assert.ok(known.has(s), "step " + s + " is not in " + c.flowId);
  }
});

test("rows are ordered by when the flow was last touched", () => {
  const w = coclea();
  const rows = rowsOf(w);
  for (let i = 1; i < rows.length; i += 1) {
    assert.ok(rows[i - 1]!.last >= rows[i]!.last, "rows must run most-recent first");
  }
  assert.deepEqual(rows.map((r) => r.row), rows.map((_, i) => i));
});

test("zooming out merges squares instead of shrinking them", () => {
  const w = coclea();
  const px = 1200;
  const fine = bucketFor(w.span, px, 8, [1, 2, 5, 10, 15, 30, 60, 180, 720, 1440]);
  const coarse = bucketFor(w.span, px / 4, 8, [1, 2, 5, 10, 15, 30, 60, 180, 720, 1440]);
  assert.ok(coarse >= fine, "less room means a wider bucket, not a narrower column");
  assert.ok(px / Math.ceil(w.span / fine) >= 8, "a column must stay wide enough to point at");
  const many = whole(w, fine).length, few = whole(w, coarse).length;
  assert.ok(few <= many, "merging cannot produce more squares than it started with");
});

test("no square is emitted outside the window it was asked for", () => {
  const w = coclea();
  const t0 = w.span * 0.4, t1 = w.span * 0.7;
  for (const c of cellsOf(w, { t0, t1, bucket: 10 })) {
    assert.ok(c.t1 > t0 && c.t0 < t1 + 10);
    assert.ok(c.col >= 0);
  }
});

test("a bucket of zero or less is refused rather than guessed at", () => {
  const w = coclea();
  assert.throws(() => cellsOf(w, { t0: 0, t1: w.span, bucket: 0 }), /positive/);
  assert.throws(() => cellsOf(w, { t0: 0, t1: w.span, bucket: -5 }), /positive/);
});

test("a step the flow calls blocked is held, not failed", () => {
  // In the flow vocabulary a *step* state of "blocked" is work that has been
  // stated and cannot proceed — hemo's A4 carries a null observation and a note
  // saying it is open work. Drawing that as a failure would report the absence
  // of a result as a negative one, which is the error this project is about.
  const w = hemo();
  const held = w.threads.flatMap((t) => t.rests).filter((r) => r.state === "blocked");
  assert.ok(held.length > 0, "the hemo scope must contain a step the flow calls blocked");
  for (const r of held) assert.equal(r.digest, null, "such a step recorded no observation");
  const cells = whole(w, 10);
  const touching = cells.filter((c) =>
    held.some((r) => r.x0 < c.t1 && r.x1 > c.t0)
    && !w.threads.find((t) => t.flowId === c.flowId)!.crosses
      .some((x) => x.state === "blocked" && x.x0 < c.t1 && x.x1 > c.t0));
  assert.ok(touching.length > 0);
  for (const c of touching) assert.notEqual(c.state, "blocked");
});
