/**
 * The activity canvas — a grid of squares, one row per flow.
 *
 * ## The proposal
 *
 * > I think the UI is GitHub's contribution grid, with colours, where each little
 * > square is an agent, a human or a task, and the traces or flows are horizontal
 * > lanes — and it is easier to manipulate, and ends up being genuinely a canvas
 * > of activity.
 *
 * ## Why this beats the bundle, in one property
 *
 * **Every mark has an address you can point at.** On the bundle a step is a
 * stretch of curve whose position depends on the rotation, the zoom and the
 * phase of four other strands; to click it you have to catch it. Here a step is
 * at (row, column) — a flow and a moment — and both of those are things a person
 * already has in their head before they look. Pointing is free, hit areas are
 * rectangles, and the surface stops being something you steer.
 *
 * The bundle was better at one thing and it is worth naming: it showed the
 * *braid*, the fact that flows are one object moving together. The grid shows
 * that as a column — two filled squares in the same column is two flows held at
 * the same moment — which is less beautiful and much easier to check.
 *
 * ## What a square is
 *
 * A square is **one flow, in one bucket of time, held by somebody**. Its colour
 * is who held it; its texture is what happened. Reading across a row is the
 * sequence of hands a flow passed through. Reading down a column is who was busy
 * at that moment, which is the contention question the desk answered with a
 * badge and the swimlanes answered with a crossing.
 *
 * ## The rule this module exists to enforce
 *
 * **A bucket with nothing recorded in it produces no square.** Not a pale one,
 * not a zero — nothing, so the row has a hole in it and the hole is the finding.
 * GitHub's grid can use the palest green for zero because zero commits is a
 * measured fact; here 'nothing was written down' and 'nothing happened' are
 * different claims and only one of them is ours to make. `cellsOf` never emits a
 * cell for an empty bucket, and there is a test that says so.
 *
 * The second rule is the one every surface in this repository carries: a cell
 * that reports what a step observed must cite where that observation is. See
 * `assertCitedCell`, which throws rather than let one through.
 *
 * Pure. No fetch, no clock, no DOM.
 */
import type { ThreadWorld } from "./threads.ts";

/** How significant a state is when several land in one bucket. */
export type CellState =
  | "running"
  | "blocked"
  | "ignored"
  | "open"
  | "pending"
  | "carried";

export interface Cell {
  flowId: string;
  title: string;
  /** The flow's own index, so a renderer can chip the row with its hue. */
  ordinal: number;
  /** Row index in the grid, top to bottom. */
  row: number;
  /** Column index, 0 at the left edge of the window. */
  col: number;
  /** The bucket's bounds, in the world's own units. */
  t0: number;
  t1: number;
  /** Who held the flow in this bucket, most overlap first. */
  holders: string[];
  /** The one with the most of the bucket. Never invented: always in `holders`. */
  holder: string;
  kind: "human" | "agent";
  state: CellState;
  /** How many recorded things — steps and handoffs — fall in this bucket. */
  count: number;
  /** Step indices this square stands for, ascending. */
  steps: number[];
  /** What the most significant thing here observed, when it observed anything. */
  digest: string | null;
  /** Where that observation is. Null only when `digest` is null. */
  source: string | null;
}

export interface Row {
  flowId: string;
  title: string;
  ordinal: number;
  row: number;
  state: string;
  delivered: boolean;
  /** The last moment anything was recorded on this flow. */
  last: number;
}

export interface GridOpts {
  /** Left edge of the window, in world units. */
  t0: number;
  /** Right edge. */
  t1: number;
  /** Bucket width, same units. Must be > 0. */
  bucket: number;
}

export const GRID_JS = String.raw`
/**
 * State precedence inside one bucket.
 *
 * A bucket that contains a blocked handoff and three clean ones is *blocked* —
 * the square reports the most consequential thing in it, never an average. An
 * average would be a number nobody measured, and it would let a failure hide
 * behind the successes standing next to it.
 *
 * 'carried' is the floor rather than a state of its own worth shouting about:
 * it is what the surface looks like when the machine is working.
 */
var RANK = { running: 5, blocked: 4, ignored: 3, open: 2, pending: 1, carried: 0 };

function worseOf(a, b) {
  if (a === null) return b;
  if (b === null) return a;
  return RANK[b] > RANK[a] ? b : a;
}

/**
 * A rest's state, in the grid's vocabulary — and one word that means two things.
 *
 * A *step* whose state is 'blocked' is not a step that failed. In the flow
 * vocabulary it is work that has been stated and cannot proceed: hemo's A4 step
 * carries the note 'stated as open work, because a scope with nothing red in it
 * reads as a finished one', and its observation is null. Nothing ran. Nothing
 * said no. That is **open** here — held, no verdict — and drawing it as a
 * failure would be this project's own headline error committed by its own
 * surface: reporting the absence of a result as a negative one.
 *
 * A *handoff* whose state is 'blocked' is the other thing: it arrived, the
 * receiving step ran, and it did not pass. Same word, opposite claim, and the
 * only reason the collision is survivable is that the two arrive here through
 * different fields. A step that really ran and failed is 'failed'.
 */
function restState(r) {
  if (r.state === 'running') return 'running';
  if (r.state === 'failed') return 'blocked';
  if (r.state === 'blocked') return 'open';
  if (r.state === 'pending' || r.state === 'waiting' || r.state === 'draft') return 'pending';
  return 'carried';
}

/**
 * Rows, ordered by when the flow was last touched.
 *
 * Most recently active at the top, which is the order an activity canvas is
 * read in and the order the answer to 'what is going on' lives in. It is
 * determined by the data, not by declaration order, so two renders of the same
 * world are the same picture.
 */
function rowsOf(world) {
  var rows = [];
  for (var i = 0; i < world.threads.length; i += 1) {
    var th = world.threads[i];
    var last = -Infinity;
    for (var j = 0; j < th.rests.length; j += 1) if (th.rests[j].x1 > last) last = th.rests[j].x1;
    for (var k = 0; k < th.crosses.length; k += 1) if (th.crosses[k].x1 > last) last = th.crosses[k].x1;
    rows.push({
      flowId: th.flowId, title: th.title, ordinal: th.ordinal, row: 0,
      state: th.state, delivered: th.delivered, last: last === -Infinity ? 0 : last,
    });
  }
  rows.sort(function (a, b) { return b.last - a.last || a.ordinal - b.ordinal; });
  for (var n = 0; n < rows.length; n += 1) rows[n].row = n;
  return rows;
}

/** How much of [a0,a1] falls inside [b0,b1]. Zero when they do not touch. */
function overlap(a0, a1, b0, b1) {
  var lo = a0 > b0 ? a0 : b0;
  var hi = a1 < b1 ? a1 : b1;
  return hi > lo ? hi - lo : 0;
}

/**
 * Squares.
 *
 * One pass per flow: walk its rests and crosses, drop each into every bucket it
 * overlaps, and emit a square only for buckets that caught something. A step
 * that lasted an hour at a ten-minute bucket lights six squares, which is the
 * duration the desk could not draw and the grid gets for free.
 *
 * A moment with zero width — a handoff between two steps that touch — still
 * lands in exactly one bucket, because a thing that was recorded happened
 * somewhere and the surface must not lose it to arithmetic.
 */
function cellsOf(world, opts) {
  if (!(opts.bucket > 0)) throw new Error('grid: bucket must be positive');
  var rows = rowsOf(world);
  var byId = {};
  for (var r = 0; r < rows.length; r += 1) byId[rows[r].flowId] = rows[r];
  var lane = {};
  for (var l = 0; l < world.lanes.length; l += 1) lane[world.lanes[l].id] = world.lanes[l];

  var colOf = function (t) { return Math.floor((t - opts.t0) / opts.bucket); };
  var nCols = Math.max(1, Math.ceil((opts.t1 - opts.t0) / opts.bucket));
  var acc = {};

  var put = function (flowId, c, holderId, state, ms, stepIndex, digest, source) {
    if (c < 0 || c >= nCols) return;
    var key = flowId + '|' + c;
    var cell = acc[key];
    if (!cell) {
      cell = acc[key] = {
        flowId: flowId, col: c, held: {}, state: null, count: 0, steps: [],
        digest: null, source: null, top: null,
      };
    }
    cell.count += 1;
    if (stepIndex !== null && cell.steps.indexOf(stepIndex) < 0) cell.steps.push(stepIndex);
    if (holderId !== null) cell.held[holderId] = (cell.held[holderId] || 0) + Math.max(ms, 1);
    var before = cell.state;
    cell.state = worseOf(cell.state, state);
    // The square reports what the most consequential thing in it observed. When
    // two things tie, the first one recorded wins: no ordering is invented.
    if (before === null || cell.state !== before) {
      cell.digest = digest === undefined ? null : digest;
      cell.source = source === undefined ? null : source;
    }
  };

  for (var ti = 0; ti < world.threads.length; ti += 1) {
    var th = world.threads[ti];
    for (var ri = 0; ri < th.rests.length; ri += 1) {
      var rest = th.rests[ri];
      var c0 = colOf(rest.x0), c1 = colOf(rest.x1);
      for (var c = c0; c <= c1; c += 1) {
        var b0 = opts.t0 + c * opts.bucket, b1 = b0 + opts.bucket;
        var ms = overlap(rest.x0, rest.x1, b0, b1);
        if (ms <= 0 && c !== c0) continue;
        put(th.flowId, c, rest.lane, restState(rest), ms, rest.index, rest.digest, rest.source);
      }
    }
    for (var ci = 0; ci < th.crosses.length; ci += 1) {
      var cross = th.crosses[ci];
      // Nothing was recorded about this hop. It is a hole, and a hole is not a
      // square: see the module note.
      if (cross.state === 'unknown') continue;
      var d0 = colOf(cross.x0), d1 = colOf(cross.x1);
      for (var d = d0; d <= d1; d += 1) {
        var e0 = opts.t0 + d * opts.bucket, e1 = e0 + opts.bucket;
        var over = overlap(cross.x0, cross.x1, e0, e1);
        if (over <= 0 && d !== d0) continue;
        put(th.flowId, d, cross.to, cross.state, over, cross.toIndex, cross.digest, cross.source);
      }
    }
  }

  var out = [];
  for (var key in acc) {
    if (!Object.prototype.hasOwnProperty.call(acc, key)) continue;
    var cell = acc[key];
    var ids = Object.keys(cell.held);
    if (!ids.length) continue;
    ids.sort(function (a, b) { return cell.held[b] - cell.held[a] || (a < b ? -1 : 1); });
    var row = byId[cell.flowId];
    var holder = ids[0];
    out.push({
      flowId: cell.flowId, title: row.title, ordinal: row.ordinal,
      row: row.row, col: cell.col,
      t0: opts.t0 + cell.col * opts.bucket, t1: opts.t0 + (cell.col + 1) * opts.bucket,
      holders: ids, holder: holder,
      kind: lane[holder] ? lane[holder].kind : 'agent',
      state: cell.state === null ? 'carried' : cell.state,
      count: cell.count, steps: cell.steps.slice().sort(function (a, b) { return a - b; }),
      digest: cell.digest === undefined ? null : cell.digest,
      source: cell.source === undefined ? null : cell.source,
    });
  }
  out.sort(function (a, b) { return a.row - b.row || a.col - b.col; });
  return out;
}

/**
 * A square that says what was observed must say where it is written.
 *
 * Same rule as every other surface here, and it throws for the same reason: a
 * finding with no address is the failure this whole project is about, and the
 * cheap way to keep it out is to make it impossible to render.
 */
function assertCitedCell(cell) {
  if (cell.digest !== null && cell.digest !== undefined
      && (cell.source === null || cell.source === undefined || cell.source === '')) {
    throw new Error('grid: a square reporting an observation must cite it — ' + cell.flowId
      + ' column ' + cell.col);
  }
  return cell;
}

/**
 * How wide a bucket should be so the canvas is legible, and why it is a choice.
 *
 * A square below about six pixels is not a square any more, and a grid whose
 * columns are thinner than the gaps between them is a texture rather than a
 * thing you can point at. So the bucket is the smallest of the offered widths
 * that keeps a column at or above 'floor' pixels — which means zooming out does
 * not shrink the squares, it *merges* them, and the header has to say by how
 * much. That is the aliasing rule from zoom.ts in its simplest possible form.
 */
function bucketFor(span, px, floor, steps) {
  for (var i = 0; i < steps.length; i += 1) {
    var cols = Math.ceil(span / steps[i]);
    if (cols <= 0) continue;
    if (px / cols >= floor) return steps[i];
  }
  return steps[steps.length - 1];
}
`;

const RULES = new Function(
  GRID_JS +
    "\nreturn { rowsOf: rowsOf, cellsOf: cellsOf, assertCitedCell: assertCitedCell," +
    " bucketFor: bucketFor, RANK: RANK };",
)() as {
  rowsOf: (w: ThreadWorld) => Row[];
  cellsOf: (w: ThreadWorld, o: GridOpts) => Cell[];
  assertCitedCell: (c: Cell) => Cell;
  bucketFor: (span: number, px: number, floor: number, steps: number[]) => number;
  RANK: Record<CellState, number>;
};

/** Rows, most recently touched first. */
export const rowsOf = (world: ThreadWorld): Row[] => RULES.rowsOf(world);

/** The squares in a window. Never one for a bucket with nothing in it. */
export const cellsOf = (world: ThreadWorld, opts: GridOpts): Cell[] =>
  RULES.cellsOf(world, opts);

/** Throws on a square that reports an observation with no address. */
export const assertCitedCell = (cell: Cell): Cell => RULES.assertCitedCell(cell);

/** The coarsest-to-finest bucket that keeps a column at least `floor` px wide. */
export const bucketFor = (
  span: number,
  px: number,
  floor: number,
  steps: number[],
): number => RULES.bucketFor(span, px, floor, steps);

/** State precedence, exported so a renderer and a test read the same table. */
export const CELL_RANK = RULES.RANK;
