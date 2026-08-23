/**
 * The thread view — a sketch, not a replacement.
 *
 * ## The proposal
 *
 * > A horizontal flow of threads of thought between the agents and the human,
 * > where you can stand on a rope or on an agent and inspect it and work there.
 *
 * ## Why it is a better metaphor than the desk, in one property
 *
 * **The desk has no time axis.** Cards sit in a grid; the order a flow happened
 * in is only inferable from following the wires, and their arrangement is
 * whatever the layout engine or the user's hand decided. A flow *is* a sequence,
 * and the surface drawing it threw its one intrinsic dimension away.
 *
 * Here X is time and Y is **who is holding the thought**. One flow is one
 * continuous rope: it starts in the human's lane, dips into `DERIVADOR`'s, rises
 * to `CONSTRUCTOR`'s, and — if it finishes — comes back to the human, because a
 * finished flow delivers something to somebody. Nothing on the desk could say
 * that last part at all.
 *
 * Four things the rope shows that a wire cannot:
 *
 * - **Duration.** A step that took an hour is a long stretch of rope.
 * - **Simultaneity.** Two ropes crossing one lane at the same X is one agent
 *   holding two thoughts at once, which is the thing an agent surface most needs
 *   to show and the desk showed as a multiplier badge.
 * - **A gap.** An unrecorded hop is drawn as *rope that is not there* — you see
 *   the dark through it. A dashed line is a line; absence should look absent.
 * - **Slack.** A handoff that carried nothing lands and the rope goes dark from
 *   the landing onward, which is what "it arrived and nothing used it" looks
 *   like.
 *
 * ## What is deliberately not decided here
 *
 * This module computes geometry and nothing else. No WebGL, no renderer, no
 * camera. That is the point of it: the question worth testing first is *does
 * standing on a rope help you answer what happened here*, and that question is
 * answerable in SVG for a day's work. Depth of field and a thousand ropes are
 * what a real engine buys, and they are worth buying **after** the answer is
 * yes, not before.
 *
 * Pure. No fetch, no clock, no DOM.
 */
import type { WireState } from "./bus.ts";
import type { FlowTrace } from "./trace.ts";

/** A row. The human gets one, every agent that appears gets one. */
export interface Lane {
  id: string;
  label: string;
  kind: "human" | "agent";
  /** Row index, top to bottom. The human is always 0. */
  row: number;
}

/**
 * A stretch of rope where the thought is *resting* in one lane.
 *
 * This is a step. Its length is its duration, which is the first thing the desk
 * could not draw.
 */
export interface Rest {
  flowId: string;
  lane: string;
  index: number;
  state: string;
  x0: number;
  x1: number;
  /** What the step said, trimmed. Never the flow's goal (doc/11). */
  result: string | null;
  /** The observation the attempt closed with, when it closed with one. */
  digest: string | null;
  source: string | null;
}

/**
 * A stretch of rope *crossing* between two lanes.
 *
 * This is a handoff, and it carries `bus.ts`'s five states unchanged — the whole
 * vocabulary transfers, which is the test of whether the metaphor is a redesign
 * or a rewrite. It is a redesign: nothing about what may be claimed changes.
 */
export interface Cross {
  flowId: string;
  from: string;
  to: string;
  fromIndex: number;
  toIndex: number;
  x0: number;
  x1: number;
  state: WireState;
  because: string;
  digest: string | null;
  source: string | null;
}

export interface Thread {
  flowId: string;
  title: string;
  state: string;
  /** 0..n-1, so a renderer can give each rope its own hue. */
  ordinal: number;
  rests: Rest[];
  crosses: Cross[];
  /**
   * Where the rope ends, and whether it got back to the human.
   *
   * `delivered` only when the flow is `done`. A rope that stops in an agent's
   * lane is a thought nobody ever received, and that is a different picture from
   * one that came home.
   */
  delivered: boolean;
}

export interface ThreadWorld {
  lanes: Lane[];
  threads: Thread[];
  /** The widest X any rope reaches, so a renderer can scale to it. */
  span: number;
  /**
   * What X actually is, and the field a renderer must not ignore.
   *
   * `clock` — every attempt carried `startedAt`, so X is epoch milliseconds and
   * a gap on screen is a gap that happened. Panning left is going back in time.
   *
   * `sequence` — the store recorded no clock for at least one attempt, so X is
   * step order and every step is the same width. **The axis must say so.** A
   * surface that labels sequence as time is drawing durations nobody measured,
   * which is the same failure as drawing an unrecorded hop like one that worked.
   */
  basis: "clock" | "sequence";
  /** In `clock`, the epoch ms of the earliest and latest thing recorded. */
  t0: number;
  t1: number;
}

/** In `sequence`, how much of a step's slot is rest; the rest is the crossing. */
const REST = 0.68;

/**
 * In `clock`, how much of the gap between two steps the crossing occupies.
 *
 * A handoff is not instantaneous and it is not half the flow. The rope leaves
 * when the producing step closed and arrives when the next one opened, so this
 * only matters when the two touch exactly — then the crossing borrows a sliver
 * from the receiving step so the rope has somewhere to bend.
 */
const MIN_CROSS_MS = 20_000;

/** Room in front of the earliest step, so a person's question can be drawn. */
const LEAD_MS = 4 * 60_000;

interface ThreadDoc {
  id: string;
  title: string;
  state: string;
  trace: FlowTrace;
}

/**
 * The last attempt that closed with an observation.
 *
 * Same rule as `bus.ts`, for the same reason: an attempt that failed and was
 * retried produced nothing the next step could have read.
 */
function obs(step: FlowTrace["steps"][number]) {
  for (let i = step.attempts.length - 1; i >= 0; i -= 1) {
    const a = step.attempts[i]!;
    if (a.digest) return { digest: a.digest, source: a.source };
  }
  return { digest: null as string | null, source: null as string | null };
}

/**
 * When a step opened and closed, from its attempts.
 *
 * First open to last close, across every attempt: a step that failed twice and
 * succeeded on the third took as long as all three took. `null` for `end` when
 * the last attempt never closed, which is what a held step looks like and must
 * keep looking like.
 */
function stepSpan(step: FlowTrace["steps"][number]): { start: number | null; end: number | null } {
  let start: number | null = null;
  let end: number | null = null;
  for (const a of step.attempts) {
    if (a.startedAt != null && (start === null || a.startedAt < start)) start = a.startedAt;
    if (a.finishedAt != null && (end === null || a.finishedAt > end)) end = a.finishedAt;
  }
  // An attempt that is still open ends the step's clock: it has not finished.
  if (step.attempts.some((a) => a.startedAt != null && a.finishedAt == null)) end = null;
  return { start, end };
}

/**
 * Lay the threads out.
 *
 * Lanes are ordered by **first appearance across all flows**, not alphabetically
 * and not by declaration order. That makes the ropes tend downward rather than
 * zig-zag, which matters: a reader following one rope through eight lanes is
 * following a shape, and a shape with no tendency is a tangle.
 */
export function threadsOf(docs: ThreadDoc[], humanLabel = "you"): ThreadWorld {
  /**
   * Lanes ordered by where each agent *tends* to sit in a flow, not by where it
   * first appeared.
   *
   * First-appearance was the obvious rule and it produced a bad picture: in the
   * hemodynamics scope the shortest thread happens to be listed first, so its
   * agents claimed the top lanes, and the long thread then had to plunge past
   * them and climb back. Every rope was legible and the *set* of them was a
   * tangle.
   *
   * The mean step index is a cheap, honest ordering: an agent that usually acts
   * early sits high, one that usually acts last sits low, so a rope tends
   * downward and a reader following one is following a slope rather than a
   * search. It is not an optimal crossing-minimisation — that is a real problem
   * with real algorithms — and it does not need to be. The measure of a sketch
   * is whether the question it exists to answer becomes answerable.
   *
   * Ties break on first appearance, so the ordering is deterministic: a layout
   * that reshuffles between renders of the same data is a layout nobody can
   * learn.
   */
  const stat = new Map<string, { sum: number; n: number; first: number }>();
  let seq = 0;
  for (const doc of docs) {
    const ordered = doc.trace.steps.filter((s) => s.agent).sort((a, b) => a.index - b.index);
    ordered.forEach((s, i) => {
      const denom = Math.max(1, ordered.length - 1);
      const at = i / denom;
      const cur = stat.get(s.agent!);
      if (cur) {
        cur.sum += at;
        cur.n += 1;
      } else {
        stat.set(s.agent!, { sum: at, n: 1, first: seq++ });
      }
    });
  }

  const ordered = [...stat.entries()].sort((a, b) => {
    const ma = a[1].sum / a[1].n;
    const mb = b[1].sum / b[1].n;
    return ma === mb ? a[1].first - b[1].first : ma - mb;
  });

  const lanes: Lane[] = [{ id: "@human", label: humanLabel, kind: "human", row: 0 }];
  ordered.forEach(([name], i) =>
    lanes.push({ id: name, label: name, kind: "agent", row: i + 1 }),
  );
  const seen = new Set<string>(lanes.map((l) => l.id));
  const laneOf = (name: string) => {
    // Every agent has a lane by construction; this stays as the assertion that
    // says so, because a step whose agent has no row is drawn at y=undefined and
    // vanishes silently.
    if (!seen.has(name)) throw new Error(`${name} has steps and no lane`);
    return name;
  };

  /**
   * Which basis the data can actually support, decided once for the whole world.
   *
   * All-or-nothing on purpose. Mixing a clock-drawn thread with a
   * sequence-drawn one on a single axis puts two incomparable things in the same
   * picture and invites a reader to compare their widths — which is a duration
   * claim about a step nobody timed. If any settled attempt lacks a start, the
   * whole world falls back to sequence and says so.
   */
  const settled = docs.flatMap((d) =>
    d.trace.steps.filter((s) => s.agent).flatMap((s) => s.attempts),
  );
  const basis: ThreadWorld["basis"] =
    settled.length > 0 && settled.every((a) => a.startedAt != null) ? "clock" : "sequence";

  let t0 = Infinity;
  let t1 = -Infinity;
  if (basis === "clock") {
    for (const a of settled) {
      if (a.startedAt != null && a.startedAt < t0) t0 = a.startedAt;
      const e = a.finishedAt ?? a.startedAt;
      if (e != null && e > t1) t1 = e;
    }
    /**
     * The axis starts before the earliest recorded moment.
     *
     * A person asked for the first thread, and asking took time nobody wrote
     * down. Without room in front of it that opening crossing had nowhere to go
     * and came out with zero width — a segment nobody can see or click, and an
     * assertion that the question was instantaneous.
     *
     * The lead-in is the one segment on the page drawn from a convention rather
     * than from a record, and it is the only one.
     */
    t0 -= LEAD_MS;
  }

  const threads: Thread[] = [];
  let widest = 0;

  docs.forEach((doc, ordinal) => {
    const steps = doc.trace.steps.filter((s) => s.agent).sort((a, b) => a.index - b.index);
    if (steps.length === 0) return;
    for (const s of steps) laneOf(s.agent!);

    const rests: Rest[] = [];
    const crosses: Cross[] = [];

    /**
     * Where a step sits on the axis.
     *
     * In `clock`, X is minutes since the earliest recorded moment in the world,
     * so a step's width **is** how long it took and the space between two steps
     * is time nobody was working. In `sequence` every step gets an identical
     * slot, which is the honest picture when no clock exists — and the axis says
     * which of the two it is drawing.
     *
     * A step that started and never closed is given a minimum width rather than
     * zero: it is running or held, it has a position on the page, and a segment
     * of zero length is a segment nobody can click.
     */
    const MIN_REST = 1.2;
    /**
     * `after` is where the thread had got to, and it is not optional.
     *
     * A step that has not started has no timestamp, and the first version fell
     * back to `t0` for it — so a pending step was drawn at the very beginning of
     * the world and its thread stretched across the whole axis to reach it.
     * GATE-D1, whose last steps are pending, came out spanning seventy-two hours
     * of a seventy-two hour window: a picture of a flow that has been running
     * for three days, of a flow that started forty minutes ago.
     *
     * A step that has not begun belongs **just after the last thing that did**.
     * That is a statement about order, which is known, rather than about time,
     * which is not — and the renderer draws it dim so the distinction survives.
     */
    let after = 0;
    const at = (s: (typeof steps)[number], i: number) => {
      if (basis !== "clock") {
        const base = i + (1 - REST);
        after = base + 1;
        return { x0: base, x1: base + REST };
      }
      const sp = stepSpan(s);
      const x0 = sp.start != null ? (sp.start - t0) / 60_000 : after + MIN_CROSS_MS / 60_000;
      /**
       * A step that started and has not finished runs up to the present.
       *
       * It used to get a fixed minimum width, so the one open step in the demo
       * was drawn as a stub that ended a minute after it began — a picture of
       * work that stopped, of work that is still going. On a time axis an
       * unclosed step reaches the latest moment anything was recorded, and a
       * renderer that knows the real clock can carry it further.
       *
       * `sp.start != null && sp.end == null` is the test, not the step's state
       * string: what makes a step open is an attempt with no finish, and that is
       * a fact in the store rather than a label on top of it.
       */
      const stillOpen = sp.start != null && sp.end == null;
      const x1 = sp.end != null
        ? (sp.end - t0) / 60_000
        : stillOpen
          ? (t1 - t0) / 60_000
          : x0 + MIN_REST;
      const out = { x0, x1: Math.max(x1, x0 + MIN_REST) };
      if (out.x1 > after) after = out.x1;
      return out;
    };

    /**
     * The first step is measured twice, and that is deliberate.
     *
     * `at` advances `after`, so calling it here to find the lead-in and again in
     * the loop below would move the cursor twice. The loop is the one that
     * counts, so the cursor is put back.
     */
    const first = steps[0]!;
    const firstAt = at(first, 0);
    after = 0;
    // A person's question has no recorded duration, so the opening crossing is
    // given a lead-in rather than pretending to know when they started thinking.
    const lead = basis === "clock" ? Math.min(LEAD_MS / 60_000, firstAt.x0) : 1 - REST;
    crosses.push({
      flowId: doc.id,
      from: "@human",
      to: first.agent!,
      fromIndex: -1,
      toIndex: first.index,
      x0: Math.max(0, firstAt.x0 - lead),
      x1: firstAt.x0,
      // Always `carried`: the human posing the question is the one handoff in
      // the system that is not in question. Marking it `unknown` because no
      // observation was recorded would be pedantry pretending to be rigour.
      state: "carried",
      because: "the question came from a person",
      digest: null,
      source: null,
    });

    steps.forEach((s, i) => {
      const here = at(s, i);
      const o = obs(s);
      rests.push({
        flowId: doc.id,
        lane: s.agent!,
        index: s.index,
        state: s.state,
        x0: here.x0,
        x1: here.x1,
        result: s.result,
        digest: o.digest,
        source: o.source,
      });

      const next = steps[i + 1];
      if (!next) return;
      const no = obs(next);
      let state: WireState;
      let because: string;
      if (!o.digest && s.series === undefined) {
        state = "unknown";
        because =
          `no attempt of step ${s.index} closed with an observation, so nothing about ` +
          `this handoff has been recorded — which is not the same as nothing having ` +
          `been carried`;
      } else if (next.ignoredInput) {
        state = "ignored";
        because =
          `step ${next.index} carried ${next.ignoredInput.carried} of the ` +
          `${next.ignoredInput.inputTokens} it was given` +
          (next.ignoredInput.note ? ` (${next.ignoredInput.note})` : "");
      } else if (next.state === "blocked" || next.state === "failed") {
        // The same split `bus.ts` makes, and for the same reason: a step that ran
        // and came out against its threshold produced a negative *result*; one
        // held with nothing recorded produced no result at all.
        state = no.digest ? "blocked" : "open";
        because = no.digest
          ? `step ${next.index} is ${next.state}: it ran, recorded ${no.digest} and did not pass`
          : `step ${next.index} is ${next.state} and recorded nothing: it has not reached a verdict`;
      } else {
        state = "carried";
        because = o.source
          ? `step ${s.index} closed with an observation from ${o.source}`
          : `step ${s.index} closed with an observation`;
      }

      /**
       * The crossing runs from when this step closed to when the next opened.
       *
       * That is the honest span of a handoff, and in `clock` it is often the
       * most interesting width on the page: a long crossing is time a thought
       * spent going nowhere. When the two steps touch exactly, the crossing
       * borrows `MIN_CROSS_MS` so the rope has room to bend.
       */
      const there = at(next, i + 1);
      const min = MIN_CROSS_MS / 60_000;
      const cx0 = here.x1;
      const cx1 = basis === "clock" ? Math.max(there.x0, cx0 + min) : here.x1 + (1 - REST);
      crosses.push({
        flowId: doc.id,
        from: s.agent!,
        to: next.agent!,
        fromIndex: s.index,
        toIndex: next.index,
        x0: cx0,
        x1: cx1,
        state,
        because,
        digest: o.digest,
        source: o.source,
      });
    });

    /**
     * A finished flow comes home.
     *
     * Only when `done`. A rope that stops in an agent's lane is a thought nobody
     * ever received, and drawing every rope as returning would be the surface
     * telling a reader that work was delivered when it was abandoned.
     */
    const last = steps[steps.length - 1]!;
    const delivered = doc.state === "done";
    if (delivered) {
      const lastRest = rests[rests.length - 1]!;
      crosses.push({
        flowId: doc.id,
        from: last.agent!,
        to: "@human",
        fromIndex: last.index,
        toIndex: -1,
        x0: lastRest.x1,
        /**
         * A full slot, not the width of an ordinary handoff.
         *
         * The return climbs the whole height of the lane stack, and at a
         * handoff's width that is a near-vertical line — which reads as a wall
         * the rope hit, the exact opposite of what it is. Coming home is the one
         * crossing that has to look like arriving.
         */
        x1: lastRest.x1 + 1.15,
        state: "carried",
        because: "the flow finished, so it came back to the person who asked",
        digest: obs(last).digest,
        source: obs(last).source,
      });
    }

    const end = crosses[crosses.length - 1]!.x1;
    if (end > widest) widest = end;
    threads.push({ flowId: doc.id, title: doc.title, state: doc.state, ordinal, rests, crosses, delivered });
  });

  return {
    lanes,
    threads,
    span: widest,
    basis,
    t0: basis === "clock" ? t0 : 0,
    t1: basis === "clock" ? t1 : 0,
  };
}

/**
 * Where two threads share a lane at the same moment.
 *
 * The picture the desk could not draw: one agent holding two thoughts at once.
 * It rendered that as a small multiplier badge on a cube, which is a *count*
 * where the reader wanted a *collision*.
 */
export function contentions(w: ThreadWorld): Array<{ lane: string; x0: number; x1: number; flows: string[] }> {
  const out: Array<{ lane: string; x0: number; x1: number; flows: string[] }> = [];
  const byLane = new Map<string, Rest[]>();
  for (const t of w.threads)
    for (const r of t.rests) byLane.set(r.lane, [...(byLane.get(r.lane) ?? []), r]);

  for (const [lane, rests] of byLane) {
    for (let i = 0; i < rests.length; i += 1)
      for (let j = i + 1; j < rests.length; j += 1) {
        const a = rests[i]!;
        const b = rests[j]!;
        if (a.flowId === b.flowId) continue;
        const x0 = Math.max(a.x0, b.x0);
        const x1 = Math.min(a.x1, b.x1);
        if (x1 > x0) out.push({ lane, x0, x1, flows: [a.flowId, b.flowId] });
      }
  }
  return out;
}
