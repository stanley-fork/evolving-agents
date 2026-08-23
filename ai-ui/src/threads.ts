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
}

/** How much of a step's slot is rest, leaving the remainder for the crossing. */
const REST = 0.68;

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

  const threads: Thread[] = [];
  let span = 0;

  docs.forEach((doc, ordinal) => {
    const steps = doc.trace.steps.filter((s) => s.agent).sort((a, b) => a.index - b.index);
    if (steps.length === 0) return;
    for (const s of steps) laneOf(s.agent!);

    const rests: Rest[] = [];
    const crosses: Cross[] = [];

    // The thought starts with the human and crosses into the first agent.
    const first = steps[0]!;
    crosses.push({
      flowId: doc.id,
      from: "@human",
      to: first.agent!,
      fromIndex: -1,
      toIndex: first.index,
      x0: 0,
      x1: 1 - REST,
      // Always `carried`: the human posing the question is the one handoff in
      // the system that is not in question. Marking it `unknown` because no
      // observation was recorded would be pedantry pretending to be rigour.
      state: "carried",
      because: "the question came from a person",
      digest: null,
      source: null,
    });

    steps.forEach((s, i) => {
      const base = i + (1 - REST);
      const o = obs(s);
      rests.push({
        flowId: doc.id,
        lane: s.agent!,
        index: s.index,
        state: s.state,
        x0: base,
        x1: base + REST,
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

      crosses.push({
        flowId: doc.id,
        from: s.agent!,
        to: next.agent!,
        fromIndex: s.index,
        toIndex: next.index,
        x0: base + REST,
        x1: base + 1,
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
    if (end > span) span = end;
    threads.push({ flowId: doc.id, title: doc.title, state: doc.state, ordinal, rests, crosses, delivered });
  });

  return { lanes, threads, span };
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
