/**
 * The bundle: many strands coiled around one axis, and depth is attention.
 *
 * ## The geometry, and what each dimension is for
 *
 * `threads.ts` lays flows out as swimlanes — X is time, Y is which agent holds
 * the thought. That is legible and it does not scale: eleven agents is eleven
 * rows, and a hundred flows is a hairball no zoom fixes, because every flow needs
 * its own vertical room.
 *
 * This is the other arrangement. **One axis, along X, and it is time.** Every
 * flow is a *strand* wound around that axis at its own phase, so they occupy the
 * same band of screen and separate in **depth** instead of in height:
 *
 * ```
 *   θ(t) = φ_flow + TWIST·(t − now) + rotation
 *   y    = cy + R·sin θ          (where it sits in the band)
 *   z    = cos θ  ∈ [−1, 1]      (how far forward it is)
 * ```
 *
 * A strand at `z = 1` is at the front: bright, thick, in focus. At `z = −1` it is
 * behind the bundle: dim, thin, and drawn first so the front occludes it. Almost
 * a 3D object in a 2D scene, and the reason to want one is that **rotation is a
 * way of paying attention**: turn the bundle and a different strand comes
 * forward, without anything having to move out of the way.
 *
 * ## Two motions, two meanings, and neither is decoration
 *
 * The rule the whole surface is built on is that everything that moves is a
 * measurement. A bundle that spins gently forever would break it on the first
 * frame. So there are exactly two:
 *
 * **The content drifts left, always.** `now` is pinned to the right edge and the
 * clock is genuinely running, so a thing recorded twenty-two minutes ago is
 * twenty-three minutes ago a minute later, and it moves. That drift means *time
 * is passing*, which is always true — and it is the only motion that is
 * unconditional. Because the strands twist along X, the drift also makes the
 * bundle appear to rotate, which is free: it is the geometry, not an animation.
 *
 * **Light travels a strand only where a step is open right now.** That means
 * *work is happening*, which is usually false, and when it is false nothing on
 * the strand moves.
 *
 * ## Attention has to be measured or the rotation is a lie
 *
 * Bringing a strand to the front is a claim — *this is what deserves looking at*
 * — and this repository's whole argument is that a claim needs an address. So
 * `attentionOf` ranks by things that are recorded, in an order that is stated
 * rather than tuned, and every entry carries the reason and the address it was
 * read from. A rotation the surface cannot justify is a rotation it must not
 * make.
 *
 * Pure. No fetch, no clock, no DOM.
 */
import type { ThreadWorld, Thread } from "./threads.ts";

/** Radians of twist per minute of the axis. The pitch of the bundle. */
export const TWIST = (Math.PI * 2) / 90;

/**
 * How much the bundle coils, in radians per minute.
 *
 * A *rendering* parameter, not a property of the data, and the reason it is one
 * is the sampling argument this repository already makes about zoom
 * ([zoom.ts](zoom.ts), doc/04): a viewer sampling at one rate cannot faithfully
 * observe change faster than half of it. At three days across, one turn every
 * ninety minutes is forty-eight turns in a thousand pixels — the coil is finer
 * than the pixels drawing it, and what a reader sees is an artefact of the
 * sampling rather than the shape.
 *
 * So the renderer passes `twist: 0` once a turn is narrower than a few dozen
 * pixels. The bundle stops coiling and becomes what it is at that distance: a
 * set of parallel strands, still ordered in depth, still turnable. It does not
 * draw a coil it cannot draw.
 */
export interface Sample {
  /** Minutes on the world's axis. */
  t: number;
  /** Offset from the bundle's centre line, in units of the bundle radius. */
  y: number;
  /** Depth: 1 is fully forward, −1 fully behind. */
  z: number;
}

export interface StrandSegment {
  flowId: string;
  kind: "rest" | "cross";
  /** The step this rests on, or the receiving step of the handoff. */
  index: number;
  /** For a cross, what may be claimed about it; for a rest, the step's state. */
  state: string;
  from: string;
  to: string;
  t0: number;
  t1: number;
  samples: Sample[];
  /** Mean depth, for painter's-algorithm sorting. */
  z: number;
  digest: string | null;
  source: string | null;
  because: string;
  title: string;
  ordinal: number;
}

/**
 * A body riding a strand — the agent that is holding the thought.
 *
 * Drawn the way a polymerase is drawn on a strand it is copying: clamped to the
 * curve, at the place along it that corresponds to now, moving as the strand
 * drifts. The desk drew an agent as a box that a flow was attached *to*; here the
 * agent is attached to the flow, which is the right way round — an agent holds a
 * task for a while and then hands it on, and it is the task that persists.
 */
export interface Body {
  flowId: string;
  /** The agent's name, or `@human`. For a packet, the agent it is travelling to. */
  name: string;
  /**
   * `agent` — somebody is holding it. `human` — the person at either end of the
   * strand. `packet` — it is between two agents and nobody is holding it, which
   * is a different statement and must look like one.
   */
  kind: "agent" | "human" | "packet";
  t: number;
  y: number;
  z: number;
  state: string;
  ordinal: number;
}

/** Why the bundle is turned where it is turned. Every field is checkable. */
export interface Attention {
  flowId: string;
  title: string;
  ordinal: number;
  kind: "running" | "ignored" | "blocked" | "open" | "settled";
  /** One clause, in the surface's own vocabulary. */
  reason: string;
  /**
   * Where the claim was read from. `null` **only** for `settled` and `open`
   * without a record — the same rule the inspector lives under.
   */
  at: string | null;
  /** Higher comes forward. Derived from `kind` and never hand-tuned per flow. */
  rank: number;
}

/**
 * The rules, as the page runs them.
 *
 * Same arrangement as [bus.ts](bus.ts) and [inspector.ts](inspector.ts), for the
 * same reason: this runs in a page with no build step, and the test executes the
 * exact text the browser is served rather than a TypeScript paraphrase of it. A
 * test that paraphrases can go on passing against a rule the page no longer has.
 *
 * `RANK` is a table rather than a score, because a score invites weights and
 * weights invite tuning until the demo looks good. What is claimed is only that
 * these five kinds are in this order, which is something somebody can disagree
 * with in words. `ignored` outranks `blocked` for the reason `inspector.ts`
 * gives: a step that failed is visible in the strip, and a step that succeeded
 * while carrying nothing is not.
 *
 * Each kind is looked for in *handoffs* first and then in *steps*, because for
 * a long time it was only looked for in handoffs — and a handoff exists between
 * two steps, so a one-step flow had nothing that could carry the news. See the
 * two rest passes below.
 */
export const HELIX_JS = String.raw`
var HELIX_TWIST = (Math.PI * 2) / 90;
var HELIX_RANK = { running: 4, ignored: 3, blocked: 2, open: 1, settled: 0 };

function phaseOf(ordinal, count, t, now, rotation, twist) {
  var phi = count > 0 ? (Math.PI * 2 * ordinal) / count : 0;
  var k = twist === undefined ? HELIX_TWIST : twist;
  var theta = phi + k * (t - now) + rotation;
  return { t: t, y: Math.sin(theta), z: Math.cos(theta) };
}

function rotationFor(ordinal, count, now) {
  var phi = count > 0 ? (Math.PI * 2 * ordinal) / count : 0;
  var r = -phi;
  return ((r % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
}

function bundleOf(world, opts) {
  var n = world.threads.length;
  var base = opts.steps === undefined ? 14 : opts.steps;
  var out = [];

  /**
   * Sample the angle, not the segment.
   *
   * A fixed count per segment is a fixed count per *step*, and a step is not a
   * fixed amount of winding: a forty-minute rest on a fifteen-minute pitch
   * sweeps nearly three turns, and fourteen points across three turns is a
   * sawtooth — the drawing showed the sampling rate, not the coil. So the count
   * comes from how much angle this segment actually covers, at sixteen points a
   * turn, floored at the old count for segments that barely turn at all and
   * capped so one long stretch cannot fill the document with path data.
   */
  var sample = function (ordinal, a, b) {
    var swept = Math.abs((opts.twist === undefined ? 0 : opts.twist) * (b - a));
    var per = Math.max(base, Math.min(600, Math.ceil((swept / (Math.PI * 2)) * 16)));
    var s = [];
    for (var i = 0; i <= per; i += 1)
      s.push(phaseOf(ordinal, n, a + ((b - a) * i) / per, opts.now, opts.rotation, opts.twist));
    return s;
  };
  var meanZ = function (s) {
    var m = 0;
    for (var i = 0; i < s.length; i += 1) m += s[i].z;
    return m / s.length;
  };

  for (var ti = 0; ti < world.threads.length; ti += 1) {
    var th = world.threads[ti];
    for (var ri = 0; ri < th.rests.length; ri += 1) {
      var r = th.rests[ri];
      if (r.x1 < opts.t0 || r.x0 > opts.t1) continue;
      var rs = sample(th.ordinal, r.x0, r.x1);
      out.push({
        flowId: th.flowId, kind: 'rest', index: r.index, state: r.state,
        from: r.lane, to: r.lane, t0: r.x0, t1: r.x1, samples: rs, z: meanZ(rs),
        digest: r.digest, source: r.source, because: '', title: th.title, ordinal: th.ordinal,
      });
    }
    for (var ci = 0; ci < th.crosses.length; ci += 1) {
      var c = th.crosses[ci];
      if (c.x1 < opts.t0 || c.x0 > opts.t1) continue;
      var cs = sample(th.ordinal, c.x0, c.x1);
      out.push({
        flowId: th.flowId, kind: 'cross', index: c.toIndex, state: c.state,
        from: c.from, to: c.to, t0: c.x0, t1: c.x1, samples: cs, z: meanZ(cs),
        digest: c.digest, source: c.source, because: c.because, title: th.title, ordinal: th.ordinal,
      });
    }
  }
  return out.sort(function (a, b) { return a.z - b.z; });
}

function bodiesAt(world, t, rotation) {
  var n = world.threads.length;
  var out = [];
  for (var i = 0; i < world.threads.length; i += 1) {
    var th = world.threads[i];
    var r = null, c = null, k;
    for (k = 0; k < th.rests.length; k += 1)
      if (t >= th.rests[k].x0 && t <= th.rests[k].x1) { r = th.rests[k]; break; }
    if (r) {
      var p = phaseOf(th.ordinal, n, t, t, rotation);
      out.push({ flowId: th.flowId, name: r.lane, kind: 'agent', t: t, y: p.y, z: p.z,
                 state: r.state, ordinal: th.ordinal });
      continue;
    }
    for (k = 0; k < th.crosses.length; k += 1) {
      var x = th.crosses[k];
      if (t >= x.x0 && t <= x.x1) { c = x; break; }
    }
    if (c) {
      var q = phaseOf(th.ordinal, n, t, t, rotation);
      /**
       * Mid-handoff, so nobody is holding it.
       *
       * A person at either end is a body — they are the one who asked, or the
       * one it came back to. Between two agents there is no body, because
       * nothing is being held: what is there is the thing in transit, and that
       * is a packet. Drawing an agent here would say somebody was working when
       * the record says the work had left one and not yet reached the other.
       */
      var human = c.from === '@human' || c.to === '@human';
      out.push({
        flowId: th.flowId,
        name: human ? '@human' : c.to,
        kind: human ? 'human' : 'packet',
        t: t, y: q.y, z: q.z,
        state: human ? (c.from === '@human' ? 'asking' : 'receiving') : c.state,
        ordinal: th.ordinal,
      });
    }
  }
  return out.sort(function (a, b) { return a.z - b.z; });
}

function helixRank(th) {
  var step = function (i) { return 'flow:' + th.flowId + '#step-' + i; };
  var base = { flowId: th.flowId, title: th.title, ordinal: th.ordinal };
  var i;

  for (i = 0; i < th.rests.length; i += 1)
    if (th.rests[i].state === 'running') {
      var live = th.rests[i];
      return Object.assign({}, base, { kind: 'running',
        reason: live.lane + ' has step ' + live.index + ' open right now',
        at: step(live.index), rank: HELIX_RANK.running });
    }
  for (i = 0; i < th.crosses.length; i += 1)
    if (th.crosses[i].state === 'ignored') {
      var ig = th.crosses[i];
      return Object.assign({}, base, { kind: 'ignored',
        reason: ig.to + ' used nothing it was given at step ' + ig.toIndex,
        at: ig.source || step(ig.toIndex), rank: HELIX_RANK.ignored });
    }
  for (i = 0; i < th.crosses.length; i += 1)
    if (th.crosses[i].state === 'blocked') {
      var bl = th.crosses[i];
      return Object.assign({}, base, { kind: 'blocked',
        reason: 'step ' + bl.toIndex + ' ran and did not pass',
        at: bl.source || step(bl.toIndex), rank: HELIX_RANK.blocked });
    }
  /**
   * A step that ran and did not pass, when no handoff carries the news.
   *
   * The passes above read *handoffs*, and a handoff only exists between two
   * steps. A flow whose last step failed, or a flow of one step, has nothing
   * downstream to carry that fact — so it fell through to 'settled' and the
   * panel said nothing was wrong. Split by observation, exactly as bus.ts does:
   * a step that recorded something and came out against its threshold produced
   * a real negative result.
   */
  for (i = 0; i < th.rests.length; i += 1) {
    var rb = th.rests[i];
    if ((rb.state === 'blocked' || rb.state === 'failed') && rb.digest) {
      return Object.assign({}, base, { kind: 'blocked',
        reason: 'step ' + rb.index + ' ran and did not pass',
        at: rb.source || step(rb.index), rank: HELIX_RANK.blocked });
    }
  }
  for (i = 0; i < th.crosses.length; i += 1)
    if (th.crosses[i].state === 'open') {
      var op = th.crosses[i];
      return Object.assign({}, base, { kind: 'open',
        reason: 'step ' + op.toIndex + ' is held and has reached no verdict',
        // No address: there is nothing recorded to point at, which is what makes
        // it open rather than failed.
        at: null, rank: HELIX_RANK.open });
    }
  /**
   * Work that was stated and cannot proceed, which is the case this missed.
   *
   * In the flow vocabulary a *step* whose state is 'blocked' is not a step that
   * failed: hemo's H1 is one step, state blocked, observation null, carrying the
   * note 'stated as open work, because a scope with nothing red in it reads as a
   * finished one'. Nothing ran. Nothing said no. With no second step there is no
   * handoff to notice it, so every pass above missed it and the panel reported
   * *stopped, and nothing is open* — which is precisely the sentence the author
   * of that step wrote it to prevent.
   *
   * Unlike an open handoff this one has an address: the step exists in the store
   * and says so. What it does not have is an observation, and that is what keeps
   * it open rather than failed.
   */
  for (i = 0; i < th.rests.length; i += 1) {
    var ro = th.rests[i];
    if (ro.state === 'blocked' || ro.state === 'failed') {
      return Object.assign({}, base, { kind: 'open',
        reason: 'step ' + ro.index + ' is stated as open work and has reached no verdict',
        at: step(ro.index), rank: HELIX_RANK.open });
    }
  }
  return Object.assign({}, base, { kind: 'settled',
    reason: th.delivered ? 'finished and came back' : 'stopped, and nothing is open',
    at: null, rank: HELIX_RANK.settled });
}

function attentionOf(world) {
  var out = [];
  for (var i = 0; i < world.threads.length; i += 1) out.push(helixRank(world.threads[i]));
  return out.sort(function (a, b) { return b.rank - a.rank || a.ordinal - b.ordinal; });
}

/**
 * Refuse an attention claim that says something and cites nothing.
 *
 * The same guard inspector.ts runs, applied to the reason the bundle turned.
 * 'open' and 'settled' may cite nothing, because both are statements about an
 * absence of record.
 */
function assertJustified(a) {
  if (a.kind !== 'open' && a.kind !== 'settled' && !a.at)
    throw new Error(
      'the bundle turned to "' + a.title + '" because "' + a.reason + '" and cited nothing; ' +
      'a rotation the surface cannot justify is a rotation it must not make',
    );
  return a;
}
`;

/** The shipped rules, executed once. Not a second copy — the same text. */
const RULES = new Function(
  HELIX_JS +
    "\n;return { phaseOf, rotationFor, bundleOf, bodiesAt, attentionOf, assertJustified };",
)() as {
  phaseOf: (o: number, c: number, t: number, now: number, rot: number, twist?: number) => Sample;
  rotationFor: (o: number, c: number, now: number) => number;
  bundleOf: (
    w: ThreadWorld,
    o: { now: number; rotation: number; t0: number; t1: number; steps?: number; twist?: number },
  ) => StrandSegment[];
  bodiesAt: (w: ThreadWorld, t: number, rot: number) => Body[];
  attentionOf: (w: ThreadWorld) => Attention[];
  assertJustified: (a: Attention) => Attention;
};

export const phaseOf = RULES.phaseOf;
export const rotationFor = RULES.rotationFor;
export const bundleOf = RULES.bundleOf;
export const bodiesAt = RULES.bodiesAt;
export const attentionOf = RULES.attentionOf;
export const assertJustified = RULES.assertJustified;

/** Kept so the type import is load-bearing rather than decorative. */
export type HelixThread = Thread;
