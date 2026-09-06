/**
 * The flows of information, as things you can see and open.
 *
 * ## What this is for
 *
 * A flow, in the desk as it stood, was a list of steps in a panel. You could
 * read that a step ran and that the next one ran after it. You could not watch
 * anything *move*, and "the flow of information" was a phrase rather than a
 * picture.
 *
 * This module derives, from the same `DeskDoc`s the desk already has, the graph
 * the brief asks for: **agents are the nodes, and the handoffs between them are
 * wires with something travelling on them.**
 *
 * ## The rule that keeps it from being an animation
 *
 * > A wire carries a real artifact or it carries nothing.
 *
 * Every packet below is addressed: it has the `runId` and the observation digest
 * the flow store recorded for the attempt that produced it, and the desk can
 * open exactly that. When a hop produced nothing recorded, the wire is
 * `unknown` — dashed and labelled — and **never** drawn like a hop that carried.
 *
 * That distinction is not decoration. It is `freezeVerdict`'s split between
 * `blockers` and `unknown` (doc/19 §4): *did not run* is not *passed*, and a
 * picture that renders them the same way has thrown away the only property this
 * repository is arguing for.
 *
 * ## The fourth state, which is the interesting one
 *
 * `ignored` is a wire where bytes demonstrably arrived and the receiving step
 * used none of them — `trace.ts`'s `ignoredInput`, which `doc/13` is about. It
 * is the case a diagram of a pipeline can never show you, because on a diagram
 * an arrow that was drawn is an arrow that worked. Here the wire is drawn, the
 * packet is drawn, and the wire says the packet landed nowhere.
 *
 * ## Why the algorithm is a string
 *
 * It runs in the page, which has no build step and no imports — the constraint
 * [creatures.ts](creatures.ts) already lives under. Shipping it as source and
 * **executing that same source** in the test is the only arrangement where a
 * test cannot pass against a rule the page does not have. The TypeScript
 * `busOf` below is a wrapper around the shipped text, not a second copy of it.
 *
 * Pure. No fetch, no clock, no DOM.
 */
import type { FlowTrace, TraceStep } from "./trace.ts";

/** How a hop is drawn, and — more to the point — what may be claimed about it. */
export type WireState =
  /** Something moved and the receiving step used it. */
  | "carried"
  /** Something moved and the receiving step used none of it (doc/13). */
  | "ignored"
  /**
   * Something moved, the receiving step ran, and it did not pass.
   *
   * A *result*, and a negative one: the step recorded an observation, so there
   * is a number and a threshold somebody can go and look at.
   */
  | "blocked"
  /**
   * Something moved and the receiving step has not reached a verdict.
   *
   * Distinct from `blocked`, and the distinction is the point. A step that
   * measured something and came out against its threshold produced a negative
   * result. A step that is held with nothing recorded produced *no* result, and
   * drawing the two the same way asserts an outcome nobody has.
   */
  | "open"
  /**
   * Nothing at all was recorded about this hop.
   *
   * Not a pass, not a failure, and not the same as `open` either: `open` means
   * the packet is addressable and the verdict is pending, this means there is no
   * packet to point at.
   */
  | "unknown";

/** What travelled, addressed so the desk can open it rather than describe it. */
export interface Packet {
  /** `<flowId>:<fromIndex>-<toIndex>`; stable, so a selection survives a redraw. */
  id: string;
  /** The observation digest recorded when the producing attempt closed. */
  digest: string | null;
  /** Where the digest came from — the artifact's address. */
  source: string | null;
  /** The run that produced it, when the store recorded one. */
  runId: string | null;
  /** Numbers, when the producing step reported a shape rather than prose. */
  series?: number[];
  /** The producing step's own words, trimmed. Never the flow's goal (doc/11). */
  result: string | null;
}

export interface Wire {
  id: string;
  flowId: string;
  flowTitle: string;
  from: string;
  to: string;
  fromIndex: number;
  toIndex: number;
  state: WireState;
  /** Null exactly when `state` is `unknown`: nothing addressable moved. */
  packet: Packet | null;
  /**
   * Why this wire is in the state it is, in one clause, for the inspector.
   *
   * Written here rather than in the client so the sentence and the state that
   * produced it cannot disagree.
   */
  because: string;
}

export interface BusGraph {
  /** Every agent that appears in any step of any flow, in first-seen order. */
  nodes: string[];
  wires: Wire[];
  /** Per-agent counts, so the desk can size a node by how busy it actually is. */
  load: Record<string, { sent: number; received: number; running: number }>;
}

interface BusDoc {
  id: string;
  title: string;
  trace: FlowTrace;
}

/**
 * The rules, as the page runs them.
 *
 * `observationOf` takes the **last** attempt that closed with an observation
 * rather than the first: an attempt that failed and was retried produced nothing
 * the next step could have read, and addressing the packet to it would point the
 * inspector at the wrong bytes. `null` is what makes a wire `unknown` — it is
 * never what makes one green.
 *
 * Every wire is a *consecutive* handoff. Skipping over a step with no agent
 * would draw an arrow between two agents that never spoke, which is the tidy-
 * looking lie this module exists to refuse.
 */
export const BUS_JS = String.raw`
function busObservation(step) {
  for (let i = step.attempts.length - 1; i >= 0; i -= 1) {
    const a = step.attempts[i];
    if (a.digest) return { digest: a.digest, source: a.source, runId: a.runId };
  }
  return { digest: null, source: null, runId: null };
}

function busHops(trace) {
  return trace.steps.filter((s) => s.agent).slice().sort((a, b) => a.index - b.index);
}

function busOf(docs) {
  const nodes = [];
  const seen = new Set();
  const wires = [];
  const load = {};

  const note = (name) => {
    if (!seen.has(name)) { seen.add(name); nodes.push(name); }
    if (!load[name]) load[name] = { sent: 0, received: 0, running: 0 };
    return load[name];
  };

  for (const doc of docs) {
    const steps = busHops(doc.trace);
    for (const s of steps) {
      const l = note(s.agent);
      if (s.state === 'running') l.running += 1;
    }

    for (let i = 0; i + 1 < steps.length; i += 1) {
      const from = steps[i], to = steps[i + 1];
      const obs = busObservation(from);
      let state, because, packet = null;

      if (!obs.digest && from.series === undefined) {
        state = 'unknown';
        because =
          'no attempt of step ' + from.index + ' closed with an observation, so nothing ' +
          'about this handoff has been recorded — which is not the same as nothing ' +
          'having been carried';
      } else {
        packet = {
          id: doc.id + ':' + from.index + '-' + to.index,
          digest: obs.digest,
          source: obs.source,
          runId: obs.runId,
          result: from.result === undefined ? null : from.result,
        };
        if (from.series !== undefined) packet.series = from.series;

        if (to.ignoredInput) {
          state = 'ignored';
          because =
            'step ' + to.index + ' carried ' + to.ignoredInput.carried + ' of the ' +
            to.ignoredInput.inputTokens + ' it was given' +
            (to.ignoredInput.note ? ' (' + to.ignoredInput.note + ')' : '');
        } else if (to.state === 'blocked' || to.state === 'failed') {
          /**
           * A step that did not finish, split by whether it reached a verdict.
           *
           * These were one state and that was wrong. A step that ran, measured
           * something and came out against its threshold recorded an
           * observation: it produced a real negative result, and 'blocked' is
           * the right word. A step that is held with no observation at all
           * produced nothing — it never got to a verdict, and calling that a
           * problem asserts an outcome nobody has.
           *
           * The case that forced the distinction is hemo's A4 flow, whose entire
           * argument is that two runs on different machines have not disagreed
           * with each other — they have not been compared under conditions where
           * disagreement is defined. The desk reported that as a problem, which
           * is precisely the overclaim the flow exists to refuse.
           */
          const own = busObservation(to);
          if (own.digest) {
            state = 'blocked';
            because = 'step ' + to.index + ' is ' + to.state +
              ': it ran, recorded ' + own.digest + ' and did not pass';
          } else {
            state = 'open';
            because = 'step ' + to.index + ' is ' + to.state +
              ' and recorded nothing: it has not reached a verdict, which is not the ' +
              'same as reaching a negative one';
          }
        } else {
          state = 'carried';
          because = obs.source
            ? 'step ' + from.index + ' closed with an observation from ' + obs.source
            : 'step ' + from.index + ' closed with an observation';
        }
      }

      load[from.agent].sent += 1;
      load[to.agent].received += 1;

      wires.push({
        id: doc.id + ':' + from.index + '->' + to.index,
        flowId: doc.id,
        flowTitle: doc.title,
        from: from.agent,
        to: to.agent,
        fromIndex: from.index,
        toIndex: to.index,
        state: state,
        packet: packet,
        because: because,
      });
    }
  }

  return { nodes: nodes, wires: wires, load: load };
}

/**
 * What may be said about a graph in one line, without overclaiming.
 *
 * Reports 'unrecorded' separately rather than folding it into a percentage.
 * '14 of 16 carried' and '14 of 16 carried, 2 unrecorded' are different
 * statements, and only the second one is true.
 */
function busSummary(g) {
  const n = (s) => g.wires.filter((w) => w.state === s).length;
  const parts = [g.nodes.length + ' agent(s)', g.wires.length + ' hop(s)'];
  if (n('carried')) parts.push(n('carried') + ' carried');
  if (n('ignored')) parts.push(n('ignored') + ' carried nothing forward');
  if (n('blocked')) parts.push(n('blocked') + ' into a step that ran and did not pass');
  if (n('open')) parts.push(n('open') + ' into a step still open');
  if (n('unknown')) parts.push(n('unknown') + ' unrecorded');
  return parts.join(' · ');
}
`;

interface BusDoc {
  id: string;
  title: string;
  trace: FlowTrace;
}

/**
 * The shipped rules, executed once.
 *
 * Not a re-implementation: this evaluates the exact text the page is served, so
 * a caller here and a caller in the browser cannot disagree about what a wire
 * is. `TraceStep` stays imported for the types alone.
 */
const RULES = new Function(
  BUS_JS + "\n;return { busOf: busOf, busSummary: busSummary };",
)() as {
  busOf: (docs: BusDoc[]) => BusGraph;
  busSummary: (g: BusGraph) => string;
};

export const busOf = (docs: BusDoc[]): BusGraph => RULES.busOf(docs);
export const busSummary = (g: BusGraph): string => RULES.busSummary(g);

/** Kept so the type import is load-bearing rather than decorative. */
export type BusStep = TraceStep;
