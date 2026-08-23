/**
 * The Inspector, and the agent you can put on things.
 *
 * ## Why one panel
 *
 * NeXT had exactly one inspector. Click a different object, the panel becomes
 * about that object; there was never a question of which panel to read. The desk
 * had grown a *Selected* panel that printed facts about documents and a
 * different treatment for everything else, which meant the answer to "what is
 * this thing" depended on what kind of thing it was. One panel, one binding.
 *
 * ## The two positions, and why the second one is the point
 *
 * **Read it** returns the object's real fields. For an agent, its markdown; for
 * a packet, the bytes that moved; for a gate, the measured number and the
 * tolerance next to each other. You do the looking.
 *
 * **Ask an agent** hands the same object to `INSPECTOR`, a system agent with no
 * privileges the others do not have, and returns its finding. That is the
 * gesture `doc/15` phase 5 specified and nobody built: dropping an agent on a
 * thing declares a relationship, and the relationship gets written down.
 *
 * ## The rule that makes the second position worth having
 *
 * > A finding cites the artifact it read, and the citation is an address.
 *
 * Every `Finding` below carries `cites`. When the inspector has nothing to read,
 * it is required to return `unknown` — not a hedge, not a plausible reading —
 * and the desk draws that differently from an answer. A model's judgement is a
 * claim; a claim needs an address (doc/19 §4). This module is that sentence
 * with a type signature.
 *
 * ## What is simulated and what is not
 *
 * In the published demo there is no model, so the *wording* below is written
 * here rather than generated. What is not simulated is the citation: `cites`
 * resolves to an artifact that exists in the repository, and
 * `scripts/check-demo-provenance.py` fails the build if the number in the
 * sentence is not the number in the file.
 *
 * ## Why the wording is a string
 *
 * Same reason as [bus.ts](bus.ts) and [creatures.ts](creatures.ts): it runs in
 * the page, which has no build step. The text below is what the browser is
 * served, and `test/inspector.test.ts` executes that text rather than a
 * TypeScript paraphrase of it — otherwise the test can go on passing against a
 * rule the page no longer has, which is the one failure a test for honesty
 * cannot afford.
 *
 * Pure. No fetch, no clock, no DOM.
 */
import type { BusGraph, Wire } from "./bus.ts";

/** Anything on the desk can be inspected. That is the whole inversion. */
export type Subject =
  | { kind: "agent"; name: string }
  | { kind: "wire"; id: string }
  | { kind: "packet"; wireId: string }
  | { kind: "flow"; id: string }
  | { kind: "gate"; id: string };

/** One row of the object's real state. Values are shown verbatim, never rounded. */
export interface Field {
  label: string;
  value: string;
  /** The file this field was read out of, when it came from one. */
  from?: string;
}

/**
 * What an agent asked to look at something came back with.
 *
 * `verdict` is deliberately three-valued. A two-valued inspector has to call
 * "I could not tell" something, and whichever of the two it picks is a lie.
 */
export interface Finding {
  verdict: "ok" | "problem" | "unknown";
  says: string;
  /**
   * The artifacts read, as addresses. Empty **only** when `verdict` is
   * `unknown`; `assertCited` enforces that, and the desk calls it.
   */
  cites: Array<{ label: string; at: string }>;
  /** What asking cost. Stated before the button is pressed, as everywhere else. */
  cost: string;
}

export interface Inspection {
  title: string;
  /** What kind of object this is, in the words the desk uses for it. */
  kind: string;
  fields: Field[];
}

/**
 * The rules, as the page runs them.
 *
 * `assertCited` throws rather than warns. An inspector that produces a confident
 * sentence with no address is the exact failure this project argues against, and
 * shipping one *in the demo for that argument* would be the most expensive drift
 * available. A thrown error in the client is loud; a lint is not.
 */
export const INSPECTOR_JS = String.raw`
function assertCited(f) {
  if (f.verdict !== 'unknown' && f.cites.length === 0) {
    throw new Error(
      'the inspector claimed "' + f.says + '" and cited nothing; ' +
      'a verdict without an address must be reported as unknown',
    );
  }
  return f;
}

/** The fields of a wire: what moved, where from, and whether it landed. */
function inspectWire(w) {
  const fields = [
    { label: 'from', value: w.from + ' · step ' + w.fromIndex },
    { label: 'to', value: w.to + ' · step ' + w.toIndex },
    { label: 'in flow', value: w.flowTitle },
    { label: 'state', value: w.state },
    { label: 'because', value: w.because },
  ];
  if (w.packet) {
    if (w.packet.digest)
      fields.push({ label: 'observation', value: w.packet.digest, from: w.packet.source || undefined });
    if (w.packet.runId) fields.push({ label: 'run', value: w.packet.runId });
    if (w.packet.series) fields.push({ label: 'carried', value: w.packet.series.length + ' number(s)' });
    if (w.packet.result) fields.push({ label: 'said', value: w.packet.result });
  } else {
    // 'nothing recorded' and 'nothing moved' are different claims about the
    // world, and only the first one is supported. Printing the second would be
    // inventing a negative result.
    fields.push({
      label: 'carried',
      value: 'nothing recorded — this is not a claim that nothing moved',
    });
  }
  return { title: w.from + ' → ' + w.to, kind: 'wire', fields: fields };
}

/**
 * The finding an inspector agent returns for a wire.
 *
 * Note what it does *not* do on an unknown wire: it does not say the hop was
 * fine, and it does not say it failed. It says the store has no record — the
 * only true sentence available, and the one a confident assistant would not
 * produce.
 */
function inspectWireWithAgent(w) {
  const cites = (w.packet && w.packet.source)
    ? [{ label: 'observation ' + String(w.packet.digest || '').slice(0, 12), at: w.packet.source }]
    : [];

  /**
   * Where a claim about the *receiving* step comes from.
   *
   * A hop can be blocked or ignored on evidence that is not the packet: the
   * receiving step's own record in the flow store. The first version of this
   * returned those verdicts citing only the packet's source, so a hop whose
   * producing step recorded a digest with no source threw assertCited in the
   * page — the guard firing correctly on a real gap rather than on a mistake in
   * itself.
   *
   * The fix is not to relax the guard. It is to cite the record the claim was
   * actually read out of, which is a real address the desk can resolve: the
   * step's line in the flow. A verdict read from the store cites the store.
   */
  const fromStore = { label: 'step ' + w.toIndex + ' in the flow store', at: 'flow:' + w.flowId + '#step-' + w.toIndex };
  const stateCites = cites.length ? cites.concat([fromStore]) : [fromStore];

  if (w.state === 'ignored')
    return assertCited({
      verdict: 'problem',
      says:
        'Bytes arrived on this hop and ' + w.to + ' used none of them. ' + w.because +
        '. Every strip on this flow is green; this is the instrument that disagrees with them.',
      cites: stateCites,
      cost: 'one read of the recorded observation · no model call',
    });
  if (w.state === 'blocked')
    return assertCited({
      verdict: 'problem',
      says: 'The packet reached ' + w.to + ' and the step did not finish. ' + w.because + '.',
      cites: stateCites,
      cost: 'one read of the recorded observation · no model call',
    });
  if (w.state === 'carried')
    return assertCited({
      verdict: 'ok',
      says: w.from + ' closed with an observation and ' + w.to + ' read it. ' + w.because + '.',
      cites: cites,
      cost: 'one read of the recorded observation · no model call',
    });
  return assertCited({
    verdict: 'unknown',
    says:
      'Nothing was recorded for this hop, so there is nothing to read. ' +
      'That is not a pass: it is the absence of evidence either way.',
    cites: [],
    cost: 'nothing to read · no model call',
  });
}

/** The fields of an agent: what it is, what it may do, and what it is doing. */
function inspectAgent(a, load) {
  const fields = [
    { label: 'is for', value: a.description },
    { label: 'tools', value: a.tools.length ? a.tools.join(', ') : 'none declared' },
    { label: 'defined in', value: 'agents/' + a.name + '.md', from: 'agents/' + a.name + '.md' },
  ];
  if (a.child) fields.push({ label: 'declared as', value: 'a subagent of another agent' });
  if (a.missing)
    fields.push({
      label: 'file',
      value: 'named in a subagents: list with no markdown behind it — cannot be run',
    });
  fields.push({
    label: 'traffic',
    value: load
      ? load.sent + ' sent · ' + load.received + ' received' +
        (load.running ? ' · ' + load.running + ' running now' : '')
      : 'no hops in this scope',
  });
  return { title: a.name, kind: 'agent', fields: fields };
}

function inspectAgentWithAgent(a, load) {
  if (a.missing)
    return assertCited({
      verdict: 'problem',
      says: a.name + ' is named as a subagent and has no file. Anything composed against it ' +
        'will resolve to nothing at run time.',
      cites: [{ label: 'the declaration', at: 'agents/' + a.name + '.md' }],
      cost: 'one read of the agent directory · no model call',
    });
  if (!load || load.sent + load.received === 0)
    return assertCited({
      verdict: 'unknown',
      says: a.name + ' has no hops in this scope, so there is nothing recorded about how it ' +
        'behaves here.',
      cites: [],
      cost: 'nothing to read · no model call',
    });
  return assertCited({
    verdict: 'ok',
    says: a.name + ' has ' + load.sent + ' outgoing and ' + load.received + ' incoming hop(s) here, ' +
      'and may use: ' + (a.tools.join(', ') || 'no declared tools') + '.',
    cites: [{ label: 'agents/' + a.name + '.md', at: 'agents/' + a.name + '.md' }],
    cost: 'one read of the agent file · no model call',
  });
}

/**
 * A gate: both halves, unrounded, side by side.
 *
 * The entire difference between a gate and an opinion is that a reader can see
 * the measurement and the threshold at once and do the comparison themselves. A
 * panel that showed only the verdict would have removed exactly that.
 */
/**
 * Ask an agent about a whole flow.
 *
 * This is the one the tour ends on, so it is the one most worth getting right.
 *
 * The ranking below is not cosmetic. A flow with an *ignored* hop and a *blocked*
 * hop has two problems, and a summary that reports whichever came first would be
 * choosing by position rather than by severity. *ignored* outranks *blocked*
 * because a step that failed is visible in the strip and a step that succeeded
 * while carrying nothing is not: the second is the failure nobody would find.
 *
 * And when every hop carried but some were never recorded, it says so instead of
 * saying the flow is fine. "14 of 16 carried" and "14 of 16 carried, 2
 * unrecorded" are different claims about the same flow.
 */
function inspectFlowWithAgent(flowId, title, wires) {
  const mine = wires.filter((w) => w.flowId === flowId);
  if (mine.length === 0)
    return assertCited({
      verdict: 'unknown',
      says: '"' + title + '" has no handoffs between agents, so there is no traffic to read.',
      cites: [],
      cost: 'nothing to read · no model call',
    });

  const worst = mine.filter((w) => w.state === 'ignored')[0]
    || mine.filter((w) => w.state === 'blocked')[0];
  if (worst) return inspectWireWithAgent(worst);

  const unknown = mine.filter((w) => w.state === 'unknown');
  const carried = mine.filter((w) => w.state === 'carried');
  if (unknown.length)
    return assertCited({
      verdict: 'unknown',
      says:
        carried.length + ' of ' + mine.length + ' hop(s) in "' + title + '" carried something ' +
        'addressable, and ' + unknown.length + ' recorded nothing at all. I cannot tell you the ' +
        'flow is sound: I can only tell you about the ' + carried.length + ' I can read.',
      cites: [],
      cost: 'one read of the recorded observations · no model call',
    });

  const cited = carried.filter((w) => w.packet && w.packet.source);
  return assertCited({
    verdict: 'ok',
    says:
      'All ' + mine.length + ' hop(s) in "' + title + '" closed with an observation and the ' +
      'receiving step used it. That is a statement about the handoffs, not about whether the ' +
      'answer is right — a gate says that, and this does not.',
    cites: cited.length
      ? [{ label: 'the recorded observations', at: cited[0].packet.source }]
      : [],
    cost: 'one read per hop · no model call',
  });
}

function inspectGate(g) {
  return {
    title: g.id,
    kind: 'gate',
    fields: [
      { label: 'checks', value: g.what },
      { label: 'measured', value: g.measured, from: g.report },
      { label: 'declared before the run', value: g.tolerance, from: g.report },
      { label: 'verdict', value: g.passed ? 'within tolerance' : 'outside tolerance' },
      { label: 'report', value: g.report, from: g.report },
    ],
  };
}

function inspectGateWithAgent(g) {
  return assertCited({
    verdict: g.passed ? 'ok' : 'problem',
    says: g.id + ' measured ' + g.measured + ' against ' + g.tolerance + '. ' +
      (g.passed
        ? 'Within tolerance, so it does not hold the freeze.'
        : 'Outside tolerance, so this result cannot freeze.'),
    cites: [{ label: g.report.split('/').pop() || g.report, at: g.report }],
    cost: 'one read of the gate report · no model call',
  });
}
`;

/**
 * A gate: the number that was measured, and the number that was declared.
 *
 * `report` is required. A gate with no report is prose.
 */
export interface GateFact {
  id: string;
  what: string;
  measured: string;
  tolerance: string;
  passed: boolean;
  report: string;
}

interface AgentFact {
  name: string;
  description: string;
  tools: string[];
  child: boolean;
  missing: boolean;
}

type Load = BusGraph["load"][string] | undefined;

/** The shipped rules, executed once. Not a second copy — the same text. */
const RULES = new Function(
  INSPECTOR_JS +
    "\n;return { assertCited, inspectWire, inspectWireWithAgent, inspectAgent," +
    " inspectAgentWithAgent, inspectFlowWithAgent, inspectGate, inspectGateWithAgent };",
)() as {
  assertCited: (f: Finding) => Finding;
  inspectWire: (w: Wire) => Inspection;
  inspectWireWithAgent: (w: Wire) => Finding;
  inspectAgent: (a: AgentFact, load: Load) => Inspection;
  inspectAgentWithAgent: (a: Pick<AgentFact, "name" | "tools" | "missing">, load: Load) => Finding;
  inspectFlowWithAgent: (flowId: string, title: string, wires: Wire[]) => Finding;
  inspectGate: (g: GateFact) => Inspection;
  inspectGateWithAgent: (g: GateFact) => Finding;
};

export const assertCited = RULES.assertCited;
export const inspectWire = RULES.inspectWire;
export const inspectWireWithAgent = RULES.inspectWireWithAgent;
export const inspectAgent = RULES.inspectAgent;
export const inspectAgentWithAgent = RULES.inspectAgentWithAgent;
export const inspectFlowWithAgent = RULES.inspectFlowWithAgent;
export const inspectGate = RULES.inspectGate;
export const inspectGateWithAgent = RULES.inspectGateWithAgent;
