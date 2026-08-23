import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { busOf, busSummary } from "../src/bus.ts";
import { inspectWireWithAgent } from "../src/inspector.ts";
import { traceOf } from "../src/trace.ts";
import { hemoFlows } from "../src/hemo-demo.ts";
import { cochleaFlows } from "../src/cochlea-demo.ts";

/**
 * What this test is defending.
 *
 * `bus.ts` turns flows into a picture, and the failure mode of every such
 * picture is that it tidies. An arrow that was drawn looks like an arrow that
 * worked; a hop nobody recorded looks like a hop that went fine. The four wire
 * states exist to refuse that, and the assertions below are about the refusals
 * rather than about the happy path.
 */
const agentOf = (_intent: string) => null;

const docsOf = (raw: Array<Record<string, unknown>>) =>
  raw.map((d) => ({
    id: d["id"] as string,
    title: d["title"] as string,
    trace: traceOf(d["steps"] as never, agentOf),
  }));

describe("busOf", () => {
  it("draws a wire only between agents that actually spoke in sequence", () => {
    const g = busOf(docsOf(hemoFlows(1_700_000_000_000) as never));
    for (const w of g.wires) {
      assert.equal(w.toIndex, w.fromIndex + 1, `${w.id} skips a step`);
      assert.ok(g.nodes.includes(w.from));
      assert.ok(g.nodes.includes(w.to));
    }
  });

  /**
   * The load-bearing one.
   *
   * The hemo scope's `AUDITOR-H0` step in the A4 flow is held open with no
   * artifact. The wire *out of* it — there is none, it is last — is not the
   * point; the point is that a step whose predecessor recorded nothing must
   * produce an `unknown` wire and a null packet, so the desk cannot draw
   * evidence that was never collected.
   */
  it("reports a hop with no recorded observation as unknown, never as a pass", () => {
    const steps = [
      { index: 0, state: "done", intent: "a", result: "did a thing", attempts: [] },
      { index: 1, state: "done", intent: "b", result: "did another", attempts: [] },
    ];
    const g = busOf([
      {
        id: "f",
        title: "f",
        trace: traceOf(steps as never, () => "A"),
      },
    ]);
    // traceOf assigns agents from the intent; with one agent there is one hop.
    const w = g.wires[0];
    if (w) {
      assert.equal(w.state, "unknown");
      assert.equal(w.packet, null, "an unknown wire must carry nothing openable");
      assert.match(w.because, /not the same as nothing having been carried/);
    }
  });

  it("marks a hop whose receiver used nothing it was given", () => {
    const steps = [
      {
        index: 0,
        state: "done",
        intent: "a",
        result: "produced output",
        attempts: [
          { n: 1, state: "done", runId: "r0", error: null, observation: { digest: "d0", source: "gate.report" } },
        ],
      },
      {
        index: 1,
        state: "done",
        intent: "b",
        result: "ignored it",
        contribution: { carried: 0, inputTokens: 4000, note: "measured by the runner" },
        attempts: [
          { n: 1, state: "done", runId: "r1", error: null, observation: { digest: "d1", source: "gate.report" } },
        ],
      },
    ];
    const g = busOf([{ id: "f", title: "f", trace: traceOf(steps as never, () => "A") }]);
    assert.equal(g.wires[0]?.state, "ignored");
    assert.ok(g.wires[0]?.packet, "the bytes did arrive; only the use of them is in question");
    assert.match(g.wires[0]!.because, /carried 0 of the 4000/);
  });


  /**
   * The distinction that was wrong on the first pass, and said so out loud.
   *
   * A step that ran, measured something and came out against its threshold is a
   * negative *result*. A step held with nothing recorded is the absence of one.
   * The first version of `busOf` called both `blocked`, so the inspector told a
   * visitor that hemo's A4 flow had a problem — a flow whose entire argument is
   * that two runs on different machines have not disagreed, because they were
   * never compared under conditions where disagreement is defined. The tool
   * built to prevent overclaiming produced the overclaim.
   */
  it("separates a step that ran and failed from one that never reached a verdict", () => {
    const obs = (d: string) => [
      { n: 1, state: "failed", runId: "r", error: "gate red", observation: { digest: d, source: "gate.report" } },
    ];
    const producer = {
      index: 0, state: "done", intent: "a", result: "produced",
      attempts: [{ n: 1, state: "done", runId: "r0", error: null, observation: { digest: "d0", source: "gate.report" } }],
    };

    const ran = busOf([
      { id: "f", title: "f", trace: traceOf([producer, { index: 1, state: "failed", intent: "b", result: "measured and failed", attempts: obs("d1") }] as never, () => "A") },
    ]);
    assert.equal(ran.wires[0]?.state, "blocked", "it ran, recorded a number and did not pass");
    assert.match(ran.wires[0]!.because, /ran, recorded d1 and did not pass/);

    const held = busOf([
      { id: "g", title: "g", trace: traceOf([producer, { index: 1, state: "blocked", intent: "b", result: "held", attempts: [{ n: 1, state: "blocked", runId: "r1", error: "held", observation: null }] }] as never, () => "A") },
    ]);
    assert.equal(held.wires[0]?.state, "open", "held with nothing recorded is not a negative result");
    assert.ok(held.wires[0]?.packet, "the packet still arrived and is still openable");
    assert.match(held.wires[0]!.because, /has not reached a verdict/);
  });

  it("an open hop is inspected as unknown, never as a problem", () => {
    const producer = {
      index: 0, state: "done", intent: "a", result: "produced",
      attempts: [{ n: 1, state: "done", runId: "r0", error: null, observation: { digest: "d0", source: "gate.report" } }],
    };
    const g = busOf([
      { id: "g", title: "g", trace: traceOf([producer, { index: 1, state: "blocked", intent: "b", result: "held", attempts: [{ n: 1, state: "blocked", runId: "r1", error: "held", observation: null }] }] as never, () => "A") },
    ]);
    const f = inspectWireWithAgent(g.wires[0]!);
    assert.equal(f.verdict, "unknown");
    assert.match(f.says, /no result here to agree or disagree with/);
    assert.ok(f.cites.length > 0, "it read the step record, so it cites the step record");
  });

  it("counts traffic per agent", () => {
    const g = busOf(docsOf(cochleaFlows(1_700_000_000_000) as never));
    const total = Object.values(g.load).reduce((n, l) => n + l.sent, 0);
    assert.equal(total, g.wires.length, "every wire is sent by exactly one agent");
  });
});

describe("busSummary", () => {
  /**
   * Never folds `unknown` into a ratio.
   *
   * "14 of 16 carried" and "14 of 16 carried, 2 unrecorded" are different
   * claims, and a summary that computes a percentage has silently made the first
   * one. This asserts the word survives.
   */
  it("reports unrecorded hops separately from carried ones", () => {
    const g = busOf(docsOf(hemoFlows(1_700_000_000_000) as never));
    const s = busSummary(g);
    assert.match(s, /agent\(s\)/);
    assert.match(s, /hop\(s\)/);
    assert.ok(!/%/.test(s), "a percentage would fold the unknown hops into the answer");
  });
});
