import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  assertCited,
  inspectAgentWithAgent,
  inspectGate,
  inspectGateWithAgent,
  inspectWire,
  inspectWireWithAgent,
} from "../src/inspector.ts";
import type { Wire } from "../src/bus.ts";

const wire = (over: Partial<Wire>): Wire => ({
  id: "f:0->1",
  flowId: "f",
  flowTitle: "a flow",
  from: "A",
  to: "B",
  fromIndex: 0,
  toIndex: 1,
  state: "carried",
  packet: { id: "f:0-1", digest: "d0", source: "gate.report", runId: "r0", result: "said a thing" },
  because: "step 0 closed with an observation from gate.report",
  ...over,
});

/**
 * The single rule this module exists to enforce.
 *
 * An inspector that produces a confident sentence with no address is the exact
 * failure the project argues against, and shipping one *in the demo for that
 * argument* would be the most expensive drift available. So it is a thrown
 * error, not a lint.
 */
describe("assertCited", () => {
  it("refuses a verdict with no address", () => {
    assert.throws(
      () => assertCited({ verdict: "problem", says: "it is broken", cites: [], cost: "free" }),
      /cited nothing/,
    );
  });

  it("allows unknown to cite nothing, because that is what unknown means", () => {
    const f = assertCited({ verdict: "unknown", says: "nothing was recorded", cites: [], cost: "free" });
    assert.equal(f.cites.length, 0);
  });
});

describe("inspecting a wire", () => {
  it("says what moved and where it can be opened", () => {
    const i = inspectWire(wire({}));
    const obs = i.fields.find((f) => f.label === "observation");
    assert.equal(obs?.value, "d0");
    assert.equal(obs?.from, "gate.report", "the field has to be an address, not a label");
  });

  /**
   * The wording matters here and is asserted rather than left to review.
   *
   * "nothing recorded" and "nothing moved" are different claims about the world,
   * and only the first one is supported. A panel that prints the second is
   * inventing a negative result.
   */
  it("refuses to claim that nothing moved when nothing was recorded", () => {
    const i = inspectWire(wire({ state: "unknown", packet: null }));
    const c = i.fields.find((f) => f.label === "carried");
    assert.match(c!.value, /not a claim that nothing moved/);
  });

  it("returns unknown, not ok, for an unrecorded hop", () => {
    const f = inspectWireWithAgent(wire({ state: "unknown", packet: null }));
    assert.equal(f.verdict, "unknown");
    assert.match(f.says, /not a pass/);
  });

  it("cites the artifact when it has a verdict", () => {
    const f = inspectWireWithAgent(
      wire({ state: "ignored", because: "step 1 carried 0 of the 4000 it was given" }),
    );
    assert.equal(f.verdict, "problem");
    assert.equal(f.cites[0]?.at, "gate.report");
  });
});

describe("inspecting a gate", () => {
  const g = {
    id: "GATE-A01",
    what: "worst relative error of the first 10 eigenvalues against the closed form",
    measured: "2.592e-4",
    tolerance: "1.0e-4",
    passed: false,
    report: "projects/coclea-sr/gates/reports/report_A01.json",
  };

  /**
   * Both halves, unrounded, side by side.
   *
   * The entire difference between a gate and an opinion is that a reader can see
   * the measurement and the threshold at once and do the comparison themselves.
   * A panel that showed only the verdict would have removed exactly that.
   */
  it("shows the measurement and the threshold that was declared before the run", () => {
    const i = inspectGate(g);
    const m = i.fields.find((f) => f.label === "measured");
    const t = i.fields.find((f) => f.label === "declared before the run");
    assert.equal(m?.value, "2.592e-4");
    assert.equal(t?.value, "1.0e-4");
    assert.equal(m?.from, g.report);
    assert.equal(t?.from, g.report);
  });

  it("cites the report in its finding", () => {
    const f = inspectGateWithAgent(g);
    assert.equal(f.verdict, "problem");
    assert.equal(f.cites[0]?.at, g.report);
    assert.match(f.says, /2\.592e-4/);
    assert.match(f.says, /1\.0e-4/);
  });
});

describe("inspecting an agent", () => {
  it("calls a subagent with no file a problem, and cites the declaration", () => {
    const f = inspectAgentWithAgent({ name: "GHOST", tools: [], missing: true }, undefined);
    assert.equal(f.verdict, "problem");
    assert.match(f.says, /resolve to nothing at run time/);
    assert.equal(f.cites.length, 1);
  });

  it("says unknown about an agent with no traffic here", () => {
    const f = inspectAgentWithAgent({ name: "IDLE", tools: ["read"], missing: false }, undefined);
    assert.equal(f.verdict, "unknown");
  });
});
