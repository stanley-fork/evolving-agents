import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { traceOf } from "../src/trace.ts";
import { cochleaFlows } from "../src/cochlea-demo.ts";
import { cochleaProjectFlows } from "../src/cochlea-project.ts";
import { dspFlows } from "../src/dsp-demo.ts";
import { memoryFlows } from "../src/memory-demo.ts";
import { hemoFlows } from "../src/hemo-demo.ts";
import { busOf } from "../src/bus.ts";

/**
 * Every demo scope must actually record the observations it thinks it records.
 *
 * ## Why this file exists
 *
 * Four of the five scopes shipped for months recording **none**. `traceOf` reads
 * `attempt.observation.digest`; every demo built its attempts with a flat
 * `digest` beside `runId`. That parsed, typechecked, passed the whole suite, and
 * rendered — as `no observation` under every step, with the trace face
 * announcing "not enough to say" about a flow that had recorded six.
 *
 * `cochlea-project.ts` is the sharpest version: its helper carries a comment
 * saying it is "the shape the desk actually reads", written when the *other*
 * half of the same shape was fixed. The comment was true about `n` and `runId`
 * and false about `digest`, and nothing could tell the difference.
 *
 * It was found by drawing the handoffs. A quiet fallback string in a panel is
 * invisible; twelve grey dashed wires across the middle of the desk are not.
 *
 * ## What it asserts
 *
 * That a step which claims to have produced something produces a digest the
 * trace can read, and therefore a wire [bus.ts](../src/bus.ts) can draw as
 * carried. Deliberately *not* "most steps" or "at least one" — a threshold is
 * how the previous version of this defect would have survived: four scopes at
 * zero and one at full would have passed any "at least one scope works" check.
 */
const AT = 1_700_000_000_000;
const agentOf = () => null;

const SCOPES: Array<[string, Array<Record<string, unknown>>]> = [
  ["coclea-sr · gates", cochleaFlows(AT) as never],
  ["coclea-sr · project", cochleaProjectFlows(AT) as never],
  ["hemo-verified", hemoFlows(AT) as never],
  ["signal lab", dspFlows(AT) as never],
  ["memory lab", memoryFlows(AT) as never],
];

describe("every scope records the observations it claims", () => {
  for (const [name, raw] of SCOPES) {
    it(`${name}: a settled attempt carries a readable observation`, () => {
      const docs = raw.map((d) => ({
        id: d["id"] as string,
        title: d["title"] as string,
        trace: traceOf(d["steps"] as never, agentOf),
      }));

      let settled = 0;
      let recorded = 0;
      for (const doc of docs)
        for (const s of doc.trace.steps)
          for (const a of s.attempts) {
            // `blocked` and `failed` attempts may legitimately record nothing:
            // that is the whole point of the unknown wire state. Only attempts
            // that closed as done are held to this.
            if (a.state !== "done") continue;
            settled += 1;
            if (a.digest) recorded += 1;
          }

      assert.ok(settled > 0, `${name} has no settled attempts at all`);
      assert.equal(
        recorded,
        settled,
        `${name}: ${settled - recorded} of ${settled} settled attempt(s) recorded no observation. ` +
          `The demo builds them under a key traceOf does not read.`,
      );
    });

    it(`${name}: its handoffs are drawn from evidence, not guessed`, () => {
      const docs = raw.map((d) => ({
        id: d["id"] as string,
        title: d["title"] as string,
        trace: traceOf(d["steps"] as never, agentOf),
      }));
      const g = busOf(docs);
      const unknown = g.wires.filter((w) => w.state === "unknown");
      /**
       * An `unknown` wire is allowed — it is the honest state, and the hemo
       * scope has two on purpose. What is not allowed is a scope that is
       * *entirely* unknown, which is what the flat-digest defect produced and
       * what nothing else would have caught.
       */
      assert.ok(
        g.wires.length === 0 || unknown.length < g.wires.length,
        `${name}: every one of its ${g.wires.length} hops is unrecorded`,
      );
    });
  }
});
