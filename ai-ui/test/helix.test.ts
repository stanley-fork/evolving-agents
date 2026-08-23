import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  TWIST,
  assertJustified,
  attentionOf,
  bodiesAt,
  bundleOf,
  phaseOf,
  rotationFor,
} from "../src/helix.ts";
import { threadsOf } from "../src/threads.ts";
import { traceOf } from "../src/trace.ts";
import { agentOfIntent } from "../src/server.ts";
import { cochleaFlows } from "../src/cochlea-demo.ts";
import { cochleaProjectFlows } from "../src/cochlea-project.ts";
import { memoryFlows } from "../src/memory-demo.ts";
import { hemoFlows } from "../src/hemo-demo.ts";

const AT = 1_700_000_000_000;
const docsOf = (raw: Array<Record<string, unknown>>) =>
  raw.map((d) => ({
    id: d["id"] as string,
    title: d["title"] as string,
    state: d["state"] as string,
    trace: traceOf(d["steps"] as never, agentOfIntent),
  }));

const coc = threadsOf(docsOf([...cochleaFlows(AT), ...cochleaProjectFlows(AT)] as never));

describe("the bundle", () => {
  it("puts every strand at its own phase, evenly around the axis", () => {
    const n = 5;
    const at = Array.from({ length: n }, (_, i) => phaseOf(i, n, 0, 0, 0));
    // At the origin with no rotation, the phases are the n-th roots of unity.
    for (let i = 0; i < n; i += 1) {
      assert.ok(Math.abs(at[i]!.z - Math.cos((Math.PI * 2 * i) / n)) < 1e-12);
      assert.ok(Math.abs(at[i]!.y - Math.sin((Math.PI * 2 * i) / n)) < 1e-12);
    }
    /**
     * Distinct as *positions*, not as depths.
     *
     * The first version of this checked that the `z` values were distinct, and
     * they are not: cosine is symmetric, so a strand at 72° and one at 288° sit
     * at the same depth on opposite sides of the axis. That is correct geometry
     * and a wrong assertion — the property is that no two strands occupy the
     * same *place*, which is the pair.
     */
    const places = new Set(at.map((p) => `${p.y.toFixed(9)},${p.z.toFixed(9)}`));
    assert.equal(places.size, n, "two strands share a position, so they are one strand");
  });

  it("stays on the unit circle, so depth is always in [-1, 1]", () => {
    for (let t = -500; t <= 500; t += 37)
      for (let i = 0; i < 7; i += 1) {
        const p = phaseOf(i, 7, t, 0, 1.1);
        assert.ok(Math.abs(p.y * p.y + p.z * p.z - 1) < 1e-12, "off the circle");
        assert.ok(p.z >= -1.000001 && p.z <= 1.000001);
      }
  });

  /**
   * The property the whole arrangement is for.
   *
   * Rotation is how the surface pays attention: turning the bundle must actually
   * bring the chosen strand to the front, or "bring it forward" is a word rather
   * than a mechanism.
   */
  it("rotationFor brings the chosen strand to the front at now, and only that one", () => {
    const n = 6;
    for (let i = 0; i < n; i += 1) {
      const rot = rotationFor(i, n, 120);
      const depths = Array.from({ length: n }, (_, k) => phaseOf(k, n, 120, 120, rot).z);
      const front = depths.indexOf(Math.max(...depths));
      assert.equal(front, i, `turning to strand ${i} brought ${front} forward`);
      assert.ok(depths[i]! > 0.999, "the chosen strand must be fully forward");
    }
  });

  /**
   * The drift *is* the rotation, and that is why the drift is free.
   *
   * Content moves left because the clock runs; because the strands twist along
   * the axis, that same motion changes every phase. Nothing has to be animated
   * for the bundle to appear to turn — if this stopped being true the surface
   * would need a second, meaningless animation to fake it.
   */
  it("twists along the axis, so drifting in time changes the phases", () => {
    const a = phaseOf(0, 4, 0, 0, 0);
    const quarter = Math.PI / 2 / TWIST;
    const b = phaseOf(0, 4, quarter, 0, 0);
    assert.ok(Math.abs(a.z - 1) < 1e-9, "starts at the front");
    assert.ok(Math.abs(b.z) < 1e-9, "a quarter turn later it is edge-on");
  });

  it("returns segments sorted back to front, so drawing in order occludes", () => {
    const segs = bundleOf(coc, { now: coc.span, rotation: 0.4, t0: 0, t1: coc.span });
    assert.ok(segs.length > 0);
    for (let i = 1; i < segs.length; i += 1)
      assert.ok(segs[i]!.z >= segs[i - 1]!.z - 1e-9, "not sorted by depth");
  });

  it("draws nothing outside the window it was asked for", () => {
    const segs = bundleOf(coc, { now: coc.span, rotation: 0, t0: coc.span - 30, t1: coc.span });
    for (const s of segs)
      assert.ok(s.t1 >= coc.span - 30 && s.t0 <= coc.span, `${s.flowId} is outside the window`);
  });
});

describe("bodies on the strand", () => {
  /**
   * A body is an agent *holding* something, so there is no body where nobody is.
   *
   * The desk drew an agent as a box a flow was attached to. Here the agent is
   * attached to the flow, which is the right way round: an agent holds a task for
   * a while and hands it on, and it is the task that persists. Drawing a body at
   * a moment when the strand has no step spanning it would assert work nobody was
   * doing.
   */
  it("puts a body only where a step actually spans that moment", () => {
    const t = coc.span - 5;
    const bodies = bodiesAt(coc, t, 0);
    for (const b of bodies) {
      const th = coc.threads.find((x) => x.flowId === b.flowId)!;
      const spanning =
        th.rests.some((r) => t >= r.x0 && t <= r.x1) ||
        th.crosses.some((c) => t >= c.x0 && t <= c.x1 && (c.from === "@human" || c.to === "@human"));
      assert.ok(spanning, `${b.flowId} has a body at a moment nothing spans`);
    }
  });

  it("names the agent that holds it, and marks the human at the ends", () => {
    const first = coc.threads[0]!;
    const opening = first.crosses[0]!;
    const mid = (opening.x0 + opening.x1) / 2;
    const b = bodiesAt(coc, mid, 0).find((x) => x.flowId === first.flowId);
    assert.ok(b, "nobody is holding the thread while the person is handing it over");
    assert.equal(b!.kind, "human");
    assert.equal(b!.name, "@human");
  });

  it("is sorted back to front like the strands", () => {
    const bodies = bodiesAt(coc, coc.span - 20, 0.9);
    for (let i = 1; i < bodies.length; i += 1)
      assert.ok(bodies[i]!.z >= bodies[i - 1]!.z - 1e-9);
  });
});

describe("attention", () => {
  /**
   * The rotation is a claim, so it carries an address.
   *
   * Bringing a strand forward says *this is what deserves looking at*. This
   * project's whole argument is that a claim needs an address, and a bundle that
   * turns for a reason it cannot point at is exactly the confident gesture the
   * surface exists to refuse.
   */
  it("every reason to turn cites something, except the two that are about absence", () => {
    for (const world of [
      coc,
      threadsOf(docsOf(hemoFlows(AT) as never)),
      threadsOf(docsOf(memoryFlows(AT) as never)),
    ])
      for (const a of attentionOf(world)) {
        assertJustified(a);
        if (a.kind === "running" || a.kind === "ignored" || a.kind === "blocked")
          assert.ok(a.at, `${a.kind} must cite: ${a.reason}`);
      }
  });

  it("refuses a claim with no address", () => {
    assert.throws(
      () =>
        assertJustified({
          flowId: "f",
          title: "t",
          ordinal: 0,
          kind: "running",
          reason: "something is happening",
          at: null,
          rank: 4,
        }),
      /must not make/,
    );
  });

  it("ranks a live step above a finding, and a finding above a failure", () => {
    const a = attentionOf(coc);
    assert.equal(a[0]!.kind, "running", "the cochlea scope has one open step");
    const mem = attentionOf(threadsOf(docsOf(memoryFlows(AT) as never)));
    assert.equal(mem[0]!.kind, "ignored", "the memory scope's finding must come first");
    // Stated as an ordering of kinds, so it can be disagreed with in words.
    const order = ["running", "ignored", "blocked", "open", "settled"];
    for (const world of [coc, threadsOf(docsOf(hemoFlows(AT) as never))]) {
      const ranked = attentionOf(world);
      for (let i = 1; i < ranked.length; i += 1)
        assert.ok(
          order.indexOf(ranked[i]!.kind) >= order.indexOf(ranked[i - 1]!.kind),
          "attention is not in the stated order of kinds",
        );
    }
  });
});
