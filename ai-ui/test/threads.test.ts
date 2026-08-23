import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { contentions, threadsOf } from "../src/threads.ts";
import { traceOf } from "../src/trace.ts";
import { cochleaFlows } from "../src/cochlea-demo.ts";
import { hemoFlows } from "../src/hemo-demo.ts";
import { agentOfIntent } from "../src/server.ts";

/**
 * The properties the thread view exists to have.
 *
 * A prototype with no tests is a picture, and a picture cannot be wrong about
 * anything — which is exactly why it is easy to be persuaded by one. These
 * assert the four claims the metaphor makes over the desk, so that if the
 * geometry stops making them the sketch stops being worth building.
 */
const AT = 1_700_000_000_000;
const docsOf = (raw: Array<Record<string, unknown>>) =>
  raw.map((d) => ({
    id: d["id"] as string,
    title: d["title"] as string,
    state: d["state"] as string,
    trace: traceOf(d["steps"] as never, agentOfIntent),
  }));

describe("threadsOf", () => {
  const coc = threadsOf(docsOf(cochleaFlows(AT) as never));

  it("puts the human first and every agent below, in first-appearance order", () => {
    assert.equal(coc.lanes[0]?.id, "@human");
    assert.equal(coc.lanes[0]?.kind, "human");
    assert.deepEqual(
      coc.lanes.map((l) => l.row),
      coc.lanes.map((_, i) => i),
      "rows must be dense and ordered; a renderer positions by row",
    );
  });

  it("every thread starts with the person who asked", () => {
    for (const t of coc.threads) {
      assert.equal(t.crosses[0]?.from, "@human", `${t.title} did not start with a person`);
      assert.equal(t.crosses[0]?.x0, 0);
    }
  });

  /**
   * The claim the desk could not make at all.
   *
   * A flow that finished delivered something to somebody. A flow that is blocked
   * did not. Drawing every rope as returning to the human would tell a reader
   * that work was delivered when it was abandoned — which is the same family of
   * lie as drawing an unrecorded hop like a hop that worked.
   */
  it("a finished thread comes home and a blocked one does not", () => {
    const done = coc.threads.find((t) => t.state === "done");
    const held = coc.threads.find((t) => t.state === "blocked");
    assert.ok(done && held, "the cochlea scope should carry one of each");
    assert.equal(done!.delivered, true);
    assert.equal(held!.delivered, false);
    assert.equal(done!.crosses[done!.crosses.length - 1]!.to, "@human");
    assert.notEqual(held!.crosses[held!.crosses.length - 1]!.to, "@human");
  });

  it("time only moves forward: every segment ends after it starts, and they abut", () => {
    for (const t of coc.threads) {
      const all = [...t.rests, ...t.crosses].sort((a, b) => a.x0 - b.x0);
      for (const s of all) assert.ok(s.x1 > s.x0, `${t.title} has a segment with no duration`);
      for (let i = 1; i < all.length; i += 1)
        assert.ok(
          Math.abs(all[i]!.x0 - all[i - 1]!.x1) < 1e-9,
          `${t.title} has a gap or an overlap between segments — the rope must be continuous`,
        );
    }
  });

  /**
   * The vocabulary transfers unchanged.
   *
   * This is the test of whether the thread view is a redesign or a rewrite. It
   * is a redesign: the five states from `bus.ts` mean exactly what they meant,
   * and nothing about what may be claimed changes because the picture did.
   */
  it("carries bus.ts's five states, and splits held from failed the same way", () => {
    const hemo = threadsOf(docsOf(hemoFlows(AT) as never));
    const states = new Set(hemo.threads.flatMap((t) => t.crosses.map((c) => c.state)));
    assert.ok(states.has("open"), "hemo's A4 flow is held with nothing recorded");
    assert.ok(!states.has("blocked"), "nothing in hemo ran and came out against a threshold");
    for (const t of hemo.threads)
      for (const c of t.crosses)
        assert.ok(
          ["carried", "ignored", "blocked", "open", "unknown"].includes(c.state),
          `${c.state} is not one of the five`,
        );
  });

  it("an unrecorded hop carries no address, so nothing can be cited for it", () => {
    for (const t of coc.threads)
      for (const c of t.crosses)
        if (c.state === "unknown")
          assert.equal(c.digest, null, "an unknown crossing must have nothing to open");
  });
});

describe("contentions", () => {
  /**
   * One agent holding two thoughts at once.
   *
   * The desk drew this as a multiplier badge on a cube — a *count* where the
   * reader wanted a *collision*. Here it is two ropes through one lane at the
   * same X, which is a thing you can see without being told to look.
   */
  it("finds where two threads occupy one lane at the same moment", () => {
    const w = threadsOf(docsOf(cochleaFlows(AT) as never));
    const c = contentions(w);
    assert.ok(c.length > 0, "the two membrane chains use the same agents at the same steps");
    for (const x of c) {
      assert.ok(x.x1 > x.x0, "a contention with no width is not one");
      assert.equal(new Set(x.flows).size, 2, "a thread does not contend with itself");
    }
  });
});

describe("lane order", () => {
  /**
   * The ordering must be a function of the data, not of the listing order.
   *
   * First-appearance ordering made the picture depend on which flow happened to
   * be listed first — reorder the input and every rope changes shape. A layout
   * that reshuffles for a reason the reader cannot see is a layout nobody can
   * learn, and a sketch whose shape is an accident of argument order proves
   * nothing about the metaphor.
   */
  it("does not depend on the order the flows are given in", () => {
    const raw = docsOf(hemoFlows(AT) as never);
    const a = threadsOf(raw).lanes.map((l) => l.id);
    const b = threadsOf([...raw].reverse()).lanes.map((l) => l.id);
    assert.deepEqual(a, b, "reversing the input changed the lanes");
  });

  it("puts agents that act early above agents that act late", () => {
    const w = threadsOf(docsOf(cochleaFlows(AT) as never));
    const row = new Map(w.lanes.map((l) => [l.id, l.row]));
    for (const t of w.threads) {
      const rows = t.rests.map((r) => row.get(r.lane)!);
      // Not strictly monotonic — an agent can be revisited, and VERIFICADOR-MATH
      // is. What must hold is that the thread ends below where it started, which
      // is what makes a rope a slope rather than a search.
      assert.ok(
        rows[rows.length - 1]! > rows[0]!,
        `${t.title} ends no lower than it starts; the ordering is not helping`,
      );
    }
  });
});
