import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { contentions, threadsOf } from "../src/threads.ts";
import { traceOf } from "../src/trace.ts";
import { cochleaFlows } from "../src/cochlea-demo.ts";
import { cochleaProjectFlows } from "../src/cochlea-project.ts";
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

  /**
   * On a clock axis only the earliest thread starts at zero.
   *
   * This asserted `x0 === 0` for every thread, which was true when X was step
   * order and every thread began at step zero. On a clock a thread begins when
   * it began — that is the entire point of the axis — and the property that
   * survives is that the question comes from a person and lands before the first
   * step, not that everything starts together.
   */
  it("every thread starts with the person who asked, before its first step", () => {
    for (const t of coc.threads) {
      const open = t.crosses[0]!;
      assert.equal(open.from, "@human", `${t.title} did not start with a person`);
      assert.ok(open.x0 >= 0, `${t.title} starts before the axis does`);
      assert.equal(
        open.x1,
        Math.min(...t.rests.map((r) => r.x0)),
        `${t.title}'s opening crossing does not land on its first step`,
      );
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
  it("draws on the clock when every attempt has one", () => {
    assert.equal(coc.basis, "clock", "the cochlea attempts all record startedAt");
    assert.ok(coc.t1 > coc.t0, "a clock world must span some real time");
    // Widths are durations now, and they differ. A world where every step is the
    // same width has silently fallen back to sequence while claiming a clock.
    const widths = coc.threads.flatMap((t) => t.rests.map((r) => r.x1 - r.x0));
    assert.ok(new Set(widths.map((w) => w.toFixed(2))).size > 1, "every step is the same width");
  });

  /**
   * A step that has not begun has no time, and must not be given one.
   *
   * The first version fell back to the start of the world for a step with no
   * attempts, so GATE-D1 — whose last steps are pending — was drawn spanning the
   * full seventy-two hour window: a flow that started forty minutes ago, drawn
   * as three days of work.
   */
  it("places a step that has not started after the last one that did", () => {
    const w = threadsOf(docsOf(cochleaProjectFlows(AT) as never));
    for (const t of w.threads) {
      const ordered = [...t.rests].sort((a, b) => a.index - b.index);
      for (let i = 1; i < ordered.length; i += 1)
        assert.ok(
          ordered[i]!.x0 >= ordered[i - 1]!.x0,
          `${t.title} step ${ordered[i]!.index} starts before the step before it`,
        );
      const width = Math.max(...ordered.map((r) => r.x1)) - Math.min(...ordered.map((r) => r.x0));
      assert.ok(width < 24 * 60, `${t.title} spans ${(width / 60).toFixed(0)}h; nothing here ran that long`);
    }
  });

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
   * One agent holding two thoughts at once — and what the clock said about it.
   *
   * This test used to assert that the two membrane chains contend, because on a
   * sequence axis they do: same agents, same step numbers, same X. **On the
   * clock they ran thirty hours apart.** The assertion was true about the
   * drawing and false about the world, which is the precise failure the whole
   * surface argues against, committed by its own test.
   *
   * So it asserts the shape of a contention and not that one exists. On this
   * data none does, and the surface has to say that rather than leave a mark
   * unexplained — the desk's `×2` badge, it turns out, was counting an agent
   * appearing twice *inside one flow*, never two flows at once.
   */
  it("reports only overlaps that are real on the axis being drawn", () => {
    for (const raw of [cochleaFlows(AT), hemoFlows(AT)]) {
      const w = threadsOf(docsOf(raw as never));
      for (const x of contentions(w)) {
        assert.ok(x.x1 > x.x0, "a contention with no width is not one");
        assert.equal(new Set(x.flows).size, 2, "a thread does not contend with itself");
        // Every reported overlap must be checkable against the rests it came
        // from, or the mark is decoration.
        const rests = w.threads
          .flatMap((t) => t.rests)
          .filter((r) => r.lane === x.lane && x.flows.includes(r.flowId));
        assert.ok(rests.length >= 2, "a contention must name two real rests");
        for (const r of rests)
          assert.ok(r.x0 <= x.x1 && r.x1 >= x.x0, "a rest that does not overlap the span it is in");
      }
    }
  });

  /**
   * The mark exists whether or not this data draws it.
   *
   * A vocabulary that only appears when the sample happens to contain it is a
   * vocabulary nobody can rely on. Two rests deliberately made to overlap must
   * be found; if they are not, the surface would go quiet about a real collision
   * the first time one happened.
   */
  it("finds an overlap when there is one", () => {
    const w = threadsOf(docsOf(cochleaFlows(AT) as never));
    const lane = w.threads[0]!.rests[0]!.lane;
    const a = w.threads[0]!.rests[0]!;
    const forged = {
      ...w,
      threads: [
        w.threads[0]!,
        { ...w.threads[1]!, rests: [{ ...a, flowId: w.threads[1]!.flowId, lane }] },
      ],
    };
    const c = contentions(forged);
    assert.equal(c.length, 1);
    assert.equal(c[0]!.lane, lane);
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

describe("what may move", () => {
  /**
   * The rule the animated renderer is built on, asserted where it can be.
   *
   * > Everything that moves is a measurement. If nothing moves, nothing is
   * > happening.
   *
   * The renderer animates a segment if and only if its step is `running`, and
   * cancels its loop when none is — so a still surface is a true statement, and
   * that is the only thing that makes a moving one worth believing. What can be
   * checked here is the half that decides it: that `running` reaches the layout
   * intact and is distinguishable from every other state, including `pending`,
   * which looks similar and means the opposite.
   */
  it("carries `running` through to the segment, distinct from pending", () => {
    const w = threadsOf(docsOf(cochleaProjectFlows(AT) as never));
    const rests = w.threads.flatMap((t) => t.rests);
    const live = rests.filter((r) => r.state === "running");
    const pending = rests.filter((r) => r.state === "pending");
    assert.equal(live.length, 1, "the project scope has exactly one open step");
    assert.ok(pending.length > 0, "and steps that have not begun, which must not animate");
    // An open step has a start and no end. That is what makes it open, and it is
    // what the renderer's stillness claim rests on.
    assert.ok(live[0]!.x1 > live[0]!.x0);
  });

  it("a scene with nothing running has nothing that may move", () => {
    for (const raw of [cochleaFlows(AT), hemoFlows(AT)]) {
      const w = threadsOf(docsOf(raw as never));
      assert.equal(
        w.threads.flatMap((t) => t.rests).filter((r) => r.state === "running").length,
        0,
        "these scopes are settled; if one gains a running step the surface must move for it",
      );
    }
  });
});
