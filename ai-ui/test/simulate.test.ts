import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { SIMULATION_JS } from "../src/simulate.ts";
import { cochleaFlows } from "../src/cochlea-demo.ts";
import { hemoFlows } from "../src/hemo-demo.ts";

/**
 * The shim's own `traceOf`, executed rather than paraphrased.
 *
 * ## What this is defending, and why it needed defending four times
 *
 * The flow store records `attempts[].observation = { digest, source }`. The desk,
 * the panel and `bus.ts` read `attempt.digest` and `attempt.source`, because that
 * is what the real `traceOf` in `trace.ts` produces. Every place that builds a
 * trace has to do that flattening, and **four separate places in this repository
 * failed to**, each independently, each silently:
 *
 * 1–3. `cochlea-demo.ts`, `dsp-demo.ts`, `memory-demo.ts` and
 *      `cochlea-project.ts` built flat attempts. Covered by
 *      `observations.test.ts`.
 * 4.   This file's subject: the shim passed its attempts through untouched, so
 *      **the demo's own five-second poll erased every observation on the page**.
 *      The server-rendered first frame had them; five seconds later the same
 *      desk had none, every wire went grey and dashed, and the trace face said
 *      "not enough to say" about flows that had recorded six.
 *
 * The fourth is the worst of them because it is invisible to any check that
 * looks at the built file: the HTML is correct, and the page destroys itself
 * after it loads. It was found by a reader opening the published site and
 * sending a screenshot of a desk full of grey dashes.
 *
 * `simulate.ts` already carries a comment about the *first* time this exact
 * confusion happened in this exact function, on `contribution`. It was fixed
 * without anybody checking whether the field next to it had the same problem.
 * Hence a test rather than a third comment.
 */
function shimTraceOf() {
  /**
   * Execute the shipped source and reach in for the function.
   *
   * `SIMULATION_JS` is an IIFE that installs a fetch shim, so it cannot simply
   * be evaluated here. The declaration is lifted out by name — brittle on
   * purpose: if `traceOf` is renamed or restructured, this fails loudly rather
   * than silently testing nothing, which is the failure mode that would make the
   * whole file worthless.
   */
  const src = SIMULATION_JS;
  const start = src.indexOf("const traceOf = (steps) => {");
  assert.ok(start >= 0, "the shim's traceOf was renamed or restructured; this test is testing nothing");
  const agentOf = src.indexOf("const agentOf = ");
  assert.ok(agentOf >= 0, "the shim's agentOf moved");

  // Find the end of the traceOf declaration by brace matching.
  let i = src.indexOf("{", start + "const traceOf = (steps) =>".length);
  let depth = 0;
  let end = -1;
  for (let k = i; k < src.length; k += 1) {
    if (src[k] === "{") depth += 1;
    else if (src[k] === "}") {
      depth -= 1;
      if (depth === 0) {
        end = k + 1;
        break;
      }
    }
  }
  assert.ok(end > 0, "could not find the end of the shim's traceOf");

  const body = src.slice(start, end) + ";";
  return new Function(
    "const agentOf = (intent) => { const m = /agent=\"([A-Za-z0-9_-]+)\"/.exec(intent || ''); return m ? m[1] : null; };\n" +
      body +
      "\nreturn traceOf;",
  )() as (steps: unknown[]) => { steps: Array<{ index: number; attempts: Array<{ digest: string | null; source: string | null }> }> };
}

const SCOPES: Array<[string, Array<Record<string, unknown>>]> = [
  ["coclea-sr", cochleaFlows(1_700_000_000_000) as never],
  ["hemo-verified", hemoFlows(1_700_000_000_000) as never],
];

describe("the simulation shim's trace", () => {
  const traceOf = shimTraceOf();

  for (const [name, raw] of SCOPES) {
    it(`${name}: a poll does not erase the observations the page was rendered with`, () => {
      let settled = 0;
      let kept = 0;
      for (const doc of raw) {
        const t = traceOf(doc["steps"] as unknown[]);
        for (const s of t.steps)
          for (const a of s.attempts as Array<{ state?: string; digest: string | null }>) {
            if (a.state !== "done") continue;
            settled += 1;
            if (a.digest) kept += 1;
          }
      }
      assert.ok(settled > 0, `${name} has no settled attempts to check`);
      assert.equal(
        kept,
        settled,
        `${name}: the shim dropped ${settled - kept} of ${settled} observation(s). ` +
          `The first poll would replace a correct page with one where every wire is unknown.`,
      );
    });

    it(`${name}: it keeps the source, so a citation still has an address`, () => {
      for (const doc of raw) {
        const t = traceOf(doc["steps"] as unknown[]);
        for (const s of t.steps)
          for (const a of s.attempts as Array<{ digest: string | null; source: string | null }>)
            if (a.digest)
              assert.ok(
                a.source,
                `${name} step ${s.index}: an observation with no source is a packet the ` +
                  `inspector can describe and cannot cite`,
              );
      }
    });
  }
});
