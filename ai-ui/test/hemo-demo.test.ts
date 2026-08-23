import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";
import { A4_ELSEWHERE, A4_HERE, H0, hemoAgents, hemoFlows } from "../src/hemo-demo.ts";

/**
 * The anti-drift test for the hemodynamics scope.
 *
 * `src/hemo-demo.ts` states H0's numbers as constants so the demo can be built
 * without a Python interpreter. That is a copy, and a copy of a number is a
 * number that will be wrong eventually — two cells of the project's own README
 * already had been. This resolves every constant back to
 * `gates/reports/h0.json` and fails if any of them has moved.
 *
 * Note what it does *not* do: it does not skip when the artifact is missing. A
 * missing attestation is exactly the condition under which the demo would go on
 * publishing last month's result, so it is a failure and not a skip. That rule
 * was learned here the expensive way — two `-Infinity` values turned four of the
 * cochlea drift tests into silent passes on 2026-08-14.
 */
const HERE = dirname(fileURLToPath(import.meta.url));
const H0_JSON = join(HERE, "..", "..", "projects", "hemo-verified", "gates", "reports", "h0.json");

function attested(): Record<string, any> {
  assert.ok(
    existsSync(H0_JSON),
    `${H0_JSON} is missing; the demo's H0 numbers have nothing to be checked against`,
  );
  return JSON.parse(readFileSync(H0_JSON, "utf8"));
}

describe("the hemo scope cannot drift from the attested artifact", () => {
  const a = attested();

  it("carries the run's shape exactly", () => {
    assert.equal(H0.n, a["n"]);
    assert.equal(H0.accepted, a["accepted"]);
    assert.equal(H0.rejected, a["rejected"]);
    assert.equal(H0.escalated, a["escalated"]);
    assert.equal(H0.killThreshold, a["kill_threshold"]);
  });

  it("carries the composite scores exactly, unrounded", () => {
    assert.equal(H0.aucComposite, a["auc_composite"]);
    assert.equal(H0.spearmanComposite, a["spearman_composite"]);
    assert.equal(H0.falseAcceptRate, a["false_accept_rate"]);
    assert.equal(H0.badFraction, a["bad_fraction"]);
  });

  it("carries every per-oracle AUC exactly", () => {
    for (const [k, v] of Object.entries(H0.perOracle)) {
      assert.equal(v, a["auc_per_oracle"][k], `${k} alone`);
    }
    assert.equal(
      Object.keys(H0.perOracle).length,
      Object.keys(a["auc_per_oracle"]).length,
      "an oracle was added or removed and the demo did not notice",
    );
  });

  it("carries the content hash of every oracle", () => {
    for (const [k, v] of Object.entries(H0.oracleHashes)) {
      assert.equal(v, a["oracle_hashes"][k], `${k} hash`);
    }
  });

  /**
   * The environment is the field whose absence was the finding.
   *
   * An artifact with no environment cannot distinguish *this disagrees* from
   * *this was produced somewhere else*, and the A4 flow in this scope is
   * entirely about that distinction. If the field disappears again, the flow is
   * telling a story its own evidence no longer supports.
   */
  it("carries the stack the run was produced on", () => {
    const env = a["environment"];
    assert.ok(env, "h0.json records no environment; the A4 flow's argument has no artifact");
    for (const k of ["python", "numpy", "scipy", "machine"] as const) {
      assert.equal(H0.environment[k], env[k], `environment.${k}`);
    }
  });

  it("keeps the two A4 readings distinct and orders them as the finding records", () => {
    assert.notEqual(A4_HERE, A4_ELSEWHERE, "the finding is that these differ");
    assert.equal(
      Number(H0.perOracle.A4.toFixed(3)),
      A4_ELSEWHERE,
      "the attested A4 is the second-machine reading; if that flips, the flow's prose is backwards",
    );
  });
});

describe("the hemo scope is built the way the desk expects", () => {
  const flows = hemoFlows(1_700_000_000_000);
  const names = new Set(hemoAgents().map((x) => x.name));

  it("names only agents it declares", () => {
    for (const f of flows)
      for (const s of f["steps"] as Array<{ agent: string }>)
        assert.ok(names.has(s.agent), `${s.agent} has no agent file in this scope`);
  });

  /**
   * The one property the redesign turns on.
   *
   * A step held open with no artifact must record *no observation*, so
   * [bus.ts](../src/bus.ts) draws the wire into it as `unknown` rather than as a
   * failure. Green-vs-red is a two-valued picture and this project's whole
   * argument is that the third value is where the honesty lives.
   */
  it("records no observation on the steps that have no artifact", () => {
    const held = flows.flatMap((f) =>
      (f["steps"] as Array<{ state: string; attempts: Array<{ observation: unknown }> }>).filter(
        (s) => s.state === "blocked",
      ),
    );
    assert.ok(held.length >= 2, "the scope should carry open questions, not only settled ones");
    for (const s of held)
      assert.equal(
        s.attempts[0]!.observation,
        null,
        "a blocked step with an observation would be drawn as evidence that does not exist",
      );
  });

  it("has at least one flow that is not done", () => {
    assert.ok(
      flows.some((f) => f["state"] !== "done"),
      "a scope with nothing open reads as a finished project, and this one is not",
    );
  });
});
