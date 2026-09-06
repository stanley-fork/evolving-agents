/**
 * Run the navigation benchmark.
 *
 *   node bench/navigation.ts --oracle                       # the ceiling
 *   node bench/navigation.ts --engine llama.cpp             # the model
 *   node bench/navigation.ts --sizes 100,1000,10000 --json out.json
 *
 * With `--oracle` it runs the perfect navigator and reports the ceiling: what
 * the store's shape allows. That is not a result about any model and it is
 * labelled `oracle` in every row so it cannot be mistaken for one.
 *
 * Without it, it needs a local server. See `scripts/verify-model.ts` — a run
 * whose weights were not checked records `verified: false` and should be read
 * as a run under a transcription.
 */
import { writeFileSync } from "node:fs";
import { approxCounter, budgetFor } from "../src/context/budget.ts";
import { OracleNavigator } from "../src/model/fake.ts";
import { LlamaCppModel } from "../src/model/llama-cpp.ts";
import { OllamaModel } from "../src/model/ollama.ts";
import { DEFAULT_PROFILE, profileOf, verifyAgainst, type ProfileName } from "../src/model/qwen-profile.ts";
import { runNavigationBench, renderSummary, type RunRow } from "../src/bench/navigation.ts";
import type { LocalModel } from "../src/model/local-model.ts";

const flag = (n: string) => process.argv.includes("--" + n);
const arg = (n: string, d: string) => {
  const i = process.argv.indexOf("--" + n);
  return i >= 0 ? (process.argv[i + 1] ?? d) : d;
};

const sizes = arg("sizes", "200,1000,10000,50000").split(",").map(Number).filter((n) => n > 0);
const profile = profileOf(arg("profile", DEFAULT_PROFILE) as ProfileName);
const engine = arg("engine", "llama.cpp");
const baseUrl = arg("base-url", engine === "ollama" ? "http://127.0.0.1:11434/v1" : "http://127.0.0.1:8080/v1");

let verified = false;
let live: LocalModel | null = null;
if (!flag("oracle")) {
  live = engine === "ollama"
    ? new OllamaModel({ baseUrl, model: profile.modelId, numCtx: profile.effectiveContext })
    : new LlamaCppModel({ baseUrl, model: profile.modelId });
  try {
    const served = await live.served();
    const bad = verifyAgainst(profile, served);
    verified = bad.length === 0;
    if (!verified) for (const m of bad) console.error(`mismatch on ${m.field}: expected ${m.expected}, served ${m.served}`);
  } catch (err) {
    console.error(`nothing answered at ${baseUrl}: ${(err as Error).message}`);
    console.error("Run with --oracle to measure the store's ceiling without weights.");
    process.exit(2);
  }
}

console.log(
  flag("oracle")
    ? "arm: oracle — the ceiling the store's shape allows. NOT a result about any model."
    : `model: ${profile.modelId} ${profile.quantization} via ${engine}, verified: ${verified}`,
);
console.log(`sizes: ${sizes.join(", ")}   effective context: ${profile.effectiveContext}\n`);

const rows: RunRow[] = [];
const { summaries } = await runNavigationBench({
  sizes,
  counter: approxCounter,
  budget: budgetFor(profile.effectiveContext),
  modelFor: (_arm, fact) => (live ? live : new OracleNavigator(fact.question, fact.answer)),
  onRow: (r) => {
    rows.push(r);
    console.log(
      `${r.arm.padEnd(8)} ${String(r.size).padStart(6)}  ${r.ending.padEnd(13)} ` +
        `${r.correct ? "correct" : "       "}  loaded ${String(r.loaded).padStart(6)}  ` +
        `${r.ratio ? r.ratio.toFixed(0) + "x" : "—"}`,
    );
  },
});

console.log("\n" + renderSummary(summaries));

const out = arg("json", "");
if (out) {
  writeFileSync(
    out,
    JSON.stringify(
      {
        model: flag("oracle") ? "oracle" : profile.modelId,
        quantization: flag("oracle") ? null : profile.quantization,
        engine: flag("oracle") ? "oracle" : engine,
        verified,
        effectiveContext: profile.effectiveContext,
        counter: approxCounter.describe,
        rows,
        summaries,
      },
      null,
      2,
    ) + "\n",
  );
  console.log(`\nwrote ${out}`);
}
