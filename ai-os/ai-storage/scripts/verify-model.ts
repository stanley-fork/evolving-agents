/**
 * Turn `verified: false` into something a run record can carry.
 *
 *   cd ai-storage && node scripts/verify-model.ts
 *   cd ai-storage && node scripts/verify-model.ts --profile constrained \
 *     --base-url http://127.0.0.1:11434/v1 --engine ollama
 *
 * `MODEL.json` says the platform runs a particular model at a particular
 * quantization. Nothing has checked that. This script asks a running server
 * what it is actually serving and prints the comparison, and it is the only
 * thing in this repository that may report a model as verified.
 *
 * It does not edit `MODEL.json`. A benchmark run records what the server said
 * at the time it ran, next to its numbers; a file that says "verified" long
 * after the server was stopped is a claim about the past that nothing checked.
 *
 * Exit codes: 0 the server matches, 1 it does not, 2 nothing answered.
 */
import {
  DEFAULT_PROFILE,
  profileOf,
  verifyAgainst,
  type ProfileName,
} from "../src/model/qwen-profile.ts";
import { LlamaCppModel } from "../src/model/llama-cpp.ts";
import { OllamaModel } from "../src/model/ollama.ts";
import type { LocalModel } from "../src/model/local-model.ts";

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 ? (process.argv[i + 1] ?? fallback) : fallback;
}

const engine = arg("engine", "llama.cpp");
const profileName = arg("profile", DEFAULT_PROFILE) as ProfileName;
const profile = profileOf(profileName);
const baseUrl = arg(
  "base-url",
  engine === "ollama" ? "http://127.0.0.1:11434/v1" : "http://127.0.0.1:8080/v1",
);

const model: LocalModel =
  engine === "ollama"
    ? new OllamaModel({ baseUrl, model: profile.modelId, numCtx: profile.effectiveContext })
    : new LlamaCppModel({ baseUrl, model: profile.modelId });

const line = (k: string, v: string) => console.log(`  ${k.padEnd(18)} ${v}`);

console.log(`profile "${profile.name}" — ${profile.purpose}`);
line("expects", `${profile.modelId} at ${profile.quantization}`);
line("effective context", String(profile.effectiveContext));
line("transcribed from", profile.citedAs);
console.log();

let served: Awaited<ReturnType<LocalModel["served"]>>;
try {
  served = await model.served();
} catch (err) {
  console.error(`nothing answered at ${baseUrl}: ${(err as Error).message}`);
  console.error(
    engine === "ollama"
      ? "  start it with:  ollama serve"
      : "  start it with:  llama serve -hf <the GGUF repository>:" +
          profile.quantization +
          " --host 127.0.0.1 --port 8080 -c " +
          profile.physicalContext,
  );
  process.exit(2);
}

console.log(`served at ${served.readFrom}`);
line("id", served.id);
line("context", served.contextLength === null ? "not reported" : String(served.contextLength));
console.log();

const bad = verifyAgainst(profile, served);
if (!bad.length) {
  console.log("match. A run against this server may record the profile as verified.");
  process.exit(0);
}
for (const m of bad) console.error(`mismatch on ${m.field}: expected ${m.expected}, served ${m.served}`);
console.error(
  "\nA result recorded under the wrong weights is worse than a result that does not " +
    "exist, because it looks like evidence.",
);
process.exit(1);
