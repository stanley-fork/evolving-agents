/**
 * The platform's model, read from one file, with its provenance attached.
 *
 * `MODEL.json` at the repository root is the single place the default model is
 * declared. Nothing here restates a model id, a quantization or a size: this
 * module reads that file and turns it into types, so that changing the model is
 * one edit and a failing test rather than a search across five packages.
 *
 * ## Every field is a claim, and a claim carries an address
 *
 * That is the rule the rest of this repository already runs on, and it applies
 * with more force here than anywhere else, because a model's identity is the one
 * fact that decides what every benchmark number means. So each claim in
 * `MODEL.json` carries `citedAs` — where it was read from — and `verified` —
 * whether anything has checked it against the thing it describes.
 *
 * **Right now `verified` is false for all of them.** The specification was
 * transcribed onto a machine that cannot reach `huggingface.co` (the egress
 * proxy answers 403 to CONNECT), so no size, no licence and no capability here
 * has been confirmed against the weights they describe. `verifyAgainst` turns
 * that false into a true by asking a running server what it is actually serving.
 * Until it has been run, a benchmark result that names this model is naming a
 * transcription.
 *
 * Pure, apart from reading the JSON. No fetch, no clock, no DOM.
 */
import MODEL from "../../../MODEL.json" with { type: "json" };

export type ProfileName = "constrained" | "reference" | "quality";

/** A fact that was written down, with where it was written down. */
export interface Cited<T> {
  value: T;
  citedAs: string;
  /** Has anything checked this against the thing it describes? */
  verified: boolean;
}

export interface QwenProfile {
  name: ProfileName;
  /** Why this profile exists, in the specification's own words. */
  purpose: string;
  modelId: string;
  quantization: string;
  /** Bytes on disk, as transcribed. `null` where the specification gives none. */
  approxBytes: number | null;
  /**
   * What the server is asked to allocate.
   *
   * Larger than `effectiveContext` on purpose in the reference profile: the
   * hardware can hold more, and the benchmark still refuses to use it.
   */
  physicalContext: number;
  /**
   * What ai-storage is allowed to use, which is the number that matters.
   *
   * A model that physically supports a million tokens runs here at 8192,
   * because this component exists to solve context scarcity rather than to hide
   * it. Run the benchmark at the native maximum and you have measured the
   * model's context window and nothing about the storage layer.
   */
  effectiveContext: number;
  targetHardware: string;
  citedAs: string;
  verified: boolean;
}

/** The model every component defaults to, and the address that says so. */
export const PLATFORM_MODEL: Cited<string> = {
  value: MODEL.default.id,
  citedAs: MODEL.default.citedAs,
  verified: MODEL.default.verified,
};

/** The cap ai-storage runs under regardless of what the weights support. */
export const EFFECTIVE_CONTEXT: number = MODEL.effectiveContext;

/** The reference engine, and the ones known to serve the same API. */
export const ENGINES = {
  reference: MODEL.engines.reference,
  compatible: MODEL.engines.compatible as readonly string[],
} as const;

const PROFILE_NAMES: readonly ProfileName[] = ["constrained", "reference", "quality"];

function readProfile(name: ProfileName): QwenProfile {
  const p = MODEL.profiles[name];
  return {
    name,
    purpose: p.purpose,
    modelId: MODEL.default.id,
    quantization: p.quantization,
    approxBytes: p.approxBytes,
    physicalContext: p.physicalContext,
    effectiveContext: p.effectiveContext,
    targetHardware: p.targetHardware,
    citedAs: p.citedAs,
    verified: p.verified,
  };
}

/** Every declared profile, in the order the specification states them. */
export const PROFILES: Readonly<Record<ProfileName, QwenProfile>> = Object.freeze({
  constrained: readProfile("constrained"),
  reference: readProfile("reference"),
  quality: readProfile("quality"),
});

/**
 * The profile a result belongs to when it does not say.
 *
 * There is exactly one, and it is stated rather than inferred, because a
 * benchmark number whose quantization is unknown is a benchmark number that
 * cannot be compared to anything.
 */
export const DEFAULT_PROFILE: ProfileName = "reference";

export function profileOf(name: string): QwenProfile {
  if (!PROFILE_NAMES.includes(name as ProfileName))
    throw new Error(
      `ai-storage: no profile named "${name}". Declared profiles are ` +
        `${PROFILE_NAMES.join(", ")} — see MODEL.json.`,
    );
  return PROFILES[name as ProfileName];
}

/**
 * What a server said it is serving.
 *
 * Deliberately not the same shape as a profile: this is an observation, and the
 * point of keeping it separate is that comparing the two is an explicit step
 * somebody has to take.
 */
export interface ServedModel {
  /** The id the server reports, verbatim. */
  id: string;
  /** The context length the server reports, when it reports one. */
  contextLength: number | null;
  /** Where this was read from, e.g. `http://127.0.0.1:8080/v1/models`. */
  readFrom: string;
}

export interface ProfileMismatch {
  field: "id" | "context";
  expected: string;
  served: string;
}

/**
 * Compare a profile against what a server actually serves.
 *
 * Returns the mismatches rather than throwing, because the caller has to decide
 * whether a mismatch is fatal — a benchmark run must stop, an interactive
 * session may want to say so and continue. What it will not do is quietly
 * accept a different model: a run recorded under the wrong weights is worse
 * than a run that did not happen, because it looks like evidence.
 *
 * The id comparison is case-insensitive and ignores a tag suffix after ':',
 * because Ollama reports `qwen3.8-27b:q4_k_m` for what llama.cpp reports as
 * `qwen3.8-27b`. It does not ignore anything else.
 */
export function verifyAgainst(profile: QwenProfile, served: ServedModel): ProfileMismatch[] {
  const norm = (s: string) => s.trim().toLowerCase().split(":")[0]!.replace(/^.*\//, "");
  const out: ProfileMismatch[] = [];
  if (norm(served.id) !== norm(profile.modelId))
    out.push({ field: "id", expected: profile.modelId, served: served.id });
  if (served.contextLength !== null && served.contextLength < profile.effectiveContext)
    out.push({
      field: "context",
      expected: `at least ${profile.effectiveContext}`,
      served: String(served.contextLength),
    });
  return out;
}

/**
 * A profile, once something has actually looked at the server.
 *
 * The only way to get `verified: true` in this codebase. A run record that
 * carries one of these is a run whose weights were checked; a run record that
 * carries a bare profile is a run that trusted a transcription, and the
 * difference has to survive into the results file.
 */
export function verifiedProfile(profile: QwenProfile, served: ServedModel): QwenProfile {
  const bad = verifyAgainst(profile, served);
  if (bad.length)
    throw new Error(
      `ai-storage: the server at ${served.readFrom} is not serving profile ` +
        `"${profile.name}". ` +
        bad.map((m) => `${m.field}: expected ${m.expected}, served ${m.served}`).join("; "),
    );
  return { ...profile, verified: true, citedAs: served.readFrom };
}
