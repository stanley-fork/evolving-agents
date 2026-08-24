/**
 * The boundary between ai-storage and whatever is running the weights.
 *
 * ai-storage must never depend on llama.cpp or on Ollama. It depends on this
 * interface, and the engines implement it. The reason is not portability for
 * its own sake: it is that the benchmark this component exists to run compares
 * quantizations and engines, and a storage layer that knows which engine it is
 * talking to is a storage layer that can accidentally be tuned for one.
 *
 * ## What crosses this boundary, and what does not
 *
 * Crossing it: a prompt, a schema, a tool registry, a token count.
 *
 * Not crossing it: anything about storage. The model never receives a path, a
 * note id it did not read from a tool result, or a filesystem handle. That is
 * `security/` and `tools/`, and the separation is why a hallucinated
 * `write_file` cannot do anything — the operation does not exist on its side of
 * this line.
 *
 * ## Local only, and why it is checked at construction
 *
 * `AI_STORAGE_LOCAL_ONLY` defaults to true. A base URL that is not a loopback
 * address is refused when the model object is *built*, not when a request is
 * made, because a privacy guarantee that fails on the first request has already
 * failed: by then a prompt has been assembled and something has decided to send
 * it. The check has to be in front of that.
 *
 * No fetch at import time. The engines below are the only things here that
 * touch the network, and only to loopback unless the guard is explicitly off.
 */

/** What a call cost, measured rather than estimated. */
export interface Usage {
  promptTokens: number;
  /** Reasoning tokens, where the engine reports them separately. */
  reasoningTokens: number;
  completionTokens: number;
  /** Wall clock, milliseconds. */
  latencyMs: number;
  /** Time to the first token, where the engine streams. `null` otherwise. */
  timeToFirstTokenMs: number | null;
}

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  /** Set on a tool result, matching the call it answers. */
  toolCallId?: string;
  /** Set on an assistant turn that called tools. */
  toolCalls?: ToolCall[];
}

export interface ToolCall {
  id: string;
  name: string;
  /** Parsed arguments. The engines parse; callers never see raw JSON text. */
  arguments: Record<string, unknown>;
}

export interface CompletionRequest {
  messages: Message[];
  /** Sampling temperature. Storage work defaults low; see the roles table. */
  temperature: number;
  maxTokens: number;
  /** Whether the model is asked to think before answering. */
  thinking: boolean;
  /** A hard stop, so a hung engine cannot hang a benchmark. */
  timeoutMs: number;
}

export interface CompletionResult {
  text: string;
  usage: Usage;
  /** Why generation stopped, verbatim from the engine. */
  finishReason: string;
}

export interface StructuredRequest {
  messages: Message[];
  temperature: number;
  maxTokens: number;
  thinking: boolean;
  timeoutMs: number;
  /**
   * A JSON Schema the reply must satisfy.
   *
   * Structured output solves *syntax*. It does not solve *truth*, so every
   * caller validates the parsed object again on this side — see
   * `knowledge/schema.ts`. A schema-valid note with a fabricated source range is
   * exactly the failure this component is built to catch.
   */
  schema: Record<string, unknown>;
  /** A name for the schema, which some engines require. */
  schemaName: string;
}

export interface StructuredResult<T> {
  value: T;
  usage: Usage;
}

export interface ToolSpec {
  name: string;
  description: string;
  /** JSON Schema for the arguments. */
  parameters: Record<string, unknown>;
}

export interface ToolRequest {
  messages: Message[];
  tools: ToolSpec[];
  temperature: number;
  maxTokens: number;
  thinking: boolean;
  timeoutMs: number;
}

export interface ToolResult {
  /** Empty when the model answered in prose instead of calling anything. */
  toolCalls: ToolCall[];
  text: string;
  usage: Usage;
  finishReason: string;
}

export interface LocalModel {
  /** What this object is talking to, for a run record. */
  readonly describe: { engine: string; baseUrl: string; model: string };
  complete(request: CompletionRequest): Promise<CompletionResult>;
  structured<T>(request: StructuredRequest): Promise<StructuredResult<T>>;
  tools(request: ToolRequest): Promise<ToolResult>;
  /** What the server says it is serving. Used by `verifiedProfile`. */
  served(): Promise<{ id: string; contextLength: number | null; readFrom: string }>;
}

/**
 * Addresses a local-only deployment is allowed to talk to.
 *
 * Loopback, and nothing else. Not a private range: `10.0.0.7` is somebody
 * else's machine even when it is on your desk, and the guarantee this flag
 * makes is that the prompt did not leave *this* computer.
 */
const LOOPBACK = new Set(["127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0"]);

export class LocalOnlyViolation extends Error {
  constructor(url: string) {
    super(
      `ai-storage: AI_STORAGE_LOCAL_ONLY is set and ${url} is not a loopback address. ` +
        `Refused at construction rather than at request time: by the time a request is ` +
        `made the prompt already exists and something has decided to send it.`,
    );
    this.name = "LocalOnlyViolation";
  }
}

/** Whether the local-only guard is on. Defaults to on. */
export function localOnly(env: Record<string, string | undefined> = process.env): boolean {
  const v = env["AI_STORAGE_LOCAL_ONLY"];
  if (v === undefined) return true;
  return !(v === "0" || v.toLowerCase() === "false");
}

/**
 * Refuse a non-loopback base URL when the guard is on.
 *
 * Exported and tested on its own, because this is the whole of the privacy
 * claim and a claim that lives inside a constructor is a claim nobody can point
 * at.
 */
export function assertLocal(
  baseUrl: string,
  env: Record<string, string | undefined> = process.env,
): void {
  if (!localOnly(env)) return;
  let host: string;
  try {
    host = new URL(baseUrl).hostname;
  } catch {
    throw new LocalOnlyViolation(baseUrl);
  }
  if (!LOOPBACK.has(host)) throw new LocalOnlyViolation(baseUrl);
}

/**
 * Read a usage block the way OpenAI-compatible servers write one.
 *
 * Missing fields become zero rather than an estimate. A token count nobody
 * reported is not a token count, and the benchmark's headline number is a ratio
 * of token counts — inventing one would corrupt the only measurement this
 * component exists to make.
 */
export function usageOf(
  raw: unknown,
  latencyMs: number,
  timeToFirstTokenMs: number | null = null,
): Usage {
  const u = (raw ?? {}) as Record<string, unknown>;
  const n = (k: string): number => {
    const v = u[k];
    return typeof v === "number" && Number.isFinite(v) ? v : 0;
  };
  const details = (u["completion_tokens_details"] ?? {}) as Record<string, unknown>;
  const reasoning = typeof details["reasoning_tokens"] === "number"
    ? (details["reasoning_tokens"] as number)
    : 0;
  return {
    promptTokens: n("prompt_tokens"),
    reasoningTokens: reasoning,
    completionTokens: n("completion_tokens"),
    latencyMs,
    timeToFirstTokenMs,
  };
}

/** Parse a tool call's arguments, refusing rather than guessing. */
export function parseToolCall(raw: unknown): ToolCall {
  const c = (raw ?? {}) as Record<string, unknown>;
  const fn = (c["function"] ?? {}) as Record<string, unknown>;
  const name = typeof fn["name"] === "string" ? fn["name"] : "";
  const id = typeof c["id"] === "string" ? c["id"] : "";
  const argText = typeof fn["arguments"] === "string" ? fn["arguments"] : "";
  if (!name) throw new Error("ai-storage: a tool call arrived with no name");
  let args: Record<string, unknown>;
  try {
    args = argText ? (JSON.parse(argText) as Record<string, unknown>) : {};
  } catch {
    // Not repaired, not guessed at. A local model that emits invalid JSON for
    // its arguments has failed the call, and the harness retries it once with
    // the error — see `agents/`. Repairing it here would hide the failure rate
    // that the quantization benchmark exists to measure.
    throw new Error(
      `ai-storage: tool call "${name}" had arguments that are not JSON: ${argText.slice(0, 200)}`,
    );
  }
  if (args === null || typeof args !== "object" || Array.isArray(args))
    throw new Error(`ai-storage: tool call "${name}" had arguments that are not an object`);
  return { id, name, arguments: args };
}
