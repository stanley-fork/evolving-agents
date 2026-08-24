/**
 * llama.cpp, the reference engine.
 *
 *   llama serve -hf ggml-org/Qwen3.8-27B-GGUF:Q4_K_M \
 *     --host 127.0.0.1 --port 8080 -c 8192
 *
 * `llama-server` speaks an OpenAI-compatible API at `/v1`, which is why this
 * file is short. What is deliberate here is what it does *not* do:
 *
 * - it never retries a failed request. A local model that fails is a
 *   measurement, and the quantization benchmark is partly a measurement of how
 *   often IQ2 fails to produce schema-valid output. A retry loop inside the
 *   transport would erase that number before anything could record it. Semantic
 *   retries exist, they are bounded at two, and they live in `agents/`.
 * - it never repairs malformed JSON.
 * - it never falls back to another engine or another model.
 *
 * The one thing it does add is the clock, because latency per profile is one of
 * the results this component is for.
 */
import {
  assertLocal,
  parseToolCall,
  usageOf,
  type CompletionRequest,
  type CompletionResult,
  type LocalModel,
  type Message,
  type StructuredRequest,
  type StructuredResult,
  type ToolRequest,
  type ToolResult,
} from "./local-model.ts";

export interface LlamaCppOptions {
  /** e.g. `http://127.0.0.1:8080/v1` */
  baseUrl: string;
  /** The id to send. Should match what `served()` reports. */
  model: string;
  /** Injected for tests. Defaults to the global. */
  fetchImpl?: typeof fetch;
  /** Injected for tests. Defaults to `Date.now`. */
  now?: () => number;
  env?: Record<string, string | undefined>;
}

interface ChatChoice {
  message?: { content?: string | null; tool_calls?: unknown[] };
  finish_reason?: string;
}

interface ChatResponse {
  choices?: ChatChoice[];
  usage?: unknown;
}

/** The wire shape of a message, which is not quite our shape. */
function wire(m: Message): Record<string, unknown> {
  const out: Record<string, unknown> = { role: m.role, content: m.content };
  if (m.toolCallId) out["tool_call_id"] = m.toolCallId;
  if (m.toolCalls?.length)
    out["tool_calls"] = m.toolCalls.map((c) => ({
      id: c.id,
      type: "function",
      function: { name: c.name, arguments: JSON.stringify(c.arguments) },
    }));
  return out;
}

export class LlamaCppModel implements LocalModel {
  readonly describe: { engine: string; baseUrl: string; model: string };
  readonly #baseUrl: string;
  readonly #model: string;
  readonly #fetch: typeof fetch;
  readonly #now: () => number;

  constructor(opts: LlamaCppOptions) {
    // Before anything else, and before any prompt exists.
    assertLocal(opts.baseUrl, opts.env);
    this.#baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.#model = opts.model;
    this.#fetch = opts.fetchImpl ?? fetch;
    this.#now = opts.now ?? Date.now;
    this.describe = { engine: "llama.cpp", baseUrl: this.#baseUrl, model: this.#model };
  }

  async #post(path: string, body: unknown, timeoutMs: number): Promise<{ json: ChatResponse; ms: number }> {
    const started = this.#now();
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), timeoutMs);
    try {
      const res = await this.#fetch(this.#baseUrl + path, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: ac.signal,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(
          `ai-storage: ${this.describe.engine} at ${this.#baseUrl}${path} answered ` +
            `${res.status}: ${text.slice(0, 400)}`,
        );
      }
      return { json: (await res.json()) as ChatResponse, ms: this.#now() - started };
    } finally {
      clearTimeout(timer);
    }
  }

  async complete(request: CompletionRequest): Promise<CompletionResult> {
    const { json, ms } = await this.#post(
      "/chat/completions",
      {
        model: this.#model,
        messages: request.messages.map(wire),
        temperature: request.temperature,
        max_tokens: request.maxTokens,
        chat_template_kwargs: { enable_thinking: request.thinking },
      },
      request.timeoutMs,
    );
    const choice = json.choices?.[0];
    return {
      text: choice?.message?.content ?? "",
      usage: usageOf(json.usage, ms),
      finishReason: choice?.finish_reason ?? "unknown",
    };
  }

  async structured<T>(request: StructuredRequest): Promise<StructuredResult<T>> {
    const { json, ms } = await this.#post(
      "/chat/completions",
      {
        model: this.#model,
        messages: request.messages.map(wire),
        temperature: request.temperature,
        max_tokens: request.maxTokens,
        chat_template_kwargs: { enable_thinking: request.thinking },
        response_format: {
          type: "json_schema",
          json_schema: { name: request.schemaName, strict: true, schema: request.schema },
        },
      },
      request.timeoutMs,
    );
    const text = json.choices?.[0]?.message?.content ?? "";
    let value: T;
    try {
      value = JSON.parse(text) as T;
    } catch {
      // Constrained decoding was asked for and the reply is not JSON. That is a
      // fact about this quantization at this context length, and it is reported
      // rather than repaired.
      throw new Error(
        `ai-storage: schema "${request.schemaName}" was requested and the reply is not ` +
          `JSON: ${text.slice(0, 300)}`,
      );
    }
    return { value, usage: usageOf(json.usage, ms) };
  }

  async tools(request: ToolRequest): Promise<ToolResult> {
    const { json, ms } = await this.#post(
      "/chat/completions",
      {
        model: this.#model,
        messages: request.messages.map(wire),
        temperature: request.temperature,
        max_tokens: request.maxTokens,
        chat_template_kwargs: { enable_thinking: request.thinking },
        tools: request.tools.map((t) => ({
          type: "function",
          function: { name: t.name, description: t.description, parameters: t.parameters },
        })),
      },
      request.timeoutMs,
    );
    const choice = json.choices?.[0];
    const raw = choice?.message?.tool_calls ?? [];
    return {
      toolCalls: raw.map(parseToolCall),
      text: choice?.message?.content ?? "",
      usage: usageOf(json.usage, ms),
      finishReason: choice?.finish_reason ?? "unknown",
    };
  }

  async served(): Promise<{ id: string; contextLength: number | null; readFrom: string }> {
    const url = this.#baseUrl + "/models";
    const res = await this.#fetch(url);
    if (!res.ok) throw new Error(`ai-storage: ${url} answered ${res.status}`);
    const body = (await res.json()) as { data?: Array<Record<string, unknown>> };
    const first = body.data?.[0];
    if (!first || typeof first["id"] !== "string")
      throw new Error(`ai-storage: ${url} reported no model`);
    const meta = (first["meta"] ?? {}) as Record<string, unknown>;
    const n = meta["n_ctx_train"];
    return {
      id: first["id"] as string,
      contextLength: typeof n === "number" ? n : null,
      readFrom: url,
    };
  }
}
