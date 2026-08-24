/**
 * Ollama, the compatible engine.
 *
 *   ollama serve
 *   ollama run <the platform model>
 *
 * Ollama exposes an OpenAI-compatible surface at `/v1`, so most of this is the
 * same request llama.cpp gets. Three things differ and each one is a place a
 * benchmark could quietly stop comparing like with like:
 *
 * - **The context length is per-request, not per-server.** llama.cpp is started
 *   with `-c 8192` and that is that; Ollama decides per model unless told, so
 *   `num_ctx` is sent on every call. A run that forgot it would be a run at
 *   whatever Ollama's default happens to be, and the whole point of the 8K cap
 *   is that it is the same 8K everywhere.
 * - **The id carries a tag.** `qwen3.8-27b:q4_k_m` where llama.cpp says
 *   `qwen3.8-27b`. `verifyAgainst` normalises the tag and nothing else.
 * - **Structured output goes in `format`,** not `response_format`, on the
 *   native endpoint. The `/v1` endpoint takes `response_format`, so that is
 *   what this uses — one code path rather than two.
 *
 * Same refusals as the reference engine: no transport retries, no JSON repair,
 * no fallback.
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

export interface OllamaOptions {
  /** e.g. `http://127.0.0.1:11434/v1` */
  baseUrl: string;
  model: string;
  /** The per-request context. Must be the profile's effective context. */
  numCtx: number;
  fetchImpl?: typeof fetch;
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

export class OllamaModel implements LocalModel {
  readonly describe: { engine: string; baseUrl: string; model: string };
  readonly #baseUrl: string;
  readonly #model: string;
  readonly #numCtx: number;
  readonly #fetch: typeof fetch;
  readonly #now: () => number;

  constructor(opts: OllamaOptions) {
    assertLocal(opts.baseUrl, opts.env);
    if (!(opts.numCtx > 0))
      throw new Error("ai-storage: Ollama needs an explicit num_ctx — see the note in this file");
    this.#baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.#model = opts.model;
    this.#numCtx = opts.numCtx;
    this.#fetch = opts.fetchImpl ?? fetch;
    this.#now = opts.now ?? Date.now;
    this.describe = { engine: "ollama", baseUrl: this.#baseUrl, model: this.#model };
  }

  async #post(body: Record<string, unknown>, timeoutMs: number) {
    const started = this.#now();
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), timeoutMs);
    try {
      const res = await this.#fetch(this.#baseUrl + "/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        // num_ctx on every call. See the note at the top.
        body: JSON.stringify({ ...body, model: this.#model, options: { num_ctx: this.#numCtx } }),
        signal: ac.signal,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(
          `ai-storage: ollama at ${this.#baseUrl} answered ${res.status}: ${text.slice(0, 400)}`,
        );
      }
      return { json: (await res.json()) as ChatResponse, ms: this.#now() - started };
    } finally {
      clearTimeout(timer);
    }
  }

  async complete(request: CompletionRequest): Promise<CompletionResult> {
    const { json, ms } = await this.#post(
      {
        messages: request.messages.map(wire),
        temperature: request.temperature,
        max_tokens: request.maxTokens,
        think: request.thinking,
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
      {
        messages: request.messages.map(wire),
        temperature: request.temperature,
        max_tokens: request.maxTokens,
        think: request.thinking,
        response_format: {
          type: "json_schema",
          json_schema: { name: request.schemaName, strict: true, schema: request.schema },
        },
      },
      request.timeoutMs,
    );
    const text = json.choices?.[0]?.message?.content ?? "";
    try {
      return { value: JSON.parse(text) as T, usage: usageOf(json.usage, ms) };
    } catch {
      throw new Error(
        `ai-storage: schema "${request.schemaName}" was requested and the reply is not ` +
          `JSON: ${text.slice(0, 300)}`,
      );
    }
  }

  async tools(request: ToolRequest): Promise<ToolResult> {
    const { json, ms } = await this.#post(
      {
        messages: request.messages.map(wire),
        temperature: request.temperature,
        max_tokens: request.maxTokens,
        think: request.thinking,
        tools: request.tools.map((t) => ({
          type: "function",
          function: { name: t.name, description: t.description, parameters: t.parameters },
        })),
      },
      request.timeoutMs,
    );
    const choice = json.choices?.[0];
    return {
      toolCalls: (choice?.message?.tool_calls ?? []).map(parseToolCall),
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
    const hit =
      body.data?.find((m) => String(m["id"] ?? "").toLowerCase().startsWith(this.#model.toLowerCase())) ??
      body.data?.[0];
    if (!hit || typeof hit["id"] !== "string")
      throw new Error(`ai-storage: ${url} reported no model`);
    // Ollama's /v1/models does not report a context length. Saying null is the
    // honest answer; the request-level num_ctx is what actually binds.
    return { id: hit["id"] as string, contextLength: null, readFrom: url };
  }
}
