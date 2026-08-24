/**
 * The model boundary: what is refused, what is measured, and what is not
 * repaired.
 *
 * No server is started here. These run against an injected `fetch`, which is
 * the only honest thing to do on a machine with no weights on it: what they
 * assert is that the *code* behaves, not that Qwen does. Everything that
 * depends on the model's behaviour lives in `bench/`, and has not been run.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  LocalOnlyViolation,
  assertLocal,
  localOnly,
  parseToolCall,
  usageOf,
} from "../src/model/local-model.ts";
import { LlamaCppModel } from "../src/model/llama-cpp.ts";
import { OllamaModel } from "../src/model/ollama.ts";
import {
  DEFAULT_PROFILE,
  EFFECTIVE_CONTEXT,
  PLATFORM_MODEL,
  PROFILES,
  profileOf,
  verifiedProfile,
  verifyAgainst,
} from "../src/model/qwen-profile.ts";

const ON = { AI_STORAGE_LOCAL_ONLY: undefined } as Record<string, string | undefined>;
const OFF = { AI_STORAGE_LOCAL_ONLY: "false" };

// ---- the platform's model --------------------------------------------------

test("the platform model comes from one file and carries where it was read", () => {
  assert.equal(PLATFORM_MODEL.value, "qwen3.8-27b");
  assert.ok(PLATFORM_MODEL.citedAs.length > 0, "a claim with no address is not a claim");
});

test("nothing in MODEL.json claims to have been verified", () => {
  // This will change the day `verify-model` is run against a live server. Until
  // then the honest state is false everywhere, and a test that says so is what
  // stops it drifting to true by accident.
  assert.equal(PLATFORM_MODEL.verified, false);
  for (const name of ["constrained", "reference", "quality"] as const)
    assert.equal(PROFILES[name].verified, false, `${name} must not claim verification`);
});

test("the effective context is 8192 whatever the weights support", () => {
  assert.equal(EFFECTIVE_CONTEXT, 8192);
  for (const name of ["constrained", "reference", "quality"] as const)
    assert.equal(
      PROFILES[name].effectiveContext,
      8192,
      "a profile that raises the cap is measuring the model's context window, not the storage",
    );
  // The reference profile deliberately allocates more than it will use.
  assert.ok(PROFILES.reference.physicalContext > PROFILES.reference.effectiveContext);
});

test("results without a profile belong to exactly one, and it is stated", () => {
  assert.equal(DEFAULT_PROFILE, "reference");
  assert.equal(profileOf(DEFAULT_PROFILE).quantization, "Q4_K_M");
  assert.throws(() => profileOf("fast"), /no profile named/);
});

test("a server serving something else is a mismatch, not a warning to ignore", () => {
  const p = profileOf("reference");
  const same = verifyAgainst(p, { id: "qwen3.8-27b", contextLength: 32768, readFrom: "x" });
  assert.deepEqual(same, []);
  // Ollama tags and repository prefixes are normalised; nothing else is.
  assert.deepEqual(
    verifyAgainst(p, { id: "ggml-org/Qwen3.8-27B:Q4_K_M", contextLength: null, readFrom: "x" }),
    [],
  );
  const other = verifyAgainst(p, { id: "llama-3.1-8b", contextLength: 8192, readFrom: "x" });
  assert.equal(other.length, 1);
  assert.equal(other[0]!.field, "id");
});

test("a server with less context than the profile needs is a mismatch", () => {
  const bad = verifyAgainst(profileOf("reference"), {
    id: "qwen3.8-27b",
    contextLength: 4096,
    readFrom: "x",
  });
  assert.equal(bad.length, 1);
  assert.equal(bad[0]!.field, "context");
});

test("verified is only reachable by looking at a server", () => {
  const p = profileOf("reference");
  assert.equal(p.verified, false);
  const v = verifiedProfile(p, {
    id: "qwen3.8-27b",
    contextLength: 32768,
    readFrom: "http://127.0.0.1:8080/v1/models",
  });
  assert.equal(v.verified, true);
  // And the citation becomes the server, not the document it was transcribed
  // from — which is the point of the field.
  assert.equal(v.citedAs, "http://127.0.0.1:8080/v1/models");
  assert.throws(
    () => verifiedProfile(p, { id: "something-else", contextLength: null, readFrom: "u" }),
    /is not serving profile/,
  );
});

// ---- the local-only guarantee ----------------------------------------------

test("local only is on unless it is explicitly turned off", () => {
  assert.equal(localOnly({}), true);
  assert.equal(localOnly({ AI_STORAGE_LOCAL_ONLY: "true" }), true);
  assert.equal(localOnly({ AI_STORAGE_LOCAL_ONLY: "1" }), true);
  assert.equal(localOnly({ AI_STORAGE_LOCAL_ONLY: "0" }), false);
  assert.equal(localOnly({ AI_STORAGE_LOCAL_ONLY: "false" }), false);
});

test("a non-loopback base URL is refused, including private ranges", () => {
  for (const url of [
    "https://openrouter.ai/api/v1",
    "https://api.openai.com/v1",
    "http://10.0.0.7:8080/v1",
    "http://192.168.1.4:11434/v1",
    "http://my-gpu-box.local:8080/v1",
    "not a url",
  ])
    assert.throws(() => assertLocal(url, ON), LocalOnlyViolation, `${url} must be refused`);

  for (const url of ["http://127.0.0.1:8080/v1", "http://localhost:11434/v1", "http://[::1]:8080/v1"])
    assert.doesNotThrow(() => assertLocal(url, ON), `${url} must be allowed`);
});

test("a private address on your own desk is still somebody else's machine", () => {
  // The guarantee is that the prompt did not leave *this* computer, not that it
  // stayed on the LAN. Written as its own test because the temptation to widen
  // this to 10/8 will come up.
  assert.throws(() => assertLocal("http://10.1.2.3:8080/v1", ON), LocalOnlyViolation);
});

test("the guard fires when the model is built, before any prompt exists", () => {
  let called = 0;
  const fetchImpl = (async () => {
    called += 1;
    return new Response("{}");
  }) as unknown as typeof fetch;
  assert.throws(
    () => new LlamaCppModel({ baseUrl: "https://openrouter.ai/api/v1", model: "m", fetchImpl, env: ON }),
    LocalOnlyViolation,
  );
  assert.equal(called, 0, "nothing may be sent in order to discover it should not be sent");
  // With the guard off it is allowed, which is what makes the flag meaningful.
  assert.doesNotThrow(
    () => new LlamaCppModel({ baseUrl: "https://openrouter.ai/api/v1", model: "m", fetchImpl, env: OFF }),
  );
});

test("Ollama refuses to run without an explicit context", () => {
  // Ollama picks a default per model; llama.cpp is started with -c 8192. A run
  // that forgot num_ctx is a run at some other context, and the whole 8K cap
  // depends on it being the same 8K everywhere.
  assert.throws(
    () => new OllamaModel({ baseUrl: "http://127.0.0.1:11434/v1", model: "m", numCtx: 0, env: ON }),
    /explicit num_ctx/,
  );
});

// ---- what is measured, and what is not repaired -----------------------------

test("a token count nobody reported is zero, never an estimate", () => {
  const u = usageOf({ prompt_tokens: 120 }, 40);
  assert.equal(u.promptTokens, 120);
  assert.equal(u.completionTokens, 0);
  assert.equal(u.reasoningTokens, 0);
  assert.equal(u.latencyMs, 40);
  assert.equal(u.timeToFirstTokenMs, null);
  // The headline result is a ratio of token counts. Inventing one corrupts the
  // only measurement this component exists to make.
  assert.equal(usageOf(undefined, 1).promptTokens, 0);
  assert.equal(usageOf({ prompt_tokens: "many" }, 1).promptTokens, 0);
});

test("reasoning tokens are read where the engine separates them", () => {
  const u = usageOf(
    { prompt_tokens: 10, completion_tokens: 90, completion_tokens_details: { reasoning_tokens: 70 } },
    5,
  );
  assert.equal(u.reasoningTokens, 70);
  assert.equal(u.completionTokens, 90);
});

test("a tool call with arguments that are not JSON is a failure, not a repair job", () => {
  // How often this happens at IQ2 is one of the results the quantization
  // benchmark is for. Repairing it here would erase the number.
  assert.throws(
    () => parseToolCall({ id: "1", function: { name: "memory_open", arguments: "{id: kn_1" } }),
    /not JSON/,
  );
  assert.throws(
    () => parseToolCall({ id: "1", function: { name: "memory_open", arguments: "[1,2]" } }),
    /not an object/,
  );
  assert.throws(() => parseToolCall({ id: "1", function: { arguments: "{}" } }), /no name/);
  const ok = parseToolCall({ id: "7", function: { name: "memory_index", arguments: '{"path":"/"}' } });
  assert.deepEqual(ok, { id: "7", name: "memory_index", arguments: { path: "/" } });
  // An empty argument string is an empty object, not a failure: some engines
  // send "" for a call with no arguments.
  assert.deepEqual(parseToolCall({ id: "8", function: { name: "memory_done", arguments: "" } }).arguments, {});
});

test("a transport error is not retried", async () => {
  let calls = 0;
  const fetchImpl = (async () => {
    calls += 1;
    return new Response("upstream is unhappy", { status: 500 });
  }) as unknown as typeof fetch;
  const m = new LlamaCppModel({ baseUrl: "http://127.0.0.1:8080/v1", model: "m", fetchImpl, env: ON });
  await assert.rejects(
    () => m.complete({ messages: [], temperature: 0, maxTokens: 10, thinking: false, timeoutMs: 100 }),
    /answered 500/,
  );
  assert.equal(calls, 1, "a retry loop in the transport erases the failure rate being measured");
});

test("a reply that is not JSON when a schema was requested is reported, not repaired", async () => {
  const fetchImpl = (async () =>
    new Response(
      JSON.stringify({ choices: [{ message: { content: "Sure! Here is the note:" } }], usage: {} }),
      { headers: { "content-type": "application/json" } },
    )) as unknown as typeof fetch;
  const m = new LlamaCppModel({ baseUrl: "http://127.0.0.1:8080/v1", model: "m", fetchImpl, env: ON });
  await assert.rejects(
    () =>
      m.structured({
        messages: [],
        temperature: 0,
        maxTokens: 10,
        thinking: false,
        timeoutMs: 100,
        schema: {},
        schemaName: "proposal",
      }),
    /is not JSON/,
  );
});

test("the request carries the thinking flag and the schema the caller asked for", async () => {
  let sent: Record<string, unknown> = {};
  const fetchImpl = (async (_url: string, init: RequestInit) => {
    sent = JSON.parse(String(init.body));
    return new Response(JSON.stringify({ choices: [{ message: { content: "{}" } }], usage: {} }), {
      headers: { "content-type": "application/json" },
    });
  }) as unknown as typeof fetch;
  const m = new LlamaCppModel({ baseUrl: "http://127.0.0.1:8080/v1", model: "qwen3.8-27b", fetchImpl, env: ON });
  await m.structured({
    messages: [{ role: "user", content: "hello" }],
    temperature: 0.1,
    maxTokens: 256,
    thinking: true,
    timeoutMs: 1000,
    schema: { type: "object" },
    schemaName: "proposal",
  });
  assert.equal(sent["model"], "qwen3.8-27b");
  assert.equal(sent["temperature"], 0.1);
  assert.deepEqual(sent["chat_template_kwargs"], { enable_thinking: true });
  const fmt = sent["response_format"] as Record<string, Record<string, unknown>>;
  assert.equal(fmt["json_schema"]!["name"], "proposal");
  assert.equal(fmt["json_schema"]!["strict"], true);
});

test("Ollama sends num_ctx on every call", async () => {
  const seen: unknown[] = [];
  const fetchImpl = (async (_url: string, init: RequestInit) => {
    seen.push(JSON.parse(String(init.body)).options);
    return new Response(JSON.stringify({ choices: [{ message: { content: "hi" } }], usage: {} }), {
      headers: { "content-type": "application/json" },
    });
  }) as unknown as typeof fetch;
  const m = new OllamaModel({
    baseUrl: "http://127.0.0.1:11434/v1",
    model: "qwen3.8-27b",
    numCtx: 8192,
    fetchImpl,
    env: ON,
  });
  const req = { messages: [], temperature: 0, maxTokens: 8, thinking: false, timeoutMs: 100 };
  await m.complete(req);
  await m.tools({ ...req, tools: [] });
  assert.deepEqual(seen, [{ num_ctx: 8192 }, { num_ctx: 8192 }]);
});
