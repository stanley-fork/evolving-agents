/**
 * Models that are not models: a script, and an oracle.
 *
 * ## Why these exist
 *
 * No weights on this machine, and the whole loop — tools, budget, loop
 * detection, endings — has to be testable anyway. So the loop takes a
 * `LocalModel` and these implement it without generating anything.
 *
 * **`ScriptedModel`** replays a fixed sequence of replies. It is how the loop's
 * failure endings get tested: a malformed reply, a prose answer, a repeat, a
 * refusal to stop. Those are the paths a real run hits at IQ2 and the ones a
 * benchmark most needs to tell apart.
 *
 * **`OracleNavigator`** plays the index perfectly: it reads the directory it is
 * shown, descends the branch whose name best matches the question, opens the
 * one note that answers, and stops. It cheats — it knows the answer string —
 * and that is the point.
 *
 * ## What the oracle is for, and what it is not
 *
 * It is the **ceiling**. The Navigation Efficiency Ratio it achieves is the
 * best the store's shape allows: perfect navigation, nothing wasted. A real
 * model's ratio read against it separates two questions that otherwise blur —
 * *is the store navigable* and *can this model navigate it*. A corpus where the
 * oracle scores badly is a corpus with a bad index, and no model will save it.
 *
 * It is **not** a result about any model, and nothing may report it as one. It
 * is labelled `oracle` in every run record for that reason.
 */
import type {
  CompletionRequest,
  CompletionResult,
  LocalModel,
  StructuredRequest,
  StructuredResult,
  ToolRequest,
  ToolResult,
  Usage,
} from "./local-model.ts";

const noUsage = (): Usage => ({
  promptTokens: 0,
  reasoningTokens: 0,
  completionTokens: 0,
  latencyMs: 0,
  timeToFirstTokenMs: null,
});

export type ScriptedReply =
  | { kind: "tools"; calls: Array<{ name: string; arguments: Record<string, unknown> }> }
  | { kind: "text"; text: string }
  | { kind: "throw"; message: string };

export class ScriptedModel implements LocalModel {
  readonly describe = { engine: "scripted", baseUrl: "none", model: "scripted" };
  #i = 0;
  readonly #script: ScriptedReply[];
  /** Every request the loop made, so a test can assert what was sent. */
  readonly seen: ToolRequest[] = [];

  constructor(script: ScriptedReply[]) {
    this.#script = script;
  }

  async tools(request: ToolRequest): Promise<ToolResult> {
    this.seen.push(request);
    const next = this.#script[this.#i++] ?? { kind: "text", text: "" };
    if (next.kind === "throw") throw new Error(next.message);
    if (next.kind === "text")
      return { toolCalls: [], text: next.text, usage: noUsage(), finishReason: "stop" };
    return {
      toolCalls: next.calls.map((c, i) => ({ id: `c${this.#i}_${i}`, name: c.name, arguments: c.arguments })),
      text: "",
      usage: noUsage(),
      finishReason: "tool_calls",
    };
  }

  async complete(): Promise<CompletionResult> {
    throw new Error("ScriptedModel: complete() is not scripted");
  }
  async structured<T>(): Promise<StructuredResult<T>> {
    throw new Error("ScriptedModel: structured() is not scripted");
  }
  async served() {
    return { id: "scripted", contextLength: null, readFrom: "none" };
  }
}

/**
 * A model shaped like one that navigates perfectly.
 *
 * The strategy, in full:
 *
 * 1. `memory_index('/')`.
 * 2. From whatever listing came back, descend into the directory whose name
 *    shares the most words with the question. Repeat until the listing has
 *    notes in it.
 * 3. Open notes in the order their hints match the question, stopping at the
 *    first one that contains the answer.
 * 4. `memory_done`.
 *
 * Step 3 is where it cheats: a real model has to judge from the hint alone,
 * and this one checks the answer. Everything before it is the navigation the
 * store's shape either supports or does not, which is the thing being measured.
 */
export class OracleNavigator implements LocalModel {
  readonly describe = { engine: "oracle", baseUrl: "none", model: "oracle" };
  readonly #question: string;
  readonly #answer: string;
  readonly #words: string[];
  /** Directories still worth visiting, best match first — globally, not locally. */
  #frontier: Array<{ path: string; score: number }> = [];
  /** Notes worth opening, best hint first. */
  #queue: string[] = [];
  #visited = new Set<string>();
  #opened = new Set<string>();
  #at = "/";
  #searched = false;
  #have = new Set<string>();

  constructor(question: string, answer: string) {
    this.#question = question;
    this.#answer = answer;
    this.#words = keywordsOf(question);
  }

  async tools(request: ToolRequest): Promise<ToolResult> {
    const last = request.messages[request.messages.length - 1];
    const result = last?.role === "tool" ? last.content : null;
    const have = new Set(request.tools.map((t) => t.name));
    this.#have = have;
    const call = (name: string, args: Record<string, unknown>): ToolResult => ({
      toolCalls: [{ id: "o" + request.messages.length, name, arguments: args }],
      text: "",
      usage: noUsage(),
      finishReason: "tool_calls",
    });

    /**
     * Use the tools this arm actually has.
     *
     * The first version always opened with `memory_index`, which the flat
     * baseline does not have — so every flat run answered nothing and the
     * summary reported `done` with zero tokens loaded and zero correct. That is
     * not a baseline, it is an empty column, and a benchmark with an empty
     * column silently flatters everything beside it.
     */
    if (result === null) {
      if (have.has("memory_index")) {
        this.#visited.add("/");
        return call("memory_index", { path: "/" });
      }
      if (have.has("memory_all")) return call("memory_all", {});
      return call("memory_find_exact", { query: this.#words.join(" "), limit: 5 });
    }

    if (result.includes(this.#answer)) {
      const id = /^(kn_[a-z0-9-]+_[0-9a-f]+)/.exec(result)?.[1];
      return call("memory_done", { answer: this.#answer, cites: id ? [id] : [] });
    }
    // The flat arm has exactly one move and it has been made. Anything after
    // this is the model insisting; there is nothing else to call.
    if (have.has("memory_all") && have.size === 2)
      return call("memory_done", { answer: "", cites: [] });

    if (result.startsWith("/")) {
      // A listing. Queue the directories that might hold the answer, and the
      // notes whose hint actually says something about the question.
      const { dirs, notes } = parseListing(result);
      const base = this.#at === "/" ? "" : this.#at;
      /**
       * Best-first, not depth-first.
       *
       * The first version pushed a listing's children to the front of the
       * frontier, so one wrong turn at the top meant exhausting that whole
       * subtree before trying the sibling that actually matched. `keys` and
       * `rotation` both score on a question about key rotation; picking `keys`
       * first is a coin flip, and a navigator that cannot recover from a coin
       * flip is not a ceiling worth measuring against. Ranking the whole
       * frontier means a depth-two directory scoring 2 beats a depth-three one
       * scoring 0, whichever was discovered first.
       */
      for (const d of dirs) {
        const path = base + "/" + d;
        if (this.#visited.has(path)) continue;
        this.#frontier.push({ path, score: overlap(d, this.#words) });
      }
      this.#frontier.sort((a, b) => b.score - a.score || (a.path < b.path ? -1 : 1));

      /**
       * A hint that says nothing buys nothing.
       *
       * The first version opened every note in a directory it had descended
       * into, one at a time, and burned the step cap doing it. That is what a
       * *bad* navigator does, and this one is supposed to be the ceiling: the
       * index gives each note a hint, and a hint with no overlap is the index
       * saying "not this one". Believe it, and go back up.
       */
      const scored = notes.map((n) => ({
        n,
        score: overlap(n.name + " " + n.hint, this.#words),
      }));
      const best = Math.max(0, ...scored.map((x) => x.score));
      /**
       * Only the best matches in this listing, and only if there are any.
       *
       * Matching one common word is not a reason to pay for a note. In a
       * directory called `keys`, every filler entry matches "key" and none of
       * them answers anything — so a navigator that opens all of them has
       * confused *being in the right area* with *being the right note*. Taking
       * only the entries at the listing's best score skips two hundred of those
       * and costs nothing when the best score is what the answer scores.
       */
      const promising = best > 0 ? scored.filter((x) => x.score === best).slice(0, 3) : [];
      this.#queue = [...promising.map((x) => x.n.name), ...this.#queue];
    } else if (result.includes("  matched ")) {
      /**
       * Search hits, and only search hits.
       *
       * The first version matched any line starting with an id, which a *note*
       * render also does — so opening a note queued that same note again, and
       * the run alternated between an open and a REPEATED_TOOL_LOOP until the
       * step cap. `  matched ` appears in a hit line and nowhere else.
       */
      this.#queue = [
        ...[...result.matchAll(/^(kn_[a-z0-9-]+_[0-9a-f]+)/gm)].map((m) => m[1]!),
        ...this.#queue,
      ];
    }

    return this.#next(call);
  }

  #next(call: (n: string, a: Record<string, unknown>) => ToolResult): ToolResult {
    let id = this.#queue.shift();
    // Never twice. A perfect navigator does not pay for the same note again,
    // and letting it would measure the loop detector rather than the store.
    while (id && this.#opened.has(id)) id = this.#queue.shift();
    if (id) {
      this.#opened.add(id);
      return call("memory_open", { id });
    }
    const next = this.#frontier.shift();
    if (next) {
      this.#visited.add(next.path);
      this.#at = next.path;
      return call("memory_index", { path: next.path });
    }
    if (!this.#searched && this.#have.has("memory_find_exact")) {
      this.#searched = true;
      return call("memory_find_exact", { query: this.#words.join(" "), limit: 5 });
    }
    return call("memory_done", { answer: "", cites: [] });
  }

  async complete(): Promise<CompletionResult> {
    throw new Error("OracleNavigator: complete() is not implemented");
  }
  async structured<T>(): Promise<StructuredResult<T>> {
    throw new Error("OracleNavigator: structured() is not implemented");
  }
  async served() {
    return { id: "oracle", contextLength: null, readFrom: "none" };
  }
}

function keywordsOf(question: string): string[] {
  const stop = new Set([
    "what", "which", "does", "the", "for", "use", "is", "a", "an", "of", "and", "to", "do",
    "project", "using",
  ]);
  return question
    .toLowerCase()
    .split(/[^\p{L}\p{N}_-]+/u)
    .filter((w) => w && !stop.has(w));
}

function parseListing(text: string): { dirs: string[]; notes: Array<{ name: string; hint: string }> } {
  const dirs: string[] = [];
  const notes: Array<{ name: string; hint: string }> = [];
  for (const raw of text.split("\n").slice(1)) {
    const line = raw.trim();
    if (!line) continue;
    if (line.includes("/")) {
      dirs.push(line.split("/")[0]!.trim());
      continue;
    }
    const [name, ...rest] = line.split(/\s{2,}/);
    notes.push({ name: name!.trim(), hint: rest.join(" ") });
  }
  return { dirs, notes };
}

/**
 * How many of the question's words a name speaks to.
 *
 * Matched on segments with a prefix rule rather than by substring, because the
 * names in an index are abbreviated and the query's are not. A bucket called
 * `dep-rot` is exactly where a question about *deployment rotation* should go,
 * and `"dep-rot".includes("rotation")` is false — so a substring test scored
 * the right directory at zero and sent the navigator somewhere else.
 */
function overlap(text: string, words: string[]): number {
  const segs = text.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean);
  let n = 0;
  for (const w of words) {
    const head = w.slice(0, 4);
    if (segs.some((s) => s.startsWith(head) || w.startsWith(s.slice(0, 4)))) n += 1;
  }
  return n;
}

function rankByName(names: string[], question: string): string[] {
  const words = keywordsOf(question);
  return [...names].sort((a, b) => overlap(b, words) - overlap(a, words) || (a < b ? -1 : 1));
}

function rankByHint(
  notes: Array<{ name: string; hint: string }>,
  question: string,
): Array<{ name: string; hint: string }> {
  const words = keywordsOf(question);
  return [...notes].sort(
    (a, b) => overlap(b.hint, words) - overlap(a.hint, words) || (a.name < b.name ? -1 : 1),
  );
}
