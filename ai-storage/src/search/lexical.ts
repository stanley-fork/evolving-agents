/**
 * Exact lexical search, and no vector database in v1.
 *
 * ## Why this and not embeddings
 *
 * The predecessor project measured a closely related idea — a second retrieval
 * axis carrying real semantic information — and it came back flat at 80% acc@1
 * either way. So the burden of proof is on machinery, and the cheapest thing
 * that could possibly work goes first: an inverted index over titles, claims and
 * keywords, deterministic, local, and costing the model nothing but the
 * candidate list it asks for.
 *
 * The benchmark then asks whether Qwen loses without semantic retrieval. If it
 * does, that is the written-down, measured insufficiency that buys embeddings.
 * If it does not, embeddings were a cost with no result and this file is the
 * whole of retrieval.
 *
 * ## Why not SQLite FTS
 *
 * It would be fewer lines and a dependency whose ranking is somebody else's
 * decision. Ranking is the thing being measured here, so it is in the
 * component. The index is small — one term map per scope — and it is rebuilt
 * from the notes rather than maintained incrementally, so it cannot drift out
 * of agreement with the store.
 *
 * ## Determinism
 *
 * Same store, same query, same result, in the same order. Ties broken by note
 * id, never by insertion order or by `Map` iteration. An unrepeatable search
 * makes an unrepeatable benchmark, and this component exists to produce a
 * number somebody else can reproduce.
 *
 * Pure. No fetch, no clock, no DOM.
 */
import type { KnowledgeNote } from "../knowledge/schema.ts";

/**
 * Words, lowercased, split on anything that is not a letter or a digit.
 *
 * Deliberately not stemmed. A stemmer is a guess about morphology that differs
 * per language and would make `indexing` and `indexed` the same term while the
 * store's own vocabulary — `IQ2_M`, `Q4_K_M`, `n_ctx` — is exactly the kind of
 * token a stemmer mangles. Exact means exact.
 */
export function tokenise(text: string): string[] {
  const out: string[] = [];
  for (const raw of text.toLowerCase().split(/[^\p{L}\p{N}_]+/u)) if (raw) out.push(raw);
  return out;
}

export interface Hit {
  id: string;
  /** How many of the query's terms this note matched. */
  matched: number;
  /** The score, and the three counts it is made of. */
  score: number;
  inTitle: number;
  inKeywords: number;
  inClaim: number;
}

interface Posting {
  inTitle: number;
  inKeywords: number;
  inClaim: number;
}

/**
 * Weights, stated rather than tuned.
 *
 * A keyword is a word the model chose to attach to the note, so it is a
 * stronger signal than a word that happens to appear in the prose. A title is
 * chosen too and shorter. These three numbers are the whole ranking model, they
 * are in one place, and `bench/navigation` is what moves them — not taste.
 */
export const WEIGHTS = { title: 3, keyword: 4, claim: 1 } as const;

export class LexicalIndex {
  readonly #terms = new Map<string, Map<string, Posting>>();
  readonly #ids = new Set<string>();

  /** Build from notes. Rebuilt rather than maintained, so it cannot drift. */
  static of(notes: readonly KnowledgeNote[]): LexicalIndex {
    const ix = new LexicalIndex();
    for (const n of notes) ix.add(n);
    return ix;
  }

  add(note: KnowledgeNote): void {
    this.#ids.add(note.id);
    const bump = (term: string, field: keyof Posting) => {
      let byId = this.#terms.get(term);
      if (!byId) this.#terms.set(term, (byId = new Map()));
      const p = byId.get(note.id) ?? { inTitle: 0, inKeywords: 0, inClaim: 0 };
      p[field] += 1;
      byId.set(note.id, p);
    };
    for (const t of tokenise(note.title)) bump(t, "inTitle");
    for (const k of note.keywords) for (const t of tokenise(k)) bump(t, "inKeywords");
    for (const t of tokenise(note.claim)) bump(t, "inClaim");
  }

  get size(): number {
    return this.#ids.size;
  }

  get terms(): number {
    return this.#terms.size;
  }

  /**
   * Search, and return the counts behind the score.
   *
   * The counts travel with the score because a score on its own cannot be
   * checked. A run record that says `0.83` explains nothing; one that says
   * *matched 2 of 3 terms, twice in the keywords* can be argued with.
   *
   * Notes matching more of the query's terms always rank above notes matching
   * fewer, whatever the weights say. A note that mentions one query word forty
   * times is not a better answer than one that mentions all three once, and a
   * pure weighted sum gets that wrong.
   */
  search(query: string, limit = 10): Hit[] {
    const terms = [...new Set(tokenise(query))].sort();
    if (!terms.length) return [];
    const acc = new Map<string, Hit>();
    for (const t of terms) {
      const byId = this.#terms.get(t);
      if (!byId) continue;
      for (const [id, p] of [...byId.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
        const h =
          acc.get(id) ?? { id, matched: 0, score: 0, inTitle: 0, inKeywords: 0, inClaim: 0 };
        h.matched += 1;
        h.inTitle += p.inTitle;
        h.inKeywords += p.inKeywords;
        h.inClaim += p.inClaim;
        h.score =
          h.inTitle * WEIGHTS.title + h.inKeywords * WEIGHTS.keyword + h.inClaim * WEIGHTS.claim;
        acc.set(id, h);
      }
    }
    return [...acc.values()].sort(
      (a, b) => b.matched - a.matched || b.score - a.score || (a.id < b.id ? -1 : 1),
    ).slice(0, limit);
  }

  /**
   * Notes sharing the most keywords with this one.
   *
   * `memory_related` is built on this. Keywords only, not the claim: two notes
   * whose prose shares the word "the" are not related, and the keyword list is
   * the part a model deliberately chose.
   */
  relatedTo(note: KnowledgeNote, limit = 5): Hit[] {
    return this.search(note.keywords.join(" "), limit + 1).filter((h) => h.id !== note.id).slice(0, limit);
  }
}
