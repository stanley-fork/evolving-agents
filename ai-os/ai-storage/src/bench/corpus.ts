/**
 * A synthetic store, and the fact hidden in it.
 *
 * ## What the benchmark actually asks
 *
 * Plant a fact nobody could guess — *the deployment key for project Sigma uses
 * rotation policy Zeta-17* — in one note out of ten thousand, hand the model the
 * question and the tools and nothing else, and see whether it comes back with
 * `Zeta-17` and how many tokens it had to read to do it.
 *
 * The answer has to be **unguessable**. That is why the planted values are
 * nonsense pairs rather than plausible ones: a model that answers "the key
 * rotates every 90 days" from its priors would score as a success against a
 * realistic fact, and the whole measurement would be of pretraining rather than
 * of navigation.
 *
 * ## Why the filler has to be plausible
 *
 * Ten thousand notes saying `lorem ipsum 4127` are ten thousand notes a model
 * can skip on sight, and the benchmark would measure nothing. The filler here
 * is drawn from the same vocabulary as the planted note and lands in the same
 * directories, so finding the answer means *reading the index properly* rather
 * than spotting the one note that looks different.
 *
 * ## Deterministic
 *
 * Seeded, so the same seed builds byte-identical notes with identical ids. A
 * benchmark whose corpus differs per run cannot be compared to itself, let
 * alone to a baseline.
 *
 * Pure. No fetch, no clock, no DOM, no filesystem.
 */
import { createHash } from "node:crypto";
import { LexicalIndex } from "../search/lexical.ts";
import { slugify } from "../knowledge/repository.ts";
import type { KnowledgeNote, ScopeType } from "../knowledge/schema.ts";
import {
  DEFAULT_LIMITS,
  splitUntilFits,
  type IndexLimits,
  type IndexNode,
} from "../index/tree.ts";
import type { TokenCounter } from "../context/budget.ts";
import type { MemoryView } from "../tools/memory.ts";

/** A small deterministic PRNG, so a seed reproduces a corpus exactly. */
function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 0x1_0000_0000;
  };
}

const AREAS = [
  "architecture",
  "operations",
  "experiments",
  "failures",
  "permissions",
  "deployment",
] as const;

const TOPICS = [
  "agent-capabilities",
  "model-routing",
  "storage-boundaries",
  "tool-permissions",
  "scope-model",
  "context-budget",
  "retrieval",
  "quantization",
  "compaction",
  "provenance",
  "tool-loops",
  "rotation",
  "keys",
  "backups",
  "runners",
  "schemas",
];

const VERBS = [
  "must never",
  "is required to",
  "does not",
  "may only",
  "was measured to",
  "is configured to",
];

const OBJECTS = [
  "derive progress from the largest source offset",
  "exceed its declared token budget",
  "hold a write capability",
  "run against a remote endpoint",
  "retry a malformed reply more than twice",
  "resolve a path by concatenation",
  "keep a superseded claim out of the index",
  "share a scope with another project",
];

export interface PlantedFact {
  /** The question to ask. */
  question: string;
  /** The exact string an answer must contain. */
  answer: string;
  /** The note that holds it. */
  noteId: string;
  /** Where it sits, so a failed run can be read afterwards. */
  path: string;
}

export interface Corpus {
  notes: KnowledgeNote[];
  nodes: IndexNode[];
  planted: PlantedFact[];
  /** Tokens the whole corpus would cost if it were pasted into a prompt. */
  corpusTokens: number;
  seed: number;
}

export interface CorpusOptions {
  size: number;
  seed?: number;
  /** How many facts to plant. Each is asked as its own task. */
  plant?: number;
  counter: TokenCounter;
  limits?: IndexLimits;
}

/** The whole text of a note, which is what "corpus tokens" counts. */
export function noteText(n: KnowledgeNote): string {
  return `${n.title}\n${n.keywords.join(" ")}\n${n.claim}`;
}

export function buildCorpus(opts: CorpusOptions): Corpus {
  const seed = opts.seed ?? 20260824;
  const rand = rng(seed);
  const pick = <T>(xs: readonly T[]): T => xs[Math.floor(rand() * xs.length)]!;
  const notes: KnowledgeNote[] = [];
  const at = "2026-01-01T00:00:00.000Z";

  const make = (
    area: string,
    topic: string,
    title: string,
    claim: string,
    keywords: string[],
  ): KnowledgeNote => {
    // The same id shape the real store mints, for the same reason: an index of
    // opaque ids is an index nothing can navigate. See repository.mintId.
    const id =
      `kn_${slugify(title)}_` +
      createHash("sha256").update(`${seed}\0${area}\0${topic}\0${title}\0${claim}`).digest("hex").slice(0, 8);
    return {
      version: 1,
      id,
      scope: { type: "project", id: "bench" },
      title,
      type: "fact",
      claim,
      keywords,
      evidence: [
        {
          sourceId: "src_bench",
          path: `sources/${area}/${topic}.md`,
          from: 0,
          to: Math.max(1, claim.length),
          sha256: createHash("sha256").update(claim).digest("hex"),
        },
      ],
      relations: { related: [], supersedes: [] },
      state: "active",
      createdAt: at,
      updatedAt: at,
      revision: "rev1",
    };
  };

  // The planted facts first, so their placement does not depend on how many
  // filler notes happen to precede them.
  const plantCount = opts.plant ?? 3;
  const planted: PlantedFact[] = [];
  const SUBJECTS = ["Sigma", "Thule", "Marrow", "Pallas", "Quill", "Verdigris"];
  const POLICIES = ["Zeta-17", "Kappa-4", "Orrery-92", "Bellwether-3", "Nimbus-58", "Larkspur-6"];
  for (let i = 0; i < plantCount; i += 1) {
    const subject = SUBJECTS[i % SUBJECTS.length]!;
    const policy = POLICIES[i % POLICIES.length]!;
    /**
     * Where a reader would look, not somewhere arbitrary.
     *
     * The first version planted each fact in `AREAS[i % AREAS.length]`, which
     * put a note about deployment key rotation under `architecture/`. That is
     * not a hard benchmark, it is an unfair one: navigation *cannot* find a note
     * filed under a name that contradicts its content, so the run would have
     * measured the search tool and reported it as a result about hierarchies.
     *
     * The filler spreads across every area including this one, so the directory
     * is not a giveaway — there are hundreds of notes in it and only one answers.
     */
    const area = "deployment";
    const topic = ["rotation", "keys", "backups"][i % 3]!;
    const title = `Deployment key rotation for project ${subject}`;
    const claim =
      `The deployment key for project ${subject} uses rotation policy ${policy}. ` +
      `No other project uses ${policy}, and ${subject} uses no other policy.`;
    const note = make(area, topic, title, claim, ["deployment", "key", "rotation", subject.toLowerCase()]);
    notes.push(note);

    /**
     * Decoys, and why a benchmark without them measures the wrong thing.
     *
     * The first run of this benchmark had exact search winning outright — 3/3 at
     * every size, up to 13,000×, against a hierarchy that scored 1/3. Read
     * literally that says grep beats navigation. Read carefully it says the
     * question shared a rare literal token with exactly one note, so search had
     * only to match the words and could not be wrong.
     *
     * No real store is like that. A store that has been written in for a year
     * has ten notes about project Sigma's deployment keys, nine of which point
     * somewhere else, mention an old policy, or record a decision not to use
     * one. Finding the answer means knowing *which* — and that is the question
     * the two arms actually disagree about.
     *
     * So each planted fact gets decoys: same words, same keywords, scattered
     * across other areas, none of them carrying the answer. Search now has to
     * open several. Navigation has to reach the right directory. Whichever wins
     * from here has won something.
     */
    const DECOY_AREAS = AREAS.filter((a) => a !== area);
    for (let d = 0; d < 8; d += 1) {
      const dArea = DECOY_AREAS[d % DECOY_AREAS.length]!;
      const dTopic = ["rotation", "keys", "backups", "runners"][d % 4]!;
      notes.push(
        make(
          dArea,
          dTopic,
          `Deployment key rotation for project ${subject} — ${["superseded", "proposed", "withdrawn", "duplicate"][d % 4]} (${d})`,
          `An earlier note on the deployment key rotation for project ${subject}. ` +
            `It records no policy name: the decision was taken elsewhere and this entry is ` +
            `kept only so the discussion can be found. Do not read a policy out of it.`,
          ["deployment", "key", "rotation", subject.toLowerCase()],
        ),
      );
    }
    planted.push({
      question: `What rotation policy does the deployment key for project ${subject} use?`,
      answer: policy,
      noteId: note.id,
      path: `/${area}/${topic}`,
    });
  }

  // Filler, from the same vocabulary and the same directories.
  while (notes.length < opts.size) {
    const area = pick(AREAS);
    const topic = pick(TOPICS);
    const subject = pick(SUBJECTS);
    const i = notes.length;
    const title = `${topic} — ${VERBS[i % VERBS.length]} ${OBJECTS[i % OBJECTS.length]}`.slice(0, 150);
    const claim =
      `In ${area}, a ${topic} ${pick(VERBS)} ${pick(OBJECTS)}. ` +
      `This was recorded while working on project ${subject} and applies to that scope.`;
    notes.push(make(area, topic, `${title} (${i})`, claim, [area, topic, subject.toLowerCase()]));
  }

  // The index: one directory per area, one per topic under it.
  const byPath = new Map<string, IndexNode>();
  const ensure = (path: string): IndexNode => {
    let n = byPath.get(path);
    if (!n) byPath.set(path, (n = { path, entries: [] }));
    return n;
  };
  ensure("/");
  for (const note of notes) {
    const e = note.evidence[0]!;
    const [, area, file] = /^sources\/([^/]+)\/([^/]+)\.md$/.exec(e.path)!;
    const areaPath = `/${area}`;
    const topicPath = `${areaPath}/${file}`;
    ensure(areaPath);
    ensure(topicPath).entries.push({ name: note.id, kind: "note", hint: shortHint(note.title) });
  }
  for (const path of [...byPath.keys()]) {
    if (path === "/") continue;
    const parts = path.split("/").filter(Boolean);
    const parent = "/" + parts.slice(0, -1).join("/");
    const node = ensure(parent === "//" ? "/" : parent);
    const name = parts[parts.length - 1]!;
    if (!node.entries.some((x) => x.kind === "dir" && x.name === name))
      node.entries.push({ name, kind: "dir" });
  }

  const limits = opts.limits ?? DEFAULT_LIMITS;
  const nodes: IndexNode[] = [];
  for (const node of byPath.values()) {
    const { nodes: split } = splitUntilFits(node, opts.counter, limits);
    nodes.push(...split);
  }
  // Splitting a topic directory adds children the parent listing has to know
  // about; splitUntilFits already rewrote the parent, so merging by path keeps
  // the rewritten version rather than the original.
  const merged = new Map<string, IndexNode>();
  for (const n of nodes) merged.set(n.path, n);

  const corpusTokens = notes.reduce((t, n) => t + opts.counter.count(noteText(n)), 0);
  return { notes, nodes: [...merged.values()], planted, corpusTokens, seed };
}

function shortHint(title: string): string {
  const t = title.split(" — ")[0]!;
  return t.length > 28 ? t.slice(0, 27) + "…" : t;
}

/** A `MemoryView` over a corpus held in memory. */
export function viewOf(corpus: Corpus): MemoryView {
  const byPath = new Map(corpus.nodes.map((n) => [n.path, n]));
  const byId = new Map(corpus.notes.map((n) => [n.id, n]));
  let index: LexicalIndex | null = null;
  return {
    async nodeAt(path) {
      return byPath.get(path) ?? null;
    },
    async noteById(id) {
      const note = byId.get(id);
      return note ? { note, scope: note.scope.type as ScopeType } : null;
    },
    async index() {
      return (index ??= LexicalIndex.of(corpus.notes));
    },
    async sourceSlice(note, i) {
      // The synthetic store's "source" is the claim itself. Enough to exercise
      // the tool; the filesystem view reads real bytes.
      return note.evidence[i] ? note.claim : null;
    },
  };
}
