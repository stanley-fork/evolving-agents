# 22 · ai-storage on a local model — the specification, and what is built

> **Status, 2026-08-24.** Phases 1 and 2 are built and tested: the model
> boundary, the local-only guarantee, the context invariant, the note schema,
> the provenance pipeline, derived progress, and the token-bounded index. The
> agents (Librarian, Archivist, Indexer, Reconciler, MemoryKeeper) and every
> benchmark are **not built**. Nothing here has been run against weights — see
> [§0](#0).

<a id="0"></a>

## 0. What has not been checked, before anything else

The model this document names is `qwen3.8-27b`, and **this repository has not
verified that it exists.** The specification arrived with sizes, a licence, a
context maximum and a GGUF repository; all of them were transcribed onto a
machine whose egress proxy answers `403` to `CONNECT huggingface.co`. So:

- every field in [`MODEL.json`](../MODEL.json) carries `verified: false`;
- a test asserts they all still do, so the flag cannot drift to true by
  accident;
- [`ai-storage/scripts/verify-model.ts`](../ai-storage/scripts/verify-model.ts)
  is the only thing that may report otherwise, and it does it by asking a
  running server what it is serving.

This is the same rule the rest of the repository runs on — a claim carries the
address it was read from — applied to the claim that decides what every other
number means. A benchmark result recorded under the wrong weights is worse than
one that does not exist, because it looks like evidence.

If the identifier turns out to be wrong, one file changes.

<a id="1"></a>

## 1. The question

Not *how large a context can the model hold*. This:

> How much capability can a good operating system recover by giving a
> relatively small local model better memory, navigation and tools?

The architecture exists to make the question answerable:

```text
               project knowledge
              100K / 1M / 10M tokens
                       │
                       ▼
                  ai-storage
                       │
              navigable hierarchy
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
       INDEX                     shards
          │                         │
          └──────────┬──────────────┘
                     ▼
                selected notes
                     │
                  ~2-4K
                     │
                     ▼
                 the model
                   LOCAL
                    8K
```

This is [doc/11](11-choosing-a-model.md)'s argument one layer down. That document
says the number that decides anything is the **interaction term** — how much
more the harness lifts the small model than it lifts the frontier one — and that
a single pooled lift is not a finding. ai-storage is one term of that harness,
and §17 below is the benchmark that isolates it.

<a id="2"></a>

## 2. The model, and why the context is capped anyway

`qwen3.8-27b`: 27B dense parameters, open weights, Apache 2.0 on the GGUF
distribution, native tool calling, structured output, a native context reported
up to 1,000,000 tokens, and practical local quantizations. Reference weights
`ggml-org/Qwen3.8-27B-GGUF`, `Q4_K_M` at approximately 19 GB.

**Every sentence in that paragraph is transcribed and none of it is verified.**
See [§0](#0).

ai-storage must not depend on the million-token context. The benchmark caps the
model at

```text
8,192 tokens
```

because storage should solve context scarcity rather than hide it. Run the
benchmark at the native maximum and you have measured the model's context
window and learned nothing about the storage layer.

<a id="3"></a>

## 3. Profile — constrained

For a 16 GB machine. **Not the quality reference**; it exists to answer a
separate question, in [§18](#18).

```yaml
quantization: IQ2_M          # ~10.87 GB
context:
  physical_max: 8192
  effective_max: 8192
```

<a id="4"></a>

## 4. Profile — reference

Apple Silicon ≥ 32 GB, or an NVIDIA GPU ≥ 24 GB VRAM.

```yaml
quantization: Q4_K_M         # ~19 GB
context:
  physical_max: 32768
  effective_max: 8192
```

The hardware allows more context and the benchmark still refuses it. **A result
that does not name a profile means this one** — stated rather than inferred,
because a benchmark number whose quantization is unknown cannot be compared to
anything.

<a id="5"></a>

## 5. Profile — quality

48–64 GB. Q6, Q8, BF16 where practical. Its purpose is to tell two failures
apart: *the storage layer is wrong* and *the compression broke the model*. The
same storage implementation runs unchanged across all three.

<a id="6"></a>

## 6. The rule the whole component is built on

> **The model decides meaning. Code decides mechanics.**

The model decides what matters, what a passage means, whether an observation is
a decision, whether two notes say the same thing, which directory is likely to
hold the answer, and what to open next.

The model does not decide ids, hashes, offsets, ACLs, file existence, token
counts, transaction semantics, allowed paths, context size, revisions, or
whether a write succeeded.

**And the enforcement is structural, not textual.** A prompt saying *do not
invent ids* is a request. `KnowledgeProposal` having no id field is a wall — see
[`knowledge/schema.ts`](../ai-storage/src/knowledge/schema.ts), where `noteFrom`
is the only way to produce a `KnowledgeNote` and its signature puts the model's
output on one side and the minted mechanics on the other.

<a id="7"></a>

## 7. The model is never the database

Not a system prompt holding the project history. Not a large JSON blob. Not a
100K-token `MEMORY.md`. Instead: `memory_index`, `memory_open`, `memory_find`,
`memory_source`.

Memory is external. Context is temporary.

<a id="8"></a>

## 8. Hierarchy and retrieval order

```text
.ai/storage/{system,users,projects,flows}/…
```

Four levels — `SYSTEM`, `USER`, `PROJECT`, `FLOW` — and retrieval runs
`FLOW → PROJECT → USER → SYSTEM`. Local knowledge beats distant knowledge,
because a constraint this flow recorded about this run was written knowing about
this run and the system-wide generality was not.

<a id="9"></a>

## 9. A note, and what a note is not

A note is one coherent piece of reusable knowledge — retrievable, citable,
verifiable, supersedable, promotable on its own.

A note is **not a chunk**. Chunks exist because the model has finite context;
they are an artefact of the reader, not a unit of knowledge, and they must never
become records automatically. The Archivist reads chunks and proposes concepts.

<a id="11"></a>

## 11. The note schema

See [`knowledge/schema.ts`](../ai-storage/src/knowledge/schema.ts). Eight types
(`fact`, `decision`, `constraint`, `procedure`, `failure`, `experiment`,
`preference`, `observation`), a claim, keywords, evidence, relations, a state,
timestamps, a revision.

<a id="12"></a>

## 12. What the model may propose

```typescript
interface KnowledgeProposal {
  title: string;
  type: NoteType;
  claim: string;
  keywords: string[];
  source: { artifact: string; from: number; to: number };
}
```

No id. No hash. No scope. No state. No timestamp. No revision. Not "the model
should not fill these in" — there is nowhere to put them.

<a id="13"></a>

## 13. Structured output solves syntax, not truth

Every specialist returns schema-constrained output, and every reply is then
validated again on this side. Constrained decoding will happily produce:

- a byte range that runs backwards, or has zero width;
- an artifact path that climbs out of the store;
- a citation into a file that does not exist;
- a claim about bytes 1100–2450 of a file that is 400 bytes long.

All four are schema-valid. All four are refused, and
[`test/knowledge.test.ts`](../ai-storage/test/knowledge.test.ts) and
[`test/evidence.test.ts`](../ai-storage/test/evidence.test.ts) start from
proposals a schema would have accepted, because that is the only interesting
case.

<a id="14"></a>

## 14. Provenance

```text
proposal → source exists? → range inside it? → extract → sha256 → ACL → persist
```

**The hash is of the slice, not of the file.** A digest of the whole file says
*this file has not changed*, which is the wrong claim: regenerate an artifact
and every note that cited any part of it becomes unverifiable at once, including
the ones whose bytes are identical. A digest of `[from, to)` says *the bytes this
claim was read from still say what they said*.

Re-checking gives three answers — `intact`, `changed`, `gone` — and `changed` is
never repaired. Re-hashing would turn *this claim is now unverifiable* into
*this claim is verified*, which is the exact inversion this component exists to
prevent.

<a id="15"></a>

## 15. The index answers one question

> Where should I look?

Not *what is this project about*. Prose costs six hundred tokens and narrows
nothing; a directory listing costs forty and eliminates nine tenths of the
store. The reader has two thousand navigation tokens for the whole descent.

<a id="16"></a>

## 16. Token-bounded, and enforced

`root_max_tokens: 1200`, `node_max_tokens: 1400`, `maximum_depth: 8`. A node
over budget **splits**, and `assertWithinBudget` refuses to render one that has
not — a budget nothing enforces is a comment.

Splitting is mechanics, so it is deterministic: grouped by prefix, falling back
to buckets. Asking a model where an overfull directory divides would make the
same store split differently on two runs, and an unrepeatable index makes an
unrepeatable benchmark. *Where a new note belongs* is a semantic question and it
goes to the Indexer; *where a full drawer divides* is not.

Building it found a real bug: prefix grouping can produce as many groups as
there were notes — four hundred names, four hundred prefixes — leaving the
parent listing four hundred directories, which is the node that was over budget
with a level of indirection in front of it. A split is now only accepted once
the parent it produces has been measured.

<a id="17"></a>

## 17. The context invariant, and the number it protects

```text
  System / harness           1,500
  Current task                 600
  Navigation                 2,000
  Retrieved knowledge        2,300
  Reasoning + output         1,792
                           -------
                             8,192
```

Lanes rather than one pool, because the failure they prevent is the one that
looks like success: a Librarian that spends six thousand tokens walking the
index and then has no room to read the note it found has navigated beautifully
and answered nothing.

**A read that does not fit is refused. It is never truncated.**

```json
{ "error": "MEMORY_CONTEXT_LIMIT", "requestedTokens": 2841, "availableTokens": 1719 }
```

A refusal is an event the harness can see and act on — narrow the query, open
fewer notes, split the node. A truncation is invisible, and an invisible
truncation is a model answering from half a note while the run record says it
read the whole one. Every headline number this component produces is a ratio
whose denominator is *tokens the model actually had to see*; one silent trim
anywhere and that denominator is fiction.

The five numbers are configuration to be moved by results, not physics.

<a id="18"></a>

## 18. What is to be measured

**Navigation Efficiency Ratio** = corpus tokens ÷ tokens the model actually
loaded, at stable accuracy. **Storage Amplification** = searchable corpus ÷
effective context.

Against baselines, because a candidate with no baseline is a demo:

- **A** — the model with a flat `MEMORY.md`
- **B** — the model with grep/FTS
- **C** — the model with ai-storage navigation
- **D** — later, C plus embeddings

C ships only if it beats A and B where they fail. D ships only if it beats C.
**No vector database in v1** — hierarchical navigation and exact lexical search
first, so that the benchmark can say whether semantic retrieval was needed
rather than assuming it.

And the quantization question, which is this component's own version of
doc/11's:

> Can better infrastructure compensate for aggressive model quantization?

Q8/Q6 → Q4 → IQ2, measuring navigation accuracy, tool-call correctness, schema
compliance, loop rate. The 8K runs are the ones that matter.

**None of this has been run.** `ai-storage/bench/` is empty on purpose: a
benchmark directory with unrun scripts in it reads like a result.

<a id="21"></a>

## 21. Local only, and why it is checked at construction

`AI_STORAGE_LOCAL_ONLY` defaults to true and means no OpenRouter, no Anthropic,
no OpenAI, no Google, no telemetry containing prompts, no external embedding
API, no cloud reranker.

A non-loopback base URL is refused when the model object is **built**, not when
a request is made — by then a prompt exists and something has decided to send
it, and a privacy guarantee that fails on the first request has already failed.

Loopback only, not private ranges. `10.0.0.7` is somebody else's machine even
when it is on your desk; the guarantee is that the prompt did not leave *this*
computer.

<a id="22"></a>

## 22. Specialists — one model, several capabilities

Same weights; different prompts, tools, schemas and budgets.

| Role | Decides | Tools | Thinking |
|---|---|---|---|
| Librarian | what to read next | index, open, find, done | off |
| Archivist | what a source means | source open/slice, propose | on |
| Indexer | where a note belongs | placement only | off |
| Reconciler | new / same / supersedes / conflict | read only | on |
| Auditor | what is missing | read only | on |
| MemoryKeeper | nothing; it coordinates | none | off |

**Security is structural.** The Librarian has no write tool — not a prompt
telling it not to write. A hallucinated `write_file` fails because the operation
does not exist on its side of the boundary. The MemoryKeeper cannot touch
storage directly, so it cannot route around its own specialists.

Reasoning is spent where judgement is needed and not where deterministic
navigation is enough, and that assumption is a hypothesis for
`bench/agent-tools` rather than a permanent setting.

Temperature 0.0 for Librarian and Indexer, 0.1 for the rest. Storage favours
consistency over creativity.

<a id="41"></a>

## 41. Bounded loops, bounded retries

Local models get stuck. Step caps per role (Librarian 12, Archivist 20,
Reconciler 8, Indexer 8), and the same tool with the same arguments returning
the same result twice is `REPEATED_TOOL_LOOP`.

Schema failures get **two** semantic retries and then `MEMORY_AGENT_FAILURE`.
Not sampling until something validates: how often a quantization fails to
produce valid output is one of the results, and a retry loop erases it before
anything can record it. The transports do not retry at all.

<a id="54"></a>

## 54. Phases, and where the line is now

| Phase | What | State |
|---|---|---|
| 1 | model boundary, engines, structured output, tool calls, token measurement | **built** |
| 2 | notes, provenance, derived progress, token budgets, index tree | **built** |
| 2b | filesystem backend, transactions, lexical search | not built |
| 3 | Librarian, and the first real benchmark | not built |
| 4 | Archivist | not built |
| 5 | Reconciler, Indexer | not built |
| 6 | MemoryKeeper | not built |
| 7 | scopes and ACL enforcement | not built |
| 8 | promotion, revision, restore | not built |
| 9 | the constrained-Mac run, reported separately | not built |

Phase 3 is where the central hypothesis first meets evidence: 8K of context,
ten thousand notes, an answer planted in one of them. Everything before it is
plumbing that cannot be wrong in an interesting way, and everything after it is
worth building only if Phase 3 comes back positive.

<a id="56"></a>

## 56. What would count as a result

Not *the model supports a million tokens*. This:

> A 27B model with 8K of effective working context reliably operates on project
> knowledge orders of magnitude larger than its context, because ai-storage
> gives it a navigable external memory.

And the honest form of the same sentence, which is the one this repository is
obliged to publish either way:

> …or it does not, and the flat-file baseline was just as good, and we say so.

The predecessor project measured a closely related idea — indexing memory on a
second axis — and it came back flat at 80% acc@1 either way, with the second
axis carrying real information that did not change the answer. That is the
outcome to expect and to be ready to publish. The burden of proof is on the
hierarchy.
