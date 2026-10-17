# D8: can a local 12B walk the memory when code drives the walk?

**Written 2026-10-17, before anything runs.**

## Why

Today everything in the director that reads the memory is Claude: Sonnet plans by walking it, and Opus edits it. The
target division of work is different. Claude builds and maintains the memory (pages, links, strategy, trajectories) so
that a **local Gemma 4 12B without fine-tuning** can walk it. The 12B then plans and routes from it, at no cloud cost.

So the question is not whether a 12B can plan freely. The question is whether **this memory, walked by code that only
asks the 12B closed questions, leads the 12B to the statements Sonnet relied on**. If it does not, the misses become
Claude's restructuring work. Retraining the model is not part of that work.

## The walk (code drives, the 12B chooses)

Every answer from the 12B is a JSON object constrained by Ollama's `format` (JSON schema). Each answer picks from
numbered options.

1. **Situations.** The code lists every line of `Index.md` (the situation part only). The 12B picks at most 4 that
   match the case.
2. **Entry statements.** The code collects the entry statements of the chosen situations and shows each as
   `[[Page#id]]` with its **When** and **Do**. The 12B picks the ones that apply.
3. **Links, two hops.** The code collects the `Links` of the picked statements that have not been shown yet, and the
   12B picks again. This step runs twice.

The walk's result is the set of statements the 12B picked.

## Reference, metrics and baseline

- **Memory:** the exact memory D0's treatment arm walked (`~/.config/director/d0/arm-T/sandbox/memory`, 287
  statements). The reference walks were made on it.
- **Reference:** the statements Sonnet cited in its two D0 plans per case (`arm-T/plans.jsonl`, judged mean 0.929).
  - **core** = cited in both samples;
  - **broad** = cited in either.
- **Primary metric:** recall of the core, averaged over the 14 cases.
- **Also reported:**
  - recall of the broad set;
  - precision against the broad set (picked ∩ broad / picked);
  - picks per case;
  - JSON failures;
  - wall time.
- **Baseline without a model:** BM25 over each statement's When + Do + Because. Each case's situation text is the
  query, and the baseline takes the same number of statements the 12B picked for that case. It is compared with the 12B
  per case on core recall (wins, losses, ties, exact sign test). A 12B that cannot beat keyword search adds nothing.

## Verdicts (fixed here)

| verdict | condition |
|---|---|
| **NAVIGABLE** | mean core recall ≥ 0.70 **and** the 12B beats BM25 in more cases than it loses |
| **PARTIAL** | mean core recall between 0.40 and 0.70: the missed core statements are the first restructuring list for Claude |
| **NOT NAVIGABLE** | mean core recall < 0.40, or the 12B does not beat BM25 |

## Parameters

| parameter | value |
|---|---|
| model | `gemma4:12b-mlx` on Ollama 0.40.3 (MacBook Air M4, 16 GB) |
| temperature | 0 |
| `think` | off |
| `num_ctx` | 16384 |
| runs | one deterministic run |
| cost | US$0 (local); ~1 h estimated |

Abort rule: if more than 5 answers fail JSON validation after one retry each, the run stops and reports it.
Plan quality (the D0 judge, Sonnet, paid) is **not** part of D8. It is the next step, briefed separately, and only if
D8 is not NOT NAVIGABLE.

## Outcome (2026-10-17)

**NOT NAVIGABLE** by the fixed rule. Mean core recall is **0.321**, below 0.40. The other numbers explain the verdict:

| measure | 12B | BM25 (no model) |
|---|---|---|
| mean core recall | 0.321 | 0.162 |
| mean broad recall | 0.225 | — |
| precision vs broad | **0.894** | — |
| per case vs BM25 | **12 wins, 0 losses**, 2 ties (exact sign test p = 0.0005) | — |
| picks per case | 5.6 | same |
| JSON failures | **0 of 40 calls** | — |

Wall time was ~20–40 s per case. Cost was US$0 (`gemma4:12b-mlx`, Ollama 0.40.3, M4 16 GB). Data:
`~/.config/director/d8/{walks.jsonl,report.json}`.

**Where the misses are.** This diagnosis is computed from the memory after the run; it is not a redesign. All
176 missed core statements are **entry statements of some Index situation**: the walk could have reached each one. But
each case's core is spread over **6–25 Index situations** (c08 over 25, c11 over 1), and the walk lets the 12B choose 4.
The 12B picks well (0.894 precision, beats keyword search 12/0). It picks too little because the Index is organised by
small, separate situations, while a real case touches many at once.

**What this means for the design.** The bottleneck is how the memory is organised for a closed-choice walker. It is
not the model. That is the restructuring work assigned to Claude: entries that bundle what a *type* of task needs
(trajectories), so that one choice reaches a case's core. Next step (to be briefed fresh, not run on these data):
Claude writes trajectory entries from material other than D0's cases. D8's walk is then re-run unchanged, with D0's 14
cases as the held-out set.

Caveat: the reference is citation recall against Sonnet's plans, and Sonnet cites broadly (core of 2–33 statements per
case). Whether 5–6 well-chosen statements are enough for a good *plan* is the D0 judge's question. It was out of scope
here by the brief.
