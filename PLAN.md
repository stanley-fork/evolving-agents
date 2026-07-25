# Plan

This repository is the umbrella, not a museum. It holds the thesis and the map;
the work happens in the projects it points at. But it has its own three
milestones, and two are already resolved — one of them in a direction nobody
wanted.

## M0 · Transfer and reframe — **done** (2026-07-25)

Moved into the organisation with 452 stars, 35 forks and inbound links intact,
and the README rewritten around what EAT got right and wrong.

Worth recording why this was urgent rather than cosmetic: the repository carried
two stacked sunset notices, one pointing at `EvolvingAgentsLabs/llmunix`
(archived) and one at `EvolvingAgentsLabs/llm-os` (**404 — never existed under
that name**). The most-starred repository in the organisation was sending people
to a dead link.

The code stays on the default branch. `firmware/firmware.py` sets a governance
string asking a model to *"never use dangerous imports"*; token-trie makes the
token unreachable. That diff is the most direct argument this organisation has,
and it only works if both halves stay readable.

## M2 · Recover the dual-embedding resolver — **done, and it came back flat**

EAT indexed every component twice: once for what it *is*, once for what it is
*for*. The idea was right and the implementation needed MongoDB Atlas Vector
Search, so it did not survive the move to a local stack.

Rebuilt in
[evolving-memory](https://github.com/EvolvingAgentsLabs/evolving-memory/tree/main/src/evolving_memory/resolver)
with no infrastructure, plus one improvement over the original: both indexes are
searched and the union scored, rather than picking one and re-ranking its
survivors — re-ranking cannot rank a component the first index never returned.

Then measured, which is the part that matters:

| | acc@1 | MRR |
|---|---:|---:|
| description-matching (baseline) | **80%** | 0.900 |
| applicability only | **80%** | 0.900 |

**No difference.** Not a plumbing bug — `cosine(content, applicability) = 0.753`,
so the second embedding is a genuinely distinct axis. Both residual errors are
within-domain confusions, which applicability text cannot address.

The design dates from 2025, when the gap between task language and
implementation language was wide. A modern encoder appears to close much of it
unaided.

**The run that would settle it:** a smaller or older encoder. The hypothesis
predicts the advantage returns there — and that is where this organisation
lives, since everything else here targets small local models. Full write-up and
limits in
[`benchmarks/RESULTS.md`](https://github.com/EvolvingAgentsLabs/evolving-memory/blob/main/benchmarks/RESULTS.md).

## M1 · A portable definition with a conformance suite — **not started**

The remaining milestone, and the one that would make this repository load-bearing
again rather than historical.

The five projects **already share an instruction format character-for-character**
— the opcode regex in `token-trie` and in `skillos_robot` is byte-identical,
because they were one codebase. What they do not share is *enforcement*:

| | Mechanism | Guarantee |
|---|---|---|
| token-trie | trie over logits | **structural** |
| skillos_robot | stop sequences + regex | none |
| skillos | prompt to a frontier model | none |

So a conformance table would read **one of three**. That is the number to publish
first. A table showing three of three would be less credible, not more, and
closing the gap is already planned as
[token-trie phase 5](https://github.com/EvolvingAgentsLabs/token-trie/blob/main/PLAN.md).

**Sequence, when it starts:**

1. `SPEC.md` derived from the wire format the two dispatchers already share —
   descriptive, from the code, not a manifesto written ahead of it.
2. Conformance cases as executable tests, one per *(definition, capability
   tier)* pair. The unit is the pair, not the repository: the same file can
   conform on a frontier model and fail on an embedded one, and that delta is
   the interesting output.
3. Publish the table with its reds showing.

**The metric worth defining.** Not *"does the same definition run everywhere"* —
it demonstrably does not run *equivalently*, because a 350M model ratifies a
hand-written planner while a frontier model plans. The publishable question is:

> **How much work has to move into the Program layer as the model gets smaller —
> measured?**

That is a curve, not a checkmark. Nobody publishes it because almost nobody has
all three tiers running. This organisation does.

## What is deliberately not planned

**Reviving the SmartAgentBus.** A thousand lines of agent registry and runtime
routing, welded to MongoDB, solving a problem MCP is now eating. Bringing it back
would restore the infrastructure dependency every other project here removed.
Recorded as dropped rather than left silent, so nobody re-derives it a third
time.
