# Evolving Agents

> **We got the decomposition right and the substrate wrong.**
>
> This repository was the Evolving Agents Toolkit (EAT). Its five subsystems were
> each rebuilt, separately and without noticing, on a substrate that makes them
> verifiable. This is the map of where they went and what it cost.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## What EAT was

18,680 lines of Python across twelve subsystems — a SmartLibrary of versioned
components, a SmartAgentBus for discovery and routing, Smart Memory, an evolution
loop, and a governance layer called Firmware. Backed by MongoDB Atlas.

It had **three test functions.**

That number is the whole story. EAT was not a product that failed; it was an
architecture that was written down and never pinned to anything that could
contradict it. The decomposition was right — right enough that every piece got
independently re-derived over the following year.

## The proof, in EAT's own source

`evolving_agents/firmware/firmware.py`:

```python
self.base_firmware = """
You are an AI agent operating under strict governance rules:
...
- Never use dangerous imports (os, subprocess, etc.)
"""
```

That is governance by **asking**. It is a string.

In [token-trie](https://github.com/EvolvingAgentsLabs/token-trie), every legal
instruction is pre-tokenized into a trie of token IDs and the sampler's
valid-next set is masked at each decoding step. The forbidden token is not
discouraged — it has no path. A 350M-parameter model plays Tetris in a browser
tab and *cannot* emit malformed output.

Same intention. Twelve months and one substrate apart. The difference is the
entire thesis, and you can diff it yourself.

## Where the five subsystems went

| EAT subsystem | Rebuilt as | What changed |
|---|---|---|
| **Firmware** — a prompt asking for good behaviour | [token-trie](https://github.com/EvolvingAgentsLabs/token-trie) | Constraint moved from the prompt to the decoder |
| **Evolution loop** — 337 lines | [agentvcs](https://github.com/EvolvingAgentsLabs/agentvcs) | `freeze` refuses unless the eval passes; `--force` stamps `verified: false` rather than lying |
| **Smart Memory** | [evolving-memory](https://github.com/EvolvingAgentsLabs/evolving-memory) | Behavioural tests: failures extract constraints, repetition raises confidence, domains stay isolated |
| **Agents & workflow** | [skillos](https://github.com/EvolvingAgentsLabs/skillos) | Markdown as the executable, with an AST-verified benchmark instead of an LLM judge |
| *(nothing)* | [skillos_robot](https://github.com/EvolvingAgentsLabs/skillos_robot) | EAT had no embodiment. This one has firmware, CAD and an ESP32 |
| *(nothing)* | [sleep-harness](https://github.com/EvolvingAgentsLabs/sleep-harness) | EAT had no security story beyond the Firmware prompt |

## What that bought

| | EAT | Now |
|---|---:|---:|
| Test functions | **3** | 190 · 795 · 182 · 39 across the four core repos |
| Governance | a prompt string | a token that cannot be emitted |
| Infrastructure | MongoDB Atlas | zero runtime dependencies; runs in a browser tab |
| Evidence | none published | pre-registered hypotheses, including the refuted ones |

## What it cost — two things were lost

**The SmartAgentBus is gone, deliberately.** 1,050 lines of agent registry and
runtime routing. Nothing replaces it: agentvcs versions a sub-agent swarm as a
mergeable graph, which is topology, not routing. It was dropped because it was
welded to MongoDB and because MCP is eating that problem. If an agent needs to
discover another at runtime today, there is no answer here.

**The dual-embedding resolver is gone, and that one was a mistake.** EAT
indexed every component twice — `content_embedding` for *what a thing is*, and
`applicability_embedding` for *what it is for* — and resolved tasks against the
second. Today the ecosystem resolves skills by matching against a description
field, which is the naive form of the same idea.

So a problem this repository solved in 2025 is still open in the tools that
replaced it. That piece is coming back.

## Where the work is now

- **[The thesis](https://evolvingagentslabs.github.io/thesis/)** — the through-line across the current experiments
- **[Evolving Agents Labs](https://evolvingagentslabs.github.io)** — all of it, labelled by how much evidence stands behind each part

## The code

Still here, unchanged, on the default branch. It runs if you give it a MongoDB
Atlas cluster. The original README is preserved at
[`docs/README-EAT-2025.md`](docs/README-EAT-2025.md), including its two sunset
notices — one of which pointed at a repository that no longer exists.

Left in place rather than deleted, because the diff between `firmware.py` and a
token trie is the most direct argument this organisation has, and it only works
if both halves stay readable.

---

<sub>By [Matias Molinas](https://github.com/matiasmolinas) and
[Ismael Faro](https://github.com/ismaelfaro) · Apache 2.0</sub>
