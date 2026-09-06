> **ACTIVE again — 2026-09-06.** This repository was frozen on 2026-08-01 and
> closed with a pointer to `ai-os`. `ai-os` has now come home: it is a subtree
> here, with its history intact, and this is where the organisation's active
> work lives.
>
> Nothing was rewritten to make the arrival look tidy. The 2025 toolkit stays in
> [`legacy/eat/`](legacy/eat/), the flat results stay in **Evidence** below, and
> two more have been added — because they point the same way as the first, which
> is the finding.

# Evolving Agents

<p align="center">
  <img src="docs/img/evolving-agents.jpg" alt="One empty outline on the left, drawn but never filled, becoming five solid shapes on the right — each traced back to where it came from" width="100%">
</p>

> **The loop is solved. What happens after the fork is not.**
>
> A plugin for the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview)
> that versions an agent's evolution — diff it, merge it, refuse to ship it when
> the eval fails.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## The gap

The Agent SDK runs the loop, ships the tools, manages context, and enforces
permissions. It also branches: `fork` gives you a second session starting from a
copy of the first.

Nothing brings the two back together. There is no merge, no diff between two
sessions, nothing that refuses to promote an agent whose eval regressed, and the
transcript is plain JSONL that anyone can edit. Sessions persist the
*conversation* — the skills, subagents, model and goal that produced it are not
versioned alongside it.

That gap is this repository. Most of it is closed; the last row is not, and it is
the one worth being precise about.

| | Agent SDK | This plugin |
|---|---|---|
| Branch a session | `fork` | — |
| Snapshot code, goal, models and trace as one unit | — | `avcs_commit` |
| Compare two snapshots | — | `avcs_diff`, per dimension |
| Rejoin two snapshots | — | `avcs_merge`, with a `--reconcile` seam for goal and trace |
| Refuse to ship a regression | — | `avcs_freeze`, which fails unless the eval passes |
| Prove a transcript wasn't edited | — | Ed25519-signed commits |
| Keep the trace past compaction | summarised away | archived by the `PreCompact` hook |
| **Diff or merge two forked _sessions_** | — | **not yet — [M1](PLAN.md)** |

The unit `avcs_diff` and `avcs_merge` operate on is an **agentvcs commit**, not an
SDK session. Both work today, and neither will take two session IDs: a session is
an append-only event log, and the adapter that turns one into something mergeable
does not exist yet. Two forked sessions do share a prefix, so the common ancestor
is findable — the open question is what a *conflict* means when both branches are
conversations that reached different conclusions about the same file.

That adapter is [M1](PLAN.md), it has not started, and it is the reason this
repository exists rather than a detail of it.

## Install

```python
from claude_agent_sdk import query, ClaudeAgentOptions

options = ClaudeAgentOptions(
    plugins=[{"type": "local", "path": "/path/to/evolving-agents/plugin"}],
)
```

Full details in [`plugin/README.md`](plugin/README.md). The MCP server has **zero
runtime dependencies** — standard library only, with a test that fails if that
ever stops being true.

## What is here

| | |
|---|---|
| [`ai-os/`](ai-os/) | **The active project.** An agent-based operating system on a vendored [QM](https://github.com/yc-software/qm) base: flows, a desk you arrange, agents as markdown, memory at four levels. 851 tests, CI, a running stack |
| [`plugin/`](plugin/) | The Agent SDK plugin: MCP server + three hooks |
| [`packages/agentvcs/`](packages/agentvcs/) | The version control itself — 220 tests, no dependencies. **Not on PyPI**; install from source |
| [`packages/memory/`](packages/memory/) | Structured recall above the SDK's flat `.claude/` memory files. Works; measures no better than naive matching — see [PLAN.md](PLAN.md) |
| [`demos/robot/`](demos/robot/) | A 2D robot that evolves its own skills, versioned with agentvcs |
| [`legacy/eat/`](legacy/eat/) | The Evolving Agents Toolkit, 2025. Kept readable; see below |

## Why this repository has a 2025 in it

It was the Evolving Agents Toolkit: 18,680 lines across twelve subsystems — a
component library, an agent bus, smart memory, an evolution loop, and a
governance layer called Firmware. Backed by MongoDB Atlas.

It had **three test functions.**

That number is the story. EAT was not a product that failed; it was an
architecture written down and never pinned to anything that could contradict it.

Most of it is now deleted, because the SDK does it better —
[`docs/WHAT-WAS-DELETED.md`](docs/WHAT-WAS-DELETED.md) lists the 10,165 lines and
what replaced each one. `Firmware` asked a model to *"never use dangerous
imports"* in a string; a `PreToolUse` hook returning `permissionDecision: "deny"`
stops the call whatever the model decided. We do not ship our own version of a
problem that is already solved.

What survives in `legacy/eat/` is the ancestry of what ships today. Read
`evolution/` if you read one thing: 337 lines that closed the loop between an
agent changing and that change being kept, with nothing to verify the change was
an improvement. `avcs_freeze` is the same idea with the missing half added.

## Evidence

Claims here are measured, including the ones that came back flat.

- **The dual-embedding resolver does not help.** EAT indexed every component
  twice — once for what it is, once for what it is *for*. Rebuilt and measured:
  80% acc@1 either way, no difference. The second axis is genuinely distinct
  (`cosine = 0.753`); it just buys nothing on a modern encoder.
- **This organisation's "byte-identical wire format" claim was wrong.** Two of
  three opcode regexes matched; `HALT` had diverged in a way that changes what
  parses. Found by writing the test instead of repeating the sentence.
- **Retrieving past experience buys nothing on the task.** EAT's `SmartMemory`
  and its `ContextBuilderTool` rest on the premise that recalling similar past
  work improves the next attempt. Measured three times in `ai-os` and the runtime
  beside it: **−4** against a retriever handed the solved similar cases *with
  their answers*, **+0** (p = 1.0000) against the strongest retriever
  constructible, and **−6** against an *oracle* retriever — below doing nothing.
  What did move was a **compact statement of a rule** induced from the same
  trajectories: +21 over the oracle.
- **The memory hierarchy loses to lexical search.** `ai-storage` implements the
  four levels EAT's smart memory argued for. Its first benchmark, at the ceiling
  with a perfect navigator: exact search 3/3 at every corpus size, hierarchical
  navigation 1–2/3 and out of steps, the flat file refusing to fit at all.

> **Three flat results, three architectures, one direction.** The dual axis in
> 2025, the experience retrieval in 2026, the hierarchy after it. Each was the
> obvious next structure and each was asked for a number. That is what this
> repository is for, and it is why the toolkit is in `legacy/` rather than
> deleted: it was right about where to look and wrong about what would be there.

## Breaking change

`pip install git+https://github.com/EvolvingAgentsLabs/evolving-agents` no longer
installs an `evolving_agents` package — this is a monorepo now. Install what you
want by name:

`agentvcs` was never published to PyPI — the release workflow exists but its
Trusted Publisher was never registered, so `pip install agentvcs` returns 404.
Install from source:

```bash
pip install "git+https://github.com/EvolvingAgentsLabs/evolving-agents#subdirectory=packages/agentvcs"
```

The 2025 package sits at `legacy/eat/` with its original `setup.py`, unchanged.

---

<sub>By [Matias Molinas](https://github.com/matiasmolinas) and
[Ismael Faro](https://github.com/ismaelfaro) · Apache 2.0</sub>
