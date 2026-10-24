# Evolving Agents — the director

<p align="center">
  <img src="docs/img/hub-hero.jpg" alt="A workshop room. In front, a chart table with a map of linked index cards and a route traced in red; two brass tubes leave it, one to a window and a distant domed building, one to a small machine nearby. Behind, three benches: a card-catalogue library, a machine with a sealed ledger, an evidence table with threaded cards." width="100%">
</p>

> **The director remembers how your team works — and makes every agent session work that way.**
>
> Your decisions, conventions and hard-won methods become a small memory of linked rules. Before an agent plans, the
> director finds the rules that apply and puts them in front of it. Private work goes only to machines you have
> cleared. When you correct it, it drafts a lesson that you accept or reject. It runs inside **Claude Code** today.

[![GitHub stars](https://img.shields.io/github/stars/EvolvingAgentsLabs/evolving-agents?style=social)](https://github.com/EvolvingAgentsLabs/evolving-agents/stargazers)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Claude Code plugin](https://img.shields.io/badge/Claude%20Code-plugin-8A9A6B)](#get-started)

**Contents** — [What it does](#what-it-does) · [Who it is for](#who-it-is-for) · [What to expect](#what-to-expect) ·
[Get started](#get-started) · [What comes next](#what-comes-next) · [How it works, and what we measured](docs/DIRECTOR-EVIDENCE.md) ·
[Earlier work](#earlier-work-in-this-repository)

## What it does

<p align="center">
  <img src="docs/img/what-it-does.jpg" alt="Left, a card-catalogue drawer of linked index cards; one card is lifted and carried by a red-pencil line to a writing desk, where it is pinned to one line of an open plan, a blank card set apart beside it. From the desk two brass tubes leave: one through an arched window toward a distant domed building, the other to a padlocked wooden cabinet in the same room, a sealed envelope travelling only along it. In front, a ledger whose lines end in wax seals, one freshly pressed." width="100%">
</p>

| | |
|---|---|
| **Carries your team's decisions into the work** | The agent does not guess which definition, threshold or convention your team chose. The director hands it the one you decided, at the step where it matters, and the plan says which rule it followed. |
| **Learns from your corrections — only with your yes** | Correct it once and it drafts the lesson, checks it against what it already knows, and keeps it only if you accept. Every lesson is a change you can read, compare and undo. |
| **Keeps private work private** | Mark a task confidential and it can only go to a machine you have cleared — a local model, a server in your office — never to a cloud model. That rule is enforced in code, not asked of the model. |
| **Sends each task where it belongs** | A frontier model, a model on your own machine, or another tool. Move a task once and it plans the next one like it the same way. |
| **Shows what it decided, and what it could not** | Every plan lists the rules it followed and, separately, the choices your memory does not cover yet — the decisions that are still yours. |

## Who it is for

- **Teams that work with coding agents** and keep a `CLAUDE.md`, an `AGENTS.md` or Claude Code's memory that the agent
  still forgets under pressure.
- **Any work that runs on house conventions** — how a lab defines a measurement, a firm's review checklist, a team's
  release rules — where two good answers that follow different conventions are still a disagreement.
- **People with confidential material** who want a strong model to plan the work while the files themselves are
  processed only on machines they control.

## What to expect

The director does not make a strong model smarter. With today's best models, an agent already follows general good
practice on its own — we measured that, and say so. Where the director pays is **what the model cannot know**: your
team's own decisions and conventions, your privacy rules, the methods you learned the hard way, and a record of every
lesson that you can audit. Every claim on this page has a measurement behind it, including the ones that came back
flat: [how it works, and what we measured](docs/DIRECTOR-EVIDENCE.md).

## Get started

<p align="center">
  <img src="packages/director/docs/img/compile.jpg" alt="Long paper scrolls and notebooks feed a brass binding press; out of it come index cards, filed in a small card-catalogue drawer, a few of them linked by ink lines, a brass gauge on the drawer's edge." width="70%">
</p>

Inside Claude Code:

```
/plugin install director --marketplace EvolvingAgentsLabs/evolving-agents
/director:compile        # your CLAUDE.md, AGENTS.md and memory → linked rules (asks first; ~US$5 for ~60 files)
/director:reflect        # after a task: your correction becomes a proposed lesson, kept only on your yes
```

From the next session on, every plan walks your memory and cites what it follows. A band above the prompt shows each
proposed lesson with **Diff**, **Accept** and **Reject**, and each delegated task until its result is back. The
memory stays on your machine as plain Markdown. Requirements, the two-lane local interface, running without a person
present, and costs: [`packages/director/`](packages/director/).

## What comes next

Three questions, in order, each judged by a measurement named before it runs ([ROADMAP.md](ROADMAP.md)): does an
agent get better **from one session to the next** on the work you repeat, without the memory ever making it worse
([director-bench](https://github.com/EvolvingAgentsLabs/director-bench)); can the director **improve its own
components**, not only its memory — the promise of the name *Evolving Agents*; and can the same memory run on
**other agent platforms** (Anthropic's Managed Agents, Google's Antigravity agent).

**Built with**, from the same lab: [**lora-kernel**](https://github.com/EvolvingAgentsLabs/lora-kernel) — small
specialist models that run on your own machine · [**agentvcs**](https://github.com/EvolvingAgentsLabs/agentvcs) —
every lesson the director keeps is a versioned change you can compare and undo ·
[**gene-evidence**](https://github.com/EvolvingAgentsLabs/gene-evidence) — reports in which every sentence cites its
evidence. All projects: [evolvingagentslabs.github.io](https://evolvingagentslabs.github.io).

---

## Earlier work in this repository

| | |
|---|---|
| [`packages/director/`](packages/director/) | **Active — the project.** The director and its Claude Code plugin |
| [`legacy/eat/`](legacy/eat/) | The Evolving Agents Toolkit, 2025, the director's ancestor. Kept readable; its 2025 README is [docs/README-EAT-2025.md](docs/README-EAT-2025.md) |
| [`plugin/`](plugin/) | **Paused.** A Claude Agent SDK plugin that versions an agent's evolution (commit, diff, merge, freeze on a passing eval). Its ideas continued in agentvcs; merging two forked *sessions* ([M1](PLAN.md)) never started |
| [`ai-os/`](ai-os/) | **Paused** since 2026-09-06. An agent-based operating system on the [qm](https://github.com/yc-software/qm) base, kept as a subtree |
| [`packages/memory/`](packages/memory/) | Structured recall above Claude Code's flat memory files. Works; measures no better than naive matching ([PLAN.md](PLAN.md)) |
| [`packages/agentvcs/`](packages/agentvcs/) | **Frozen copy.** agentvcs lives in [EvolvingAgentsLabs/agentvcs](https://github.com/EvolvingAgentsLabs/agentvcs) |
| [`demos/robot/`](demos/robot/) | A 2D robot that evolves its own skills, versioned with agentvcs |

<details>
<summary><b>What the 2025 toolkit taught us — three flat results</b></summary>

EAT was not a product that failed; it was an architecture written down and never pinned to anything that could
contradict it. 10,165 of its lines were deleted where the Agent SDK does the same better —
[docs/WHAT-WAS-DELETED.md](docs/WHAT-WAS-DELETED.md) lists each one and what replaced it. Rebuilt and measured since:

- **The dual-embedding resolver does not help.** EAT indexed every component twice — what it is, and what it is
  *for*. 80% acc@1 either way; the second axis is distinct (`cosine = 0.753`) and buys nothing on a modern encoder.
- **Retrieving past experience buys nothing on the task.** The premise of `SmartMemory` and `ContextBuilderTool`,
  measured three times: **−4**, **+0** (p = 1.0000) and **−6** against retrievers up to an oracle. What did move was
  a **compact statement of a rule** induced from the same trajectories: +21 over the oracle — which is what the
  director's memory is made of.
- **The memory hierarchy loses to lexical search.** Four levels, as EAT's smart memory argued: exact search 3/3 at
  every corpus size, hierarchical navigation 1–2/3.

It was right about where to look and wrong about what would be there. That is why it is in `legacy/`, not deleted.

</details>

<details>
<summary><b>Breaking change: <code>pip install</code> no longer installs <code>evolving_agents</code></b></summary>

This is a monorepo now; install what you want by name. The 2025 package sits at `legacy/eat/` with its original
`setup.py`, unchanged. agentvcs was never published to PyPI; the frozen copy installs from source:

```bash
pip install "git+https://github.com/EvolvingAgentsLabs/evolving-agents#subdirectory=packages/agentvcs"
```

The maintained agentvcs is [EvolvingAgentsLabs/agentvcs](https://github.com/EvolvingAgentsLabs/agentvcs).

</details>

---

<sub>By [Matias Molinas](https://github.com/matiasmolinas) and
[Ismael Faro](https://github.com/ismaelfaro) · Apache 2.0</sub>
