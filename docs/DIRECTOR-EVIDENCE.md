# The director — how it works, and what we measured

> This page holds the detail that used to open the README: how the director works, every measurement with its brief,
> and what comes next. The README says what the director does; this page says how, and how we know.

## Your CLAUDE.md, actually followed

You wrote the rules down. Your agent still breaks them under pressure — **on 14 real situations, a Claude Code
session planning with Sonnet 5.5 and the rules loaded planned by them 0.589 of the time; the director, walking the
same rules compiled into linked statements, 0.929.** If you keep a `CLAUDE.md`, an `AGENTS.md` or Claude Code's
memory, this is for you:

> **Measured with Sonnet. With Opus 5.5 planning (2026-10-21) the gap is not there:** today's session alone follows
> the same rules 0.964 of the time ([D0O](../packages/director/docs/BRIEF-D0O.md)) and keeps a multi-day thread's
> decisions 0.80 against the director's 0.85 ([C2](../packages/director/docs/BRIEF-C2.md)). What the director still
> showed with Opus is cost — 34 % less per plan in C2. On situations from after the memory was frozen, Opus alone
> still left room only under its own judge ([F1O](../packages/director/docs/BRIEF-F1O.md)), and Gemini follows the same
> rules written as plain notes 0.875 ([PG](../packages/director/docs/BRIEF-PG.md)). Read every number below as Sonnet's;
> what the director is for with the strongest models is the next section.

- **it recalls the rule at the moment it applies** — and cites the statement it followed in every step of its plan;
- **it learns from your corrections** — each lesson proposed, checked against what it already knew, and kept only
  if you accept it, as a reversible, versioned patch;
- **it learns where you want work done** — move a task from the frontier to your local model once, and it plans the
  next one like it there;
- **rules that must hold are code, not requests** — a task marked confidential never leaves for a lane not cleared
  for it, the frontier included.

## An auditable memory that travels

What survives the move to Opus is not recall inside one plan — the model already does that — but what lasts and can
be checked:

- **Auditable.** Every lesson is a diff of When / Do / Because statements, with two probes of its own, a lint, an
  optional check that it changes the plan (TEACHES, REDUNDANT, NO EFFECT) and a decision: your yes, or — when no
  person is present — a declared policy, recorded as `decided_by` in the journal. Every accepted edit is an
  [agentvcs](https://github.com/EvolvingAgentsLabs/agentvcs) patch you can diff, blame, bisect and revert.
- **Yours, by domain.** What it learns is your methodology and operational knowledge: your decisions (a model, a
  cap, "from now on …"), and the methods a task reached only after failed attempts, each step with the check that
  caught the failure. Which lessons get in is decided by you, or by a verifier of your domain (tests, a build):
  `director.headless` accepts lessons drawn only from sessions whose verifier passed.
- **Portable.** The memory is Markdown. Another provider's model follows it as plain notes (Gemini, 0.875 — PG);
  the walk ships as a Claude Code plugin today; the Managed Agents and Antigravity lanes are planned, not built.
- **Cheaper.** With Opus planning, 34 % less per plan for the same score (C2); a local model plans from recipes the
  memory compiles, at no cost (D15).

## Install

<p align="center">
  <img src="../packages/director/docs/img/compile.jpg" alt="Long paper scrolls and notebooks feed a brass binding press; out of it come index cards, filed in a small card-catalogue drawer, a few of them linked by ink lines, a brass gauge on the drawer's edge." width="80%">
</p>

**As a Claude Code plugin** — inside Claude Code:

```
/plugin install director --marketplace EvolvingAgentsLabs/evolving-agents
/director:compile        # your CLAUDE.md, AGENTS.md and memory → linked statements (asks first; ~US$5 for ~60 files)
/director:reflect        # after a task: the correction becomes a proposed edit, kept only on your yes
/director:recipes        # optional: recipes a local model follows to plan from the memory, free and in ~10 s
```

From the next session on, every plan walks your memory and cites what it follows. The plugin carries the walk
intact: **0.929** through it, the same as the director harness, on the same 14 situations
([P0](../packages/director/docs/BRIEF-P0.md)). The status line names the memory and what waits for you; a band above
the prompt shows each proposed lesson with **Diff**, **Accept** and **Reject**, each plan's delegations before they
run, and each task delegated to another lane until its result is fetched.

**Published flat, too.** Keeping the lanes' facts as one directory of data instead of prose routed as well (0.880 vs
0.864) but broke clearance more often and did not follow a lane going down well enough, so it was not built
([D16](../packages/director/docs/BRIEF-D16.md)).

**Or the harness, with its two-lane interface** (frontier and local model side by side):

```bash
git clone https://github.com/EvolvingAgentsLabs/evolving-agents && cd evolving-agents/packages/director
python3 -m director.compile --claude-md ~/your-project/CLAUDE.md \
    --memory-dir ~/.claude/projects/<your-project>/memory --workdir ~/your-project
python3 -m director.server --config ~/.config/director/director.json      # open http://127.0.0.1:8740
```

Needs Python 3.10+ and [Claude Code](https://code.claude.com) (`claude` on your PATH). Standard library only. The
memory stays on your machine. What it costs, what it does not do yet, and every measurement:
[`packages/director/`](../packages/director/).

## From the SystemAgent to the director

<p align="center">
  <img src="img/lineage.jpg" alt="Left, a blueprint of an elaborate machine drawn only in outline and never built: gears, a filing cabinet, a network of pipes, a control panel, a looping arrow. Dashed ochre lines carry each part across to the right, where the same parts stand real on one bench: a chart table with linked index cards and a red route, a card-catalogue drawer, two brass dispatch tubes leaving for a window, a small brass box with a lever, and a ledger whose lines end in wax seals, tied to it by a green cord." width="100%">
</p>

The Evolving Agents Toolkit (EAT, 2025, kept in [`legacy/eat/`](../legacy/eat/)) had the right parts and pinned none of
them to a measurement: 18,680 lines, three test functions. The director keeps the parts and changes how each one
works — the full analysis is [**docs/EVOLUTION.md**](EVOLUTION.md).

| EAT, 2025 | the director, today | what changed, and the evidence |
|---|---|---|
| **SystemAgent** — the orchestrator | **the director**: plans, routes, delegates, reflects | it no longer runs a loop of its own; the harness does ([P0](../packages/director/docs/BRIEF-P0.md)) |
| **ArchitectZero** — designed solutions | **the walk**: index → statements → links, every step citing the rule it follows | the design comes from the team's own rules, not a second agent: 0.929 vs 0.589 ([D0](../packages/director/docs/BRIEF-D0.md)) |
| **Smart Memory** — experiences, searched by embedding | **operational memory**: atomic When / Do / Because statements, linked; plus an episodic journal and a directory | organisation, not retrieval, carries the effect: told to walk its files, the plain session reached 0.768; similarity search found 0.77 of the rules a case needs ([D13](../packages/director/docs/BRIEF-D13.md)) |
| **ContextBuilderTool / ExperienceRecorderTool** | **`/director:compile`** in, **`/director:reflect`** out | a lesson is a gated patch, not a log line: 0.0 → 0.781 on held-out probes, D0 re-checked at 0.964 ([D2](../packages/director/docs/BRIEF-D2.md)) |
| **SmartLibrary + SmartAgentBus** — find a capability, route to it | **lanes**: `delegate`, `lane_status`, `lane_fetch` to Claude Code, local models or any command | learned from one move: 0 / 6 → 6 / 6 ([D3](../packages/director/docs/BRIEF-D3.md)) |
| **Firmware** — governance as instructions to the model | **hooks and code**: the confidentiality rule in `lanes.py`, Accept behind lint and a guard | a rule that must hold is enforced, not asked for |
| **EvolutionStrategist** — which component to evolve | **the gate**: lint, the lesson's own probes (TEACHES), a targeted guard, a person's yes, an [agentvcs](https://github.com/EvolvingAgentsLabs/agentvcs) patch | the guard refused both planted breaks and no unchanged memory ([D6](../packages/director/docs/BRIEF-D6.md), [D7](../packages/director/docs/BRIEF-D7.md)) |
| **MongoDB** | Markdown files, versioned by agentvcs | files travel to every harness; every lesson can be blamed or bisected |

What we chose **not** to revive, with the reason for each — a memory agent in the read path, vector search as the
map, governance written as prompts, a framework adapter per vendor — is in [EVOLUTION §7](EVOLUTION.md#7-promises-still-open).

## One harness today, three next

<p align="center">
  <img src="img/harnesses.jpg" alt="The chart table of linked index cards, a brass gate and a sealed ledger. Three glass-fronted, locked card drawers travel from it: to a writing desk close by, joined by a solid green cord, and to two distant buildings on hills — a sawtooth-roofed workshop and a domed observatory — joined only by dashed ochre lines. From each place a single index card returns to the brass gate." width="100%">
</p>

**The harness is rented, never built.** The director decides *what* to do (the walk), *where* (the lanes), *whether
a lesson is kept* (the gate) and keeps the *evidence* (the journal). Model calls, tools, the sandbox and context
belong to a harness. Claude Code is one on your machine; Managed Agents is the same kind of harness hosted by
Anthropic; the Antigravity agent is Google's. All three read skills as `SKILL.md` and call MCP servers, and the
memory is plain files — so moving is packaging, not a rewrite.

| lane | harness | status |
|---|---|---|
| frontier | the Claude Code session you are in — the plugin | **built** — 0.929 ([P0](../packages/director/docs/BRIEF-P0.md)) |
| `claude -p` | Claude Code, headless | **built** — and the director itself runs unattended: `python -m director.headless` records a verifier's verdicts and consolidates lessons by policy |
| local | Ollama / llama.cpp; [lora-kernel](https://github.com/EvolvingAgentsLabs/lora-kernel)'s specialists | **built** — gemma4:12b plans from recipes at 0.806 vs 0.854, at no cost ([D15](../packages/director/docs/BRIEF-D15.md)) |
| command | anything with `submit` / `status` / `fetch` | **built** |
| `anthropic-managed` | Anthropic **Managed Agents** | **planned** — P1 |
| `antigravity` | Google's **Antigravity agent** | **planned** — P2 |

Each managed harness must earn its place the same way the plugin did: plan D0's 14 situations within 0.03 of 0.929
through it, or say by how much it fails. On every harness the operational memory is mounted **read-only** — a lesson
comes back through the gate, never written in place — and a cloud lane is not cleared for confidential work unless
you clear it. Conditions and reasons: [EVOLUTION §5](EVOLUTION.md#5-agents-and-harnesses).

## Measured

<p align="center">
  <img src="../packages/director/docs/img/results.png" alt="Three bar charts. D0, planning the team's way on 14 situations: today's session 0.589, today's session told to walk its files 0.768, the director 0.929. D2, learning from tasks on 8 held-out probes: before 0.0, after learning 0.781, and D0 re-checked afterwards 0.964. D3, learning the lane: public-facing drafts sent to the frontier before 0.0, after one move 1.0, and the internal control kept local 1.0." width="100%">
</p>

| plan by walking the memory | learn from a task | learn where work runs |
|---|---|---|
| ![An open index book with one line underlined; a red-pencil route steps across five index cards, leaving each from its highlighted line, to a plan of three items, each tagged back to the card it came from.](../packages/director/docs/img/walk.jpg) | ![A finished work order with a correction pinned to it; a new index card drafted at a desk; the card checked at a small brass gate; a ledger line closed with a fresh seal; the card filed in a cabinet drawer.](../packages/director/docs/img/learning.jpg) | ![A dispatch board with two columns of card slots, frontier on the left and local on the right; one card moves mid-air from the right column to the left; a lever below is thrown.](../packages/director/docs/img/lanes.jpg) |

Every number has a brief written before it ran, with the condition that would have falsified it; the flat ones are
published too — a small model walking the memory on its own did not plan well (D8–D12), and similarity search alone
does not find the rules a case needs (D13). Briefs and runners: [`packages/director/docs/`](../packages/director/docs/).

## Measured next: between sessions (director-bench)

Every number above is one planning call against a memory. The question that matters with the strongest models is
different: **does Claude Code get better from one session to the next on the work you repeat, and does the memory
ever make it worse?** [**director-bench**](https://github.com/EvolvingAgentsLabs/director-bench) measures that,
independent of the director: chained headless sessions on recurring tasks, a binary verifier per task, the memory
store the only thing kept between them. It compares Claude Code with no memory, with its native memory, with the
director, with both, and with the director learning from every session instead of only the verified ones.

- **Primary metric:** the repeated-error rate — failures whose signature already appeared earlier in the run. Also
  success, tokens per successful task (consolidation included), turns, the learning slope, retention, regressions
  against no memory, and errors the memory induced on a task where the convention changed.
- **Win criteria fixed before it runs:** repeated errors −30 % against native memory (95 % CI excluding 0), success
  not worse than −2 pp, regressions ≤ +1 pp, memory-induced errors no higher, tokens ≤ +10 %; and learning from
  every session must do worse than learning from verified ones, or the gate is not what helps.
- **First, a pilot** ([DB0](../packages/director/docs/BRIEF-DB0.md)): one family, three configurations, three seeds,
  90 sessions, to measure whether there is headroom and how noisy it is before the full 1,000-session run.

## What comes next

[**ROADMAP.md**](../ROADMAP.md) has the steps and how each will be judged; [docs/EVOLUTION.md §8](EVOLUTION.md#8-the-order)
orders the longer view: director-bench first (does the memory help between sessions, against Claude Code's own?),
component evolution (the promise of the name, still open), then the `anthropic-managed` and `antigravity` lanes,
then one memory package that installs on every harness. The directory as data was measured and not built (D16).

**Built with**, from the same lab: [**lora-kernel**](https://github.com/EvolvingAgentsLabs/lora-kernel), the local
lane — one small resident model, many specialists, a router that abstains to the frontier ·
[**agentvcs**](https://github.com/EvolvingAgentsLabs/agentvcs), the ledger — every lesson the director keeps is a
gated patch you can diff, blame and bisect · [**gene-evidence**](https://github.com/EvolvingAgentsLabs/gene-evidence),
a case study in reports where every sentence cites its evidence. All projects:
[evolvingagentslabs.github.io](https://evolvingagentslabs.github.io).


## Status history of the package

The status block that opened `packages/director/README.md` until 2026-10-24, kept as written:

**Evolving Agents' SystemAgent, rebuilt on what we measured.** The director keeps a team's operational memory —
how it works — as small, linked, verifiable statements; plans a task by walking it (page → statement → link →
statement), citing what each step follows; delegates each step to Claude Code, to a local model or to another
harness; and after each task proposes edits to the memory, gated and versioned, so the way of working is learned,
recalled, refined and reproduced. It ships as a **Claude Code plugin** (this folder) and as a local harness with a
two-lane interface; Anthropic's Managed Agents and Google's Antigravity agent are the next harnesses, planned.

> **Status — 2026-10-20 [ran].** Planning by walking the compiled memory: **0.929** against **0.589** for today's
> session ([D0](../packages/director/docs/BRIEF-D0.md)), and the same **0.929** through the plugin ([P0](../packages/director/docs/BRIEF-P0.md)). Learning from
> tasks: held-out probes **0.0 → 0.781**, D0 kept at 0.964 ([D2](../packages/director/docs/BRIEF-D2.md)). Learning where work runs: **0 →
> 6 / 6** after one move ([D3](../packages/director/docs/BRIEF-D3.md)). Accept guarded by a targeted guard and the lesson's own probes
> ([D4–D7](../packages/director/docs/BRIEF-D6.md)). A local model plans by following recipes Claude writes: gemma4:12b at **0.806** against
> **0.854**, in ~10 s and at no cost ([D15](../packages/director/docs/BRIEF-D15.md), after D8–D14's flat and partial results, below).
> **Planned:** the `anthropic-managed` and `antigravity` lanes (P1, P2).
>
> **2026-10-19: the directory of lanes, published flat [ran].** Lane facts kept as one directory of data, cited by the
> statements, route as well as the same facts in prose (0.880 vs 0.864, 15 cases) and leave D0 at 0.929 — but the plans
> broke clearance more often (5 vs 1) and followed a lane going down below the bar (0.762 / 0.786 against 0.8 / 0.9):
> **DIRECTORY LOSES**, so it is not built and the prose stays ([D16](../packages/director/docs/BRIEF-D16.md)). Next: component evolution on
> the recipes ([D17](../packages/director/docs/BRIEF-D17.md)).
>
> **2026-10-21 — all of the above planned with Sonnet 5.5.** With Opus 5.5 planning, the session alone follows D0's
> rules 0.964 ([D0O](../packages/director/docs/BRIEF-D0O.md)) and keeps a thread's decisions 0.80 vs the director's 0.85 ([C2](../packages/director/docs/BRIEF-C2.md));
> the director's measured edge with Opus is cost (−34 % per plan). On situations from after the freeze, Opus alone
> left room only under its own judge ([F1O](../packages/director/docs/BRIEF-F1O.md)); Gemini follows the same rules as plain notes at
> 0.875 ([PG](../packages/director/docs/BRIEF-PG.md)).
>
> **2026-10-22 — what it is for now, and the next measurement.** With the strongest models the director is an
> auditable, portable memory of how a team works, and the question moves from one plan to many sessions: does Claude
> Code get better from one session to the next on recurring work, against its own native memory, without the memory
> ever making it worse? That is measured by [director-bench](https://github.com/EvolvingAgentsLabs/director-bench),
> an open benchmark independent of the director, after a pilot ([DB0](../packages/director/docs/BRIEF-DB0.md)). For it, the director runs
> with no person present: [`director.headless`](#without-a-person-directorheadless).

