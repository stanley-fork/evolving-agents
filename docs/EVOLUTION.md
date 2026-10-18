# From the SystemAgent to the director

**Where Evolving Agents goes next: memory, agents, and the harnesses they run on.**

*Written 2026-10-18. An analysis and a plan, not a result. Every number cited links to the brief that measured it;
every step proposed names how it will be judged before it is built, as [ROADMAP.md](../ROADMAP.md) requires.*

## In one page

- **The director is the SystemAgent, rebuilt on what we measured.** The 2025 toolkit (EAT) had the right parts: an
  orchestrator, a library of agents, a bus to reach them, a memory of experiences, rules that must hold. It pinned
  none of them to a measurement (three test functions in 18,680 lines). The director keeps the parts and changes how
  each one works.
- **Memory is four memories, not one.** *Operational* (how we work, gated), *encyclopedic* (what is true, verified
  against sources), *episodic* (what happened, raw material for lessons) and a *directory* (who can do what, where).
  Today the first is well built, the second lives in lora-kernel, and the last two are scattered across files.
  Recipes are a compiled view of the first, not a fifth.
- **The harness is rented, never built.** The director plans, routes and learns; the loop, the tools and the sandbox
  belong to a harness: Claude Code today, and Anthropic's Managed Agents or Google's Antigravity agent next. Both
  read skills as `SKILL.md`, both take remote MCP servers, both run in a sandbox and in the background. **The
  same memory and the same gate can therefore run on three harnesses**, and a lane is the place to plug them in.
- **The next step is ordering, then one harness at a time.** The memory model and a module boundary first, the
  directory as data second, and then a managed-agent lane, each harness held to the plugin's own bar: plan D0's 14
  situations at 0.929 through it, or say by how much it fails.

## 1. Where we came from

The Evolving Agents Toolkit (2025, now in [`legacy/eat/`](../legacy/eat/)) was agent-centric and ran on MongoDB:

| EAT piece | what it did |
|---|---|
| **SystemAgent** | a ReAct orchestrator: search the library, create or evolve a component, delegate over the bus, generate and run workflows |
| **SmartLibrary** | the catalogue of agents and tools, found by two embeddings (what a component is, and when it applies) |
| **SmartAgentBus** | a registry of capabilities and a router to whoever offered them |
| **Smart Memory** | *experiences* (goal, components used, decisions, outcome), embedded and searched; a `MemoryManagerAgent` answered memory requests in natural language; `ContextBuilderTool` pulled similar experiences into context before planning, `ExperienceRecorderTool` wrote one after |
| **Firmware** | governance rules, written as instructions to the model |
| **EvolutionStrategist** | proposed which components to evolve, from library and memory |

The diagnosis is on record: an architecture written down and never pinned to anything that could contradict it.
10,165 of its lines were deleted because the Agent SDK does the same better
([WHAT-WAS-DELETED.md](WHAT-WAS-DELETED.md)).

## 2. What we measured since

| question | answer | brief |
|---|---|---|
| Does a memory of linked statements make plans follow the team's way? | today's session 0.589, told to walk its files 0.768, the director 0.929 (14 real situations, blind judge) | [D0](../packages/director/docs/BRIEF-D0.md) |
| Can it learn from a task without breaking what it knew? | held-out probes 0.0 → 0.781; D0 re-checked 0.964 | [D2](../packages/director/docs/BRIEF-D2.md) |
| Can it learn where work runs? | public drafts to the frontier 0/6 → 6/6 after one move; internal text stayed local | [D3](../packages/director/docs/BRIEF-D3.md) |
| Can Accept refuse a bad lesson? | the targeted guard refused both planted breaks and no unchanged memory; a lesson is kept only if its own probes show it TEACHES | [D6](../packages/director/docs/BRIEF-D6.md), [D7](../packages/director/docs/BRIEF-D7.md) |
| Does the walk survive inside Claude Code? | 0.929 through the plugin, the same as the director | [P0](../packages/director/docs/BRIEF-P0.md) |
| Is similarity search enough to find the rules a case needs? | no: lexical + local embeddings reached 0.77 / 0.74 of them; partial | [D13](../packages/director/docs/BRIEF-D13.md) |
| Can a small local model plan? | not by walking (D8–D12); yes by following recipes Claude writes: 0.806 vs 0.854, at no cost | [D8](../packages/director/docs/BRIEF-D8.md)–[D15](../packages/director/docs/BRIEF-D15.md) |
| Does storing raw traces help recall? | no measured advantage over naive matching | [PLAN.md, M2](../PLAN.md) |

Five principles follow from these, and the rest of this document leans on them:

1. **Structure beats similarity.** Atomic statements, links inside them and an index of situations found what
   embeddings missed. Similarity is a fallback, not the map.
2. **A lesson is a gated patch, not a log line.** It enters only through lint, its own probes (TEACHES) and a
   person's yes, and it is versioned in agentvcs so it can be blamed or bisected.
3. **Memory operations are code; models propose.** No memory agent sits in the read path. The walk, the lint, the
   gate and the versioning are deterministic; a model drafts edits and answers yes/no questions.
4. **Rules that must hold are enforced in code.** Firmware asked the model not to use dangerous imports; a hook
   denies the call. The same way, a confidential task is refused by any lane not cleared for it, the frontier
   included, in `lanes.py`, not only as a learned statement.
5. **The harness is rented.** The loop, the tools, context management and the sandbox are solved by others. What is
   ours is the memory, the gate, the routing and the measurements.

## 3. EAT, piece by piece

| EAT | today | next |
|---|---|---|
| SystemAgent | **the director**: plans by walking the memory, delegates, reflects | unchanged in role; harness-neutral (§5) |
| ContextBuilderTool | **the walk** (index → statements → links), or **recipes** for a local model | recipes stamped with the memory version they came from |
| ExperienceRecorderTool | **`/director:reflect`** → `memory_propose` → check → accept | reads the episodic journal instead of the session alone |
| Smart Memory (experiences) | scattered: the event log, `delegations/`, `candidates/`, `lane-moves.jsonl` | **the episodic journal**: one append-only file per project, never read by planning |
| SmartLibrary | scattered: `lanes.json`, lora-kernel's specialists and domains, routing statements in prose | **the directory**: capabilities as data, with measured reliability and its evidence |
| SmartAgentBus | **lanes**: `delegate`, `lane_status`, `lane_fetch` over configured commands | more lane kinds: managed agents (§5) |
| EvolutionStrategist | the reflection's probes and the targeted guard | reliability in the directory, updated from outcomes |
| Firmware | hooks; the confidentiality rule in code | the same rule applied per harness and per region |
| MongoDB | files, versioned by agentvcs | unchanged: files travel to every harness |

## 4. The memory model

| | **operational** | **encyclopedic** | **episodic** | **directory** |
|---|---|---|---|---|
| holds | how we work: When / Do / Because, linked | what is true: projects, numbers, definitions | what happened: tasks, plans, delegations, corrections, outcomes | who or what can do which work, where, at what cost, how reliably |
| written by | a lesson: a proposed edit | a source: a document, a tool's output | the director and the lanes, as things happen | configuration (private) and outcomes |
| gate | lint, probes (TEACHES), the guard, a person's yes; agentvcs patch | checked against its source; replaced when the source changes | none (append-only); it is evidence, not instruction | config is the person's; a reliability change needs the outcomes that justify it |
| read by | every plan (the walk), recipes | a statement that cites it; a specialist answering | the reflection only | the planner, when it routes |
| lives in | `.director/memory/` | lora-kernel's library; linked, not copied | `.director/journal.jsonl` (proposed) | `.director/directory.json` + `~/.config/director/` (proposed) |
| measured | D0, D2, D6, D7, P0 | lora-kernel: edited facts 17/17 with no retraining | M2: raw traces alone did not help | D3 (in prose); a directory as data is not yet measured |

**Two rules hold across them.** Nothing reaches a plan without a gate on its write path. The episodic journal, which
has none, is never read by planning: it only feeds the reflection, which goes through the gate. And no memory is an
agent: each is files plus code.

**Harness memories are sources, not rivals.** Claude Code's auto-memory, a Managed Agents memory store a session
wrote to, Antigravity's Memory Bank: each holds things a model decided to remember, with no gate. `/director:compile`
already reads Claude Code's auto-memory as a source; the others enter the same way, through compile or a proposed
lesson, never by being read straight into a plan.

**Derived views are not memories.** `Index.md`'s situations, the guard's baseline and the recipes are compiled from
the operational memory. Each should record the memory version it was built from, so that a stale view is caught by
lint, not by a bad plan.

## 5. Agents and harnesses

### The split

The director never runs a loop of its own. It decides **what** to do (the walk), **where** (the directory),
**whether a lesson is kept** (the gate), and it keeps the **evidence** (the journal). The harness does the rest:
model calls, tools, files, the sandbox, context management.

Claude Code is the analogy to keep in mind. It is a harness on the person's machine; **Managed Agents is the same
kind of harness, hosted by Anthropic; Antigravity is Google's.** The director already drives the first (the plugin,
and `claude -p`); the other two are the same relationship in another place.

A **lane** is a harness, a place and a model:

| lane kind | harness | runs | status |
|---|---|---|---|
| `frontier` | the Claude Code session the person is in | on their machine | built (P0) |
| `claude -p` | Claude Code, headless | on their machine, isolated | built (D1) |
| local | Ollama / llama.cpp; lora-kernel's specialists | on their machine | built (D1, D4, D15) |
| command | anything with `submit` / `status` / `fetch` | wherever the command reaches | built (lanes) |
| **anthropic-managed** | Anthropic **Managed Agents** | Anthropic's sandbox, or a self-hosted one | proposed |
| **antigravity** | Google's **Antigravity agent** (Gemini API's managed agents; Gemini Enterprise Agent Platform) | Google's sandbox | proposed |

### What the two managed harnesses offer

| | Anthropic Managed Agents | Google Antigravity agent |
|---|---|---|
| the agent | a stored, versioned config (model, system prompt, tools, MCP servers, skills); a session pins a version | a named agent bundling instructions (`AGENTS.md`), skills and an environment, invoked by id |
| the sandbox | a container per session; cloud, or self-hosted (the loop stays at Anthropic, tools run on your machines) | a Linux sandbox; persists across calls through an environment id |
| skills | `SKILL.md` | `SKILL.md`, mounted under `.agents/skills/` |
| MCP | MCP servers on the agent; credentials in vaults, substituted at egress | remote MCP servers as tools |
| long work | sessions stream events; scheduled deployments (cron); hard dollar budgets per session | background interactions, polled or streamed by id |
| memory | **memory stores**: small text files mounted at `/mnt/memory/<store>/`, every change an immutable version, mountable **read-only** | **Memory Bank**: facts about a user, extracted automatically by Gemini across sessions |
| checking work | **outcomes**: a separate grader iterates the work against a rubric | — (not found in the docs read) |
| status (Oct 2026) | beta | preview; the Agent Platform docs say limited testing, not production |

Sources: Anthropic's Managed Agents documentation; Google's [Antigravity agent](https://ai.google.dev/gemini-api/docs/antigravity-agent),
[managed agents](https://ai.google.dev/gemini-api/docs/custom-agents) and
[background execution](https://ai.google.dev/gemini-api/docs/background-execution) pages, the
[Agent Platform overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/managed-agents) and
[Memory Bank](https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/api-quickstart).
Both are early products: Google's Agent Engine became the Agent Platform this year, and the Antigravity agent's
identifiers carry a date (`antigravity-preview-05-2026`, then later ones). An adapter pins the version it was tested on.

### Does supporting them make sense?

**Yes, for four reasons, and on four conditions.**

Reasons:

1. **The memory is already portable.** The operational memory is small Markdown files; the walk is `DIRECTOR.md`,
   which is a skill in all but name; the gate is an MCP server. Both harnesses mount files, load `SKILL.md` and call
   remote MCP. Supporting them is packaging, not a rewrite.
2. **Work that should not run on the person's machine.** Long tasks, nightly ones, many in parallel. Managed Agents'
   scheduled deployments and Antigravity's background interactions cover what `claude -p` on a laptop does not.
3. **Lanes already have the interface.** A lane is three commands (`submit`, `status`, `fetch`). An adapter that
   starts a session, polls it and downloads its outputs is a lane without touching the director's core, and the
   confidentiality rule applies to it unchanged.
4. **A second provider makes the measurements stronger.** Today the planner and the judge are both Claude. A plan
   by one provider judged by another, or the same memory walked by two models, separates "the memory helps" from
   "Claude likes Claude's plans".

Conditions:

1. **No harness writes the operational memory.** Managed Agents can mount a memory store read-write; ours is mounted
   **read-only**, and a lesson comes back as a proposal through the MCP gate. Memory Bank's automatic extraction is
   the opposite of a gated lesson: what it extracts may serve as episodic or user facts, never as operational
   statements without passing the gate.
2. **A cloud lane is not cleared for confidential work by default.** `confidential_ok` stays off for managed lanes
   unless the person sets it, with the provider's region (Anthropic's `inference_geo`, Google's regional endpoints)
   recorded in the directory entry.
3. **Each harness earns its place on D0.** A managed lane is trusted for planning only after the walk through it
   plans D0's 14 situations within 0.03 of 0.929 (the P0 bar). Below that, it may still execute tasks the director
   planned elsewhere.
4. **Pinned versions, preview status said out loud.** The adapter records the agent version or model identifier it
   was tested on; a preview product is never the only lane for anything.

## 6. Second pass: what would make it more powerful

The plan above orders what exists. These go further; each is a hypothesis with its measurement, not a promise.

1. **The memory as a package — "your way of working, in any agent."** One versioned artifact: pages, Index, recipes,
   directory and the walk as a skill, with the gate as a remote MCP server. Installed as a Claude Code plugin (today),
   an Agent SDK plugin, a Managed Agents memory store plus skill, or an Antigravity skill. *How we will know:* P0's
   parity on each harness, from the same artifact, byte for byte.
2. **The gate as a service.** Today `memory_propose / check / accept` run beside the session. Served remotely, any
   cloud agent proposes a lesson and the person decides on the band or the web interface; the memory never changes
   anywhere else. *How we will know:* a lesson proposed from a managed session passes or fails the same gate as one
   proposed locally, on D2's probes.
3. **Reliability that is earned, not described.** The SmartLibrary ranked components by how similar they looked. The
   directory ranks lanes by what happened: each delegation's outcome, from the journal, updates its entry
   (succeeded, abstained, judged, cost). Routing then rests on evidence. *How we will know:* D16, below.
4. **Probes and guards that run while nobody watches.** Managed Agents' outcomes are a rubric-graded loop, which is
   what D7's probes already are; scheduled deployments can re-run the targeted guard each night against the live
   memory and file a finding when a reliable case starts failing. *How we will know:* a planted break is caught the
   night after it lands.
5. **Lessons that become skills.** EAT evolved components. Here the analogue is a cluster of statements that always
   travel together, compiled into a `SKILL.md` any harness can load, still cited back to its statements. *How we will
   know:* the compiled skill plans its situations as well as the walk does, at lower cost.
6. **A team's memory across providers.** ROADMAP step 4 (lessons shared through agentvcs remotes) gains a read-only
   copy per harness: one source of truth, many mounts. *How we will know:* two members' lessons merged without losing
   either's probes, read by two harnesses.

**What not to rebuild:** a database as the memory, similarity search as the map, a memory agent in the read path,
rules as instructions to the model, our own loop or sandbox. Each was tried in 2025 or measured since.

## 7. The order

| | step | how we will know |
|---|---|---|
| 1 | **The memory model in code.** This document's §4 as the reference; a `director/memory/` module (format, lint, walk, gate, versioning) apart from the agent that plans and delegates. Kept in this package until a second repository needs it. | existing tests pass unchanged; no measurement needed |
| 2 | **The directory as data.** `lanes.json`, lora-kernel's domains and the routing statements become one directory: domains, `confidential_ok`, region, cost, reliability with its evidence. Statements cite entries instead of describing them. | **D16** (brief first): the directory routes D3's cases at least as well as the prose did, and D0 holds at 0.929 |
| 3 | **The episodic journal.** One append-only file; `/director:reflect` reads it. | a lesson drafted from the journal passes D2's probes as one drafted from the session did |
| 4 | **Views stamped with their memory version.** Recipes and baseline record it; lint flags a stale one. | a test: edit the memory, lint reports the recipes stale |
| 5 | **An `anthropic-managed` lane.** An adapter behind `submit / status / fetch`; the memory mounted read-only; the walk as a skill; the gate over MCP. | **P1**: D0 through it within 0.03 of 0.929; cost per plan published |
| 6 | **An `antigravity` lane.** The same artifact on Google's harness. | **P2**: the same bar; and D0's plans judged by the other provider, to measure self-preference |
| 7 | **The memory package.** One export that installs on every harness above. | P0, P1 and P2 re-run from the same artifact |

Steps 1–4 order what exists and need no new provider. Steps 5–7 are where the director stops being a Claude Code
plugin and becomes what the SystemAgent meant to be: one orchestrator with one memory, over whichever agents do the
work.

## Open questions

- Should encyclopedic memory stay entirely in lora-kernel, or keep a small local cache the walk can cite offline?
- Who owns a directory entry's reliability when two people's outcomes disagree: a merge rule, or a person?
- Does a model other than Claude walk the memory as well? D15 says a 12B can follow recipes; nobody has asked whether
  Gemini can walk.
- At what point is `director/memory/` its own package: when lora-kernel imports it, or when a second team uses it?
