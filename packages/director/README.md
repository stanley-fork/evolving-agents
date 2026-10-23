# director

**Evolving Agents' SystemAgent, rebuilt on what we measured.** The director keeps a team's operational memory —
how it works — as small, linked, verifiable statements; plans a task by walking it (page → statement → link →
statement), citing what each step follows; delegates each step to Claude Code, to a local model or to another
harness; and after each task proposes edits to the memory, gated and versioned, so the way of working is learned,
recalled, refined and reproduced. It ships as a **Claude Code plugin** (this folder) and as a local harness with a
two-lane interface; Anthropic's Managed Agents and Google's Antigravity agent are the next harnesses, planned.

> **Status — 2026-10-20 [ran].** Planning by walking the compiled memory: **0.929** against **0.589** for today's
> session ([D0](docs/BRIEF-D0.md)), and the same **0.929** through the plugin ([P0](docs/BRIEF-P0.md)). Learning from
> tasks: held-out probes **0.0 → 0.781**, D0 kept at 0.964 ([D2](docs/BRIEF-D2.md)). Learning where work runs: **0 →
> 6 / 6** after one move ([D3](docs/BRIEF-D3.md)). Accept guarded by a targeted guard and the lesson's own probes
> ([D4–D7](docs/BRIEF-D6.md)). A local model plans by following recipes Claude writes: gemma4:12b at **0.806** against
> **0.854**, in ~10 s and at no cost ([D15](docs/BRIEF-D15.md), after D8–D14's flat and partial results, below).
> **Planned:** the `anthropic-managed` and `antigravity` lanes (P1, P2).
>
> **2026-10-19: the directory of lanes, published flat [ran].** Lane facts kept as one directory of data, cited by the
> statements, route as well as the same facts in prose (0.880 vs 0.864, 15 cases) and leave D0 at 0.929 — but the plans
> broke clearance more often (5 vs 1) and followed a lane going down below the bar (0.762 / 0.786 against 0.8 / 0.9):
> **DIRECTORY LOSES**, so it is not built and the prose stays ([D16](docs/BRIEF-D16.md)). Next: component evolution on
> the recipes ([D17](docs/BRIEF-D17.md)).
>
> **2026-10-21 — all of the above planned with Sonnet 5.5.** With Opus 5.5 planning, the session alone follows D0's
> rules 0.964 ([D0O](docs/BRIEF-D0O.md)) and keeps a thread's decisions 0.80 vs the director's 0.85 ([C2](docs/BRIEF-C2.md));
> the director's measured edge with Opus is cost (−34 % per plan). On situations from after the freeze, Opus alone
> left room only under its own judge ([F1O](docs/BRIEF-F1O.md)); Gemini follows the same rules as plain notes at
> 0.875 ([PG](docs/BRIEF-PG.md)).
>
> **2026-10-22 — what it is for now, and the next measurement.** With the strongest models the director is an
> auditable, portable memory of how a team works, and the question moves from one plan to many sessions: does Claude
> Code get better from one session to the next on recurring work, against its own native memory, without the memory
> ever making it worse? That is measured by [director-bench](https://github.com/EvolvingAgentsLabs/director-bench),
> an open benchmark independent of the director, after a pilot ([DB0](docs/BRIEF-DB0.md)). For it, the director runs
> with no person present: [`director.headless`](#without-a-person-directorheadless).

<p align="center">
  <img src="docs/img/results.png" alt="Three bar charts. D0, planning the team's way on 14 situations: today's session 0.589, today's session told to walk its files 0.768, the director 0.929. D2, learning from tasks on 8 held-out probes: before 0.0, after learning 0.781, and D0 re-checked afterwards 0.964. D3, learning the lane: public-facing drafts sent to the frontier before 0.0, after one move 1.0, and the internal control kept local 1.0." width="100%">
</p>

## Where it comes from

![Left, a blueprint of an elaborate machine drawn only in outline and never built. Dashed ochre lines carry each of its parts across to the right, where the same parts stand real on one bench: a chart table of linked index cards with a red route, a card-catalogue drawer, two brass dispatch tubes leaving for a window, a small brass box with a lever, and a ledger whose lines end in wax seals.](../../docs/img/lineage.jpg)

The Evolving Agents Toolkit (2025, [`legacy/eat/`](../../legacy/eat/)) had a SystemAgent, ArchitectZero, Smart
Memory, a SmartLibrary and a SmartAgentBus, Firmware and an EvolutionStrategist — and three test functions. The
director keeps each idea and pins it to a measurement: the SystemAgent plans by walking the memory and leaves the
loop to the harness; ArchitectZero's design comes from the team's own rules; Smart Memory becomes linked When / Do /
Because statements; the library and the bus become lanes; Firmware becomes rules enforced in code; evolution becomes
a lesson that passes a gate and lands as an agentvcs patch. The table, piece by piece:
[the repository's README](../../README.md#from-the-systemagent-to-the-director); the analysis:
[`docs/EVOLUTION.md`](../../docs/EVOLUTION.md).

## How it works, in three pictures

**It plans by walking the memory.** The index names the situations a task is in; each entry statement links,
inside its own line, to the next one that applies; the plan's every step cites the statement it follows.

![An open index book with one line underlined; a red-pencil route leaves it and steps across five index cards, leaving each from its highlighted line, to a plan of three items, each tagged back to the card it came from.](docs/img/walk.jpg)

**It learns from a task without breaking what it knew.** After a task, the correction it received becomes a
drafted statement; the statement passes a gate (it lints clean, and a person accepts it); the edit is sealed in
agentvcs's ledger as a patch; the statement is filed where the next walk will find it (D2, D2b).

![A finished work order with a correction pinned to it; a new index card drafted at a desk; the card checked at a small brass gate; a ledger line closed with a fresh seal; the card filed in a cabinet drawer; a dashed arc back to the start.](docs/img/learning.jpg)

**It learns where work runs.** A plan waits; the user can move any task between the frontier and the local lane;
the move goes into the next reflection, and the director plans the next similar task in that lane (D3).

![A dispatch board with two columns of card slots: the left bordered in rust under a window with a distant domed building, the right in green under a small local machine. One card moves mid-air from the right column to the left; a lever below is thrown; the card beside its empty slot stays put.](docs/img/lanes.jpg)

Images: [`docs/img/`](docs/img/) — the plates' prompts are in the repository's [`docs/img/IMAGES.md`](../../docs/img/IMAGES.md);
`results.png` is drawn by [`docs/img/results.py`](docs/img/results.py) from the briefs' numbers.

## Where it sits

| piece | reuses |
|---|---|
| operational memory, atomic statements with links | the memory design of [lora-kernel](https://github.com/EvolvingAgentsLabs/lora-kernel) (pages of verifiable statements; the weights hold the navigation, the base holds the content) |
| frontier lane | Claude Code, driven as `claude -p` — the isolated, audited way [agentvcs](https://github.com/EvolvingAgentsLabs/agentvcs) `merge resolve` already runs it |
| local lane | lora-kernel's gateway on llama.cpp: local experts that abstain to the frontier |
| refining without breaking | agentvcs: each memory edit is a gated patch; `blame` says which edit moved a score, `bisect` finds the one that broke it |

## The plan, in gates

| step | what | gate |
|---|---|---|
| **D0** | today's session vs a session that walks the compiled memory, on real situations | baseline below 0.85 (headroom), then treatment ≥ +0.15 — [brief](docs/BRIEF-D0.md) |
| D1 | the director's loop and its interface: the director on top (capabilities, plan, conversation), below it the frontier lane (left) and the local lane (right), from one event log | **built** — below |
| D2 | learning: memory edits proposed after each task, gated on D0's suite, versioned in agentvcs | **learns** — [brief](docs/BRIEF-D2.md) |
| D3 | the user moves a task between lanes; the move becomes a routing statement the director learns | **learns the lane** — [brief](docs/BRIEF-D3.md) |
| D4–D7 | Accept guarded: an edit is kept only if it breaks none of the cases that relied on what it changes, and its own probes show it changes the plan | **targeted guard + positive probe** — [D4](docs/BRIEF-D4.md), [D5](docs/BRIEF-D5.md), [D6](docs/BRIEF-D6.md), [D7](docs/BRIEF-D7.md) (below) |
| P0 | the walk inside Claude Code, as a plugin | **0.929, the same as the director** — [brief](docs/BRIEF-P0.md) |
| D8–D10 | can a local 12B walk the memory itself? | **not navigable**: core recall 0.32 → 0.45, precise (~0.9), but it reaches too little of what each case needs — [D8](docs/BRIEF-D8.md), [D9](docs/BRIEF-D9.md), [D10](docs/BRIEF-D10.md) |
| D11–D12 | do the 12B's plans hold up, on a bank no strategy was designed on? | **memory helps, gap remains**: 0.68 against Sonnet's 0.92; a plan composed by code did not close it — [D11](docs/BRIEF-D11.md), [D12](docs/BRIEF-D12.md) |
| D13–D14 | retrieval in code (lexical + local embeddings), the plan written by Claude | retrieval **partial** (0.77 / 0.74); the split planned **0.837 vs 0.892** at 17× less per plan, short of its gate by 0.005 — [D13](docs/BRIEF-D13.md), [D14](docs/BRIEF-D14.md) |
| D15 | Claude writes meta-recipes from the memory; code runs them; the local model only picks a recipe and answers yes/no | **recipes work** with the 12B: 0.806 vs 0.854, 3 / 3 / 6 per case; not yet with 4B (0.58) or 2B (0.26) — [brief](docs/BRIEF-D15.md) |
| D16 | the lanes' facts as one directory of data the statements cite, against the same facts in prose (EVOLUTION step 2) | **directory loses**: routes as well (0.880 vs 0.864) and D0 holds (0.929), but 5 clearance violations vs 1 and a lane going down followed at 0.762 / 0.786 against 0.8 / 0.9; not built — [brief](docs/BRIEF-D16.md) |
| D17 | gated evolution of the recipes from D15's own runs (EVOLUTION step 3) | brief written; runs on the lab's machine (Ollama) — [brief](docs/BRIEF-D17.md) |
| H | the director with no person present: a verifier's verdicts in the journal, lessons consolidated and accepted by a declared policy (`verified`: only from sessions whose verifier passed; `all`: the ablation) | **built and tested offline**, not yet run live — [below](#without-a-person-directorheadless) |
| — | where the director should beat other memories, and why: five hypotheses (verified gating, methods with their checks, editing in place, recall, cost), what would show each, and where it should not win | **written before the bench measures** — [analysis](docs/WHERE-IT-SHOULD-WIN.md) |
| DB0 | the director-bench pilot: one family, no memory / native memory / the director, 3 seeds, 90 sessions | **brief written** — headroom (no-memory repeats its errors) and spread before the full run — [brief](docs/BRIEF-DB0.md) |
| P1 | an `anthropic-managed` lane: Managed Agents behind `submit / status / fetch`, the memory mounted read-only, the walk as a skill, the gate over MCP | **planned** — D0 through it within 0.03 of 0.929; cost per plan published ([EVOLUTION §5](../../docs/EVOLUTION.md#5-agents-and-harnesses)) |
| P2 | an `antigravity` lane: the same artifact on Google's harness | **planned** — the same bar, and D0's plans judged by the other provider |

## As a Claude Code plugin

```
/plugin install director --marketplace EvolvingAgentsLabs/evolving-agents
```

This folder is the plugin (`.claude-plugin/plugin.json`). Claude Code keeps running the session; the plugin adds
only what it lacks:

| piece | what it does |
|---|---|
| `/director:compile` | compiles your `CLAUDE.md` / `AGENTS.md`, auto-memory, agents and skills into `.director/memory/` (asks first; ~US$5 for ~60 files) |
| `SessionStart` hook | where that memory exists, puts the walk (`DIRECTOR.md`) into the session: every plan walks the memory and cites what it follows; since 0.2.6 the walk does not end with the plan — at each decision point of the work (a definition, a convention, a value someone will score or reuse) the session searches the memory again and follows it, and what it hands over cites the statements it followed and lists the choices the memory does not hold; it also says how many of the person's messages and corrections no reflection has read yet, and when the session is running an older director than the one installed (restart Claude Code to load it; `claude --resume` keeps the conversation); silent elsewhere |
| `UserPromptSubmit` hook | every message the person sends goes to the episodic journal as a `person` entry (keys scrubbed), so a decision made in passing is not lost before `/director:reflect` reads it; nothing reaches the memory except through the gate. If the installed director changes under a running session, the next prompt says so (restart; `claude --resume`), once per installed version per session |
| `PreToolUse` hook (Bash) | the **public guard**: a `git commit`, `git push` or `gh pr`/`gh issue` text bound for a repository not listed as private is refused when what it publishes carries a term from a private list or something shaped like a key — enforced in code (`director/public_guard.py`; silent without its private config) |
| `/director:reflect` | reads the journal since the last reflection and turns the person's **corrections and decisions** (a model, a cap, a tool, "from now on …"), the **methods** a task reached only after failed attempts (steps with the check that caught each failure, without that task's values) and, since 0.2.6, the **scored failures** of the agents that did the work (items failing in more than one run: a reference to check, or a definition the team never fixed, proposed as a **convention** for the person to decide) into edits plus two probes; a lesson the plain runs already follow is reported as redundant, not proposed; nothing touches the memory until you say yes |
| `/director:recipes` | writes (or rewrites) the meta-recipes a local model follows to plan from this memory (`recipes.json` beside the pages), then lints them and reports how much of the memory they cite (D15); stamps them with the memory version they were written from, so `recipes_lint`, `plan_local` and the status line say when an accepted lesson has made them stale |
| MCP server `director` | `memory_propose` (a candidate copy, linted, diffed), `memory_check` (D7's positive probe, in the background: TEACHES / REDUNDANT / NO EFFECT), `memory_accept` (your yes, a clean lint and — if checked — TEACHES; versioned as a gated agentvcs patch when agentvcs is installed; a proposal made before another lesson was accepted is rebased onto the live memory, same id, when none of the statements or Index lines it touches changed — `memory_status` reports `rebased_from` and the new diff — and refused with `conflict` naming them when they did, since accepting would undo what came since; an edit whose `situations` hold a whole Index line (`→` or `[[`) is refused), `memory_reject` (your no, or a conflicting proposal closed), `memory_lint`, `memory_status`; `recipes_lint` and `plan_local` (a plan made by following the recipes with a local model on Ollama, `DIRECTOR_LOCAL_MODEL`, default `gemma4:12b-mlx`; free, and `abstain` when no recipe fits); `lanes_list`, `plan_propose`, `plan_status`, `delegate`, `lane_status`, `lane_fetch` (lanes, below); `journal_read` and `journal_note` (the episodic journal, below) |
| status line and band (`hooks/ui.tsx`) | where that memory exists, the status line reads `director · 286 statements · 3 learned · 1 to decide · 4 to reflect`; when a plan waits, a band above the prompt shows each task with its lane (**Run**, **Move**, **Cancel**); when a lesson waits for a decision, it shows its edits and its check with **Diff**, **Accept** and **Reject** (Accept is your yes, through the same gate as `memory_accept`; it is offered only when the proposal lints clean and its check, if one ran, says TEACHES), and each delegation not yet brought back with **Status** and **Fetch**. What you press reaches the session with your next prompt. Needs a Claude Code with plugin UI modules; on older versions everything else works as before |
| live activity (`hooks/ui.tsx`) | every tool call the session makes is shown as it runs, on its lane: the status line adds `frontier ▶ <what runs now> (12s…) │ local ▶ <plan_local or a delegation>`, and the band above the prompt shows **Director · activity** — the walk through the memory (`walk → Headroom`), the director's own tools, the journal's latest events and each lane's last step. `/director-activity` hides or shows the band; the status line always keeps the lanes |

**What you see in Claude Code.** No other window: the director lives in the session. The status line names the memory
and what waits for you; a band appears above the prompt only when something does. An illustration, not a capture:

```
╭─ Director · a plan to run ───────────────────────────────────────────╮
│ Draft the 0.7 release note and summarise this week's internal log    │
│ frontier  Draft the release note (public)                            │
│ local     Summarise the week's log                          [Move]   │
│ [Run]  [Cancel]                                                      │
├─ Director · a lesson to decide ──────────────────────────────────────┤
│ Screenshots are cropped before publishing                            │
│ add Confidentiality#crop_screens · check: TEACHES                    │
│ [Diff]  [Accept]  [Reject]                                           │
╰──────────────────────────────────────────────────────────────────────╯
> _
  director · 291 statements · 3 learned · 1 to decide · 1 plan waiting
```

Nothing is delegated before **Run**; **Move** offers only a lane the task is cleared for; **Accept** appears only when
the lesson lints clean and its check, if one ran, says TEACHES. What you press reaches the session with your next
prompt (*"On the director's band, the person: moved task t2 from local to frontier"*). The memory is found at
`$DIRECTOR_MEMORY`, else `.director/memory/` in the project, else `memory/`; with none, the plugin stays silent. To use a
memory kept elsewhere: `DIRECTOR_MEMORY=~/path/to/memory claude`.

**Measured [ran] 2026-10-17 ([P0](docs/BRIEF-P0.md)):** the walk through the plugin's hook plans D0's 14 situations
at **0.929**, the same as the director driving Claude Code (T − P = 0.0; 27 of 28 plans cite the statements they
follow), against 0.589 for a session without it.

**Lanes.** The session is the `frontier` lane. Other lanes are commands named in a private config
(`$DIRECTOR_LANES`, default `~/.config/director/lanes.json`: `submit`, `status` and `fetch` per lane), so the public
code knows nothing about what they run. **With a person at the session, the plan waits for them:** `plan_propose`
puts each task with its lane on the band, where they press **Run**, **Move** (the next lane allowed for that task) or
**Cancel**; nothing is delegated before Run, each move is logged in `.director/lane-moves.jsonl` for the next
`/director:reflect` (D3 learned the lane from such moves), and `plan_status` tells the session what they did.
`delegate` still sends one task straight to a lane; `lane_status` and `lane_fetch` follow it, and the band shows each
delegation until it is fetched. **A confidential task — flagged, or with an input under a
`confidential_paths` folder — is refused by every lane without `"confidential_ok": true`, the frontier included**:
enforced in code (`director/lanes.py`), not only as a learned statement.

**Other harnesses, next.** Claude Code is one harness; Managed Agents and the Antigravity agent are the same kind,
hosted. All three load `SKILL.md` and call MCP servers and the memory is plain files, so a managed lane is an adapter
behind the same three commands — the memory mounted read-only, lessons returned through the gate, and no cloud lane
cleared for confidential work unless you clear it (P1, P2 above).

![The chart table, a brass gate and a sealed ledger. Three glass-fronted, locked card drawers travel from it: to a writing desk close by, joined by a solid green cord, and to two distant buildings on hills joined only by dashed ochre lines. From each place a single index card returns to the brass gate.](../../docs/img/harnesses.jpg)

Not in the plugin: the targeted guard (its cases are this lab's; a project has none until it has run some),
lora-kernel's local lane and the two-lane web interface — those stay in `director.server` below; the band carries
the interface's Accept / Reject into the session. Tests: `tests/test_plugin.py`, `tests/test_ui_state.py`,
`tests/test_recipes.py`, `tests/test_lanes.py`, `tests/test_plans.py`, `tests/test_journal.py`, `tests/test_versions.py`,
`tests/test_headless.py`; the band: `claude plugin test .`.

## Without a person: `director.headless`

A benchmark run or a nightly job has no one to say yes. `director.headless` gives it the same gate with the decision
declared instead of asked, and keeps everything in one **store** directory (`<store>/.director/memory`, the journal and
the proposals), so a caller that keeps only that directory between sessions keeps the whole memory:

```bash
python -m director.headless init --memory STORE                       # an empty memory, once
claude -p "<task>" --plugin-dir packages/director ...                 # the session, with DIRECTOR_MEMORY=STORE/.director/memory
python -m director.headless outcome --memory STORE --task 03 --passed true --trace session.jsonl
python -m director.headless consolidate --memory STORE --policy verified   # or --policy all; --check to require TEACHES
```

- `outcome` writes the verifier's verdict to the journal with a digest of the session's trace: its tool calls in
  order, the tool errors it met, its final words, its turns.
- `consolidate` runs the reflection with Claude Code on the sessions since the last one — under `verified` only those
  whose verifier passed (the model never sees the others), under `all` every one — and each proposal it makes is
  accepted by `gate.accept_by_policy`: clean lint and, with `--check`, TEACHES. The journal records
  `decided_by: policy:verified` (or `all`); the last line printed carries the reflection's tokens and cost.
- The person's gate is untouched: `memory_accept` still needs the user's yes, and `accept_by_policy` is reachable from
  no MCP tool and no band (a test holds that line).

Built and tested offline with a fake Claude Code (`tests/test_headless.py`); not yet run live — DB0 runs it first by
hand on one real session.

## Running it (D1)

**Your own memory first.** `director.compile` reads only what you name — your `CLAUDE.md` / `AGENTS.md`, Claude Code's
auto-memory directory, any folder of agents or skills — compiles it with Claude Code (Opus by default; one compile
of ~60 files cost about US$5) into pages of atomic statements and an index of situations, lints them, and writes a
config if you have none:

![Long paper scrolls and notebooks feed a brass binding press; out of it come index cards, filed in a small card-catalogue drawer, a few of them linked by ink lines.](docs/img/compile.jpg)

```bash
python -m director.compile --claude-md ~/your-project/CLAUDE.md \
    --memory-dir ~/.claude/projects/<your-project>/memory --workdir ~/your-project
python -m director.server --config ~/.config/director/director.json      # http://127.0.0.1:8740
```

The config is private JSON: `memory_dir` (the compiled pages, `Index.md` among them), `workdir` (where frontier
tasks run), `log` (the event log), `local_base_url`/`local_model` (an OpenAI-compatible server), `director_model`,
`frontier_model`, `frontier_tools`, `plan_budget_usd`, `task_budget_usd`; optionally `guard` (D0 behind Accept, below),
`local_kind: "lora-kernel"` with `lora_kernel` (the local lane through lora-kernel's specialists, below) and
`agentvcs` (the executable that versions the memory).

- **Top:** the director — its capabilities (pages, statements, the situations it knows how to enter), the
  memory's current agentvcs version, its plan (each task with its lane and the statements it follows), the
  conversation and its walk through the memory.
- **The plan waits** for **Run plan**. Before that, any task can be moved to the other lane (**→ local** /
  **→ frontier**); the move is logged (`lane-moves.jsonl`) and becomes part of what the director reflects on.
- **After a task, Reflect** — with an optional correction. The director proposes memory edits; each proposal
  shows its edits and lint result, with **Accept** / **Reject**. Accept runs the accept gate (the candidate lints
  clean and a person accepted it), records the edit as an applied agentvcs patch (`<log dir>/vcs`) and makes it
  the live memory. Reject leaves the memory as it was.
- **Bottom left:** what it delegated to Claude Code, streamed (`claude -p --output-format stream-json`).
  Read-only tools (`Read, Grep, Glob`) unless the config grants more.
- **Bottom right:** what it delegated to local models. **The director never starts a local model**: running one
  on the user's machine is the user's decision. With no server answering, the planner is told the lane is
  offline, and a local task that still reaches it is logged `offline`.

[ran] 2026-10-14, one real request ("check that agentvcs's two READMEs state the same number of golden cases;
draft a changelog note"): the director walked 6 pages, planned 2 tasks citing 6 statements, put both on the
frontier because the local lane was offline, and the frontier verified 117 = 117 = the cases on disk.
Tests: `python -m pytest tests -q` (a fake `claude` and a fake local server; in CI).

**Accept is guarded by D0, targeted** (`"guard": {"enabled": true, "mode": "targeted", "k": 3, "config": "<private d0
config>"}`, [D6](docs/BRIEF-D6.md)): the guard re-plans only the D0 cases whose plans relied on a statement the edit
removes or changes in substance (read from the current memory's baseline, which records what each case's plans cite),
k = 3, and refuses if one of them was reliable and now fails. Validated [ran]: it refused both planted breaks (the
subtree rule, the mirror rule) and no unchanged memory, at US$1–4 per Accept instead of US$13. Thin margins (two breaks,
two unchanged arms), a refusal on a noisy case in the control, and an added statement is not checked by it — **the
positive probe is ([D7](docs/BRIEF-D7.md)):** the reflection writes two probes with each edit; on Accept they run on the
current memory and the candidate, and the lesson is kept only if it TEACHES (candidate ≥ 0.67, +0.33 over current) —
REDUNDANT and NO EFFECT are refused. Validated on D2's lessons (3 that taught, 1 the model knew: all four read right);
the probes are the lesson's own exam, so a wrong lesson can still pass — that stays the person's Accept. The baseline is per memory version and is carried forward on each
accepted edit. Its predecessors are on record: an absolute floor refused on noise; the paired guard over all 14 cases
was blind to a four-rule break ([D4](docs/BRIEF-D4.md), [D5](docs/BRIEF-D5.md)).

**The local lane through lora-kernel** (`"local_kind": "lora-kernel"`, `"local_base_url"` = lora-kernel's gateway,
`"lora_kernel": {"path", "user", "role", "org", "domains"}`): the planner is told which domains the specialists
cover; each local task carries a role token and opens a session of its own (a specialist's abstention was measured
as a session's first turn); the specialist's tool calls stream into the right lane; when it abstains, the task
goes to the frontier by itself. Ran live on lora-kernel's tracker member ([D4](docs/BRIEF-D4.md)): a domain read was served locally with its tools and an abstention went to the frontier; the specialist's abstention was not reliable outside the phrasing it was measured on.

Not yet: any write access by default; the accept path is tested in CI only where the agentvcs CLI is installed.

## The code, in two halves

`director/memory/` is the memory's own code: where it is (`format.find`), its pages as data, edits applied to a copy
and the lint they pass (`edit`), the gate a lesson goes through — propose, check, accept or reject — (`gate`), and
each accepted edit as an agentvcs patch (`vcs`), and the episodic journal (`journal`): one append-only
`.director/journal.jsonl` where the gate, the plans and the lanes record what happened and the session notes
corrections and outcomes (`journal_note`); `/director:reflect` reads it since its last `reflected` mark
(`journal_read`), and nothing that plans ever does. It is standard library only, never calls a model, and imports
nothing from the rest. Everything else is the agent that plans with it: `core` (the loop), `learn` (reflection),
`guard`, `recipes`, `lanes`, and the interfaces (`server`, `mcp_server`, `ui_state`, `plugin_hook`, `headless`). The MCP
tools, the band and the web interface's Accept all go through `memory.gate`; only `headless` uses its unattended
accept. `tests/test_memory_boundary.py` holds the line;
the model behind it is [`docs/EVOLUTION.md`](../../docs/EVOLUTION.md) §4.

## Privacy

The memory is the lab's own way of working and names private work. **Code here is public; the memory, the
sandboxes built from it, the plans and the transcripts live in a private directory outside this
repository** (`~/.config/director/` by default; the plugin's memory in the project's `.director/`, which
`/director:compile` keeps out of git unless you say it should be shared). The lanes config is private too. Case files use placeholders for private names, filled at
run time from the private config. What is published from a run is its scores.
