# director

**A meta-agent that keeps the lab's operational memory — how we work — and uses it to plan, delegate and
improve.** It walks the memory as a trajectory (page → atomic statement → link → statement) to plan a task,
delegates each step to Claude Code (frontier) or to local models, and proposes edits to the memory after
each task, gated and versioned, so the way of working is learned, recalled, refined and reproduced.

> **Status — 2026-10-14: D0 passed, D1 built, D2 learns [ran].** On 14 real situations a session that walks the
> compiled operational memory planned the lab's way **0.929**, against **0.589** for today's session and
> **0.768** for today's session told to walk its own files ([`docs/BRIEF-D0.md`](docs/BRIEF-D0.md)). D2: after
> reflecting on five real incidents, the director's own edits took held-out probes of lab-specific lessons from
> **0.0 to 0.781** without breaking D0 (0.964), and the edit is a gated, applied patch in agentvcs that `blame`
> credits with the gain ([`docs/BRIEF-D2.md`](docs/BRIEF-D2.md)). D1 — the loop and the interface — runs
> locally (below). Not built: reflection inside the interface, moving tasks between lanes (D3).

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
| D3 | the user moves a task between lanes; the move becomes a routing statement the director learns | |

## Running it (D1)

```bash
python -m director.server --config ~/.config/director/director.json      # http://127.0.0.1:8740
```

The config is private JSON: `memory_dir` (the compiled pages, `Index.md` among them), `workdir` (where frontier
tasks run), `log` (the event log), `local_base_url`/`local_model` (an OpenAI-compatible server), `director_model`,
`frontier_model`, `frontier_tools`, `plan_budget_usd`, `task_budget_usd`.

- **Top:** the director — its capabilities (pages, statements, the situations it knows how to enter), its plan
  (each task with its lane and the statements it follows), the conversation and its walk through the memory.
- **Bottom left:** what it delegated to Claude Code, streamed (`claude -p --output-format stream-json`).
  Read-only tools (`Read, Grep, Glob`) unless the config grants more.
- **Bottom right:** what it delegated to local models. **The director never starts a local model**: running one
  on the user's machine is the user's decision. With no server answering, the planner is told the lane is
  offline, and a local task that still reaches it is logged `offline`.

[ran] 2026-10-14, one real request ("check that agentvcs's two READMEs state the same number of golden cases;
draft a changelog note"): the director walked 6 pages, planned 2 tasks citing 6 statements, put both on the
frontier because the local lane was offline, and the frontier verified 117 = 117 = the cases on disk.
Tests: `python -m pytest tests -q` (a fake `claude` and a fake local server; in CI).

Not yet: moving a task between lanes (D3), memory edits after a task (D2), any write access by default.

## Privacy

The memory is the lab's own way of working and names private work. **Code here is public; the memory, the
sandboxes built from it, the plans and the transcripts live in a private directory outside this
repository** (`~/.config/director/` by default). Case files use placeholders for private names, filled at
run time from the private config. What is published from a run is its scores.
