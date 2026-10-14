# director

**A meta-agent that keeps the lab's operational memory — how we work — and uses it to plan, delegate and
improve.** It walks the memory as a trajectory (page → atomic statement → link → statement) to plan a task,
delegates each step to Claude Code (frontier) or to local models, and proposes edits to the memory after
each task, gated and versioned, so the way of working is learned, recalled, refined and reproduced.

> **Status — 2026-10-14: D0 passed [ran].** On 14 real situations, a session that walks the compiled
> operational memory planned the lab's way **0.929** against today's session's **0.589** (8 : 0 cases,
> p = 0.004), at ~11× the cost per plan ([`docs/BRIEF-D0.md`](docs/BRIEF-D0.md)). Next: the attribution arm
> (is it the organisation, or just reading the memory at all?), then D1. Nothing of the loop, the interface
> or the learning is built yet.

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
| D1 | the director's loop and its interface: the director on top (capabilities, plan, conversation), below it the frontier lane (left) and the local lane (right), from one event log | only after D0 passes |
| D2 | learning: memory edits proposed after each task, gated on D0's suite, versioned in agentvcs | |
| D3 | the user moves a task between lanes; the move becomes a routing statement the director learns | |

## Privacy

The memory is the lab's own way of working and names private work. **Code here is public; the memory, the
sandboxes built from it, the plans and the transcripts live in a private directory outside this
repository** (`~/.config/director/` by default). Case files use placeholders for private names, filled at
run time from the private config. What is published from a run is its scores.
