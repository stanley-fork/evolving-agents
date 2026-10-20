# Roadmap — evolving-agents and the director

<p align="center">
  <img src="docs/img/roadmap.jpg" alt="A railway survey map: seven stations along one line across gentle hills. The first three, on the left, are fully drawn and coloured, joined by solid green track, each with a wax seal; the next four are pencil outlines joined by dashed ochre track, ending near a distant town. A surveyor's level stands where the built track ends." width="100%">
</p>

**Where it goes, and how each step will be judged.** Every step names its measurement before it is built, against
the version it replaces — never against a frontier model — and a step whose measurement comes back flat is
published as flat. Built steps link to their brief. Dates are not promised; order is.

## Built — measured [ran]

| | step | what it showed |
|---|---|---|
| ✔ | **D0 · plan by walking the memory** | 14 real situations, blind judge: today's Claude Code session 0.589, the same session told to walk its files 0.768, the director 0.929 — [brief](packages/director/docs/BRIEF-D0.md) |
| ✔ | **D1 · the loop and its interface** | the director on top, the frontier lane (Claude Code) and the local lane side by side, one event log — [README](packages/director/README.md#running-it-d1) |
| ✔ | **D2 · learn from a task without breaking what it knew** | held-out lab-specific probes 0.0 → 0.781, the first suite re-checked at 0.964, the lesson an applied agentvcs patch — [brief](packages/director/docs/BRIEF-D2.md) |
| ✔ | **D3 · learn where work runs** | after one lane move and one reflection, public-facing drafts 0 / 6 → 6 / 6 to the frontier, internal text still local — [brief](packages/director/docs/BRIEF-D3.md) |
| ✔ | **Guarded Accept** | an edit is kept only if, on D0's cases, the memory it produces does not lose to the current one (paired, above the measured spread); the verdict is part of the patch's gate. Live: a lesson taught from the interface was kept, 0.929 vs 0.893 — [D4](packages/director/docs/BRIEF-D4.md). The paired guard was blind to a planted break ([D5](packages/director/docs/BRIEF-D5.md)); **the targeted guard that replaced it refused both planted breaks and no unchanged memory, at US$1–4 per Accept — [D6](packages/director/docs/BRIEF-D6.md).** **And each lesson's own probes must show it changes the plan — [D7](packages/director/docs/BRIEF-D7.md)** (validated on D2's lessons; a wrong lesson can still pass its own exam). |
| ✔ | **P0 · inside Claude Code, as a plugin** | `/plugin install director --marketplace EvolvingAgentsLabs/evolving-agents`: a hook puts the walk into the session you already use, `/director:reflect` and an MCP server carry the gated learning. D0's 14 situations through the plugin: **0.929**, the same as the standalone director — [brief](packages/director/docs/BRIEF-P0.md) |
| ✔ | **Local planning · D8–D15** | Can a small model on the user's machine plan from the memory? Walking it alone: no — precise (~0.9) but reaching too little ([D8](packages/director/docs/BRIEF-D8.md)–[D10](packages/director/docs/BRIEF-D10.md)); its plans 0.68 against Sonnet's 0.92, and a plan composed by code did not close the gap ([D11](packages/director/docs/BRIEF-D11.md), [D12](packages/director/docs/BRIEF-D12.md)). Retrieval in code, plan by Claude: 0.837 vs 0.892 at 17× less per plan, short of its gate by 0.005 ([D13](packages/director/docs/BRIEF-D13.md), [D14](packages/director/docs/BRIEF-D14.md)). **Following meta-recipes Claude writes, picking and answering yes/no only: gemma4:12b 0.806 vs 0.854, in ~10 s and at no cost; 4B and 2B not yet ([D15](packages/director/docs/BRIEF-D15.md)).** |
| ✔ | **In the plugin: recipes, lanes, the band** | `/director:recipes` and `plan_local` bring D15 into the session; lanes send a task to a configured command lane, with confidential tasks refused by any lane not cleared for them — the frontier included — in code; a status line and a band above the prompt show what waits for the person: a lesson to accept or reject (through the same gate), a delegation to fetch — [README](packages/director/README.md#as-a-claude-code-plugin). Built and tested; not measured — nothing in it changes a plan |
| ✔ | **The order of [EVOLUTION](docs/EVOLUTION.md), steps 1 and 4–6** | the memory in its own module, kept apart from the agent by a boundary test; plans that wait on the band (Run, Move, Cancel), each move logged for the next reflection; one episodic journal the reflection reads and planning never does; recipes stamped with the memory version they came from, reported stale after an accepted lesson. Built and tested; the measurements that need the private memory, and D16 and D17, are in the [RUNBOOK](packages/director/docs/RUNBOOK.md) |


## Measured flat — not built

| | step | what it showed |
|---|---|---|
| ✘ | **D16 · the lanes as a directory of data** | the same lane facts as data cited by the statements, against prose: routed as well (0.880 vs 0.864, 15 cases) and D0 held (0.929), but 5 clearance violations vs 1 and a lane going down followed at 0.762 / 0.786 against bars of 0.8 / 0.9 — **DIRECTORY LOSES**, prose stays — [brief](packages/director/docs/BRIEF-D16.md) |

## Next — in this order

The longer view (memory, the directory, and managed-agent harnesses from Anthropic and Google), with its own order of steps: [docs/EVOLUTION.md](docs/EVOLUTION.md).

| | step | how we will know |
|---|---|---|
| 1 | **Your own director in ten minutes.** `python -m director.compile` turns your `CLAUDE.md` and Claude Code's memory into pages; `python -m director.server` opens the interface. *Ran from a clean clone [ran] 2026-10-14: agentvcs's public 36-line `AGENTS.md` → 2 pages, 17 statements, lint clean, a first plan citing 9 of them — clone 5 s, compile 44 s, plan 24 s, US$1.18 ([D4](packages/director/docs/BRIEF-D4.md)). Not yet: a stranger's own `CLAUDE.md` on their machine.* | a first user's instructions compile lint-clean and the director plans with them; time and cost from a clean clone, published |
| 2 | **The local lane through lora-kernel.** Tasks in a specialist's domain served on the user's machine; everything else abstains to the frontier. *Ran live (D4): a tracker read served locally by the specialist with its tools, an abstention detected and sent to the frontier — and the specialist's abstention unreliable outside its measured phrasing (one false, one missed). Next: one gateway session per plan for dependent tasks. Planning itself can now be local too (D15's recipes, in the plugin as `plan_local`); smaller models than the 12B are the next measurement.* | share of tasks served locally on a live session, and that no abstained task is lost |
| 3 | **Cheaper walks.** Today a plan costs about 11× today's session. An index-first walk with a budget per plan. *Two cheaper routes measured: local retrieval with Claude writing the plan, 0.837 vs 0.892 at 17× less (D14); a local model following recipes, 0.806 vs 0.854 at no cost (D15). Neither has yet been run on D0's own suite.* | D0 again: score held within 0.03 at ≤ 3× the cost |
| 4 | **A team's memory.** Lessons shared across machines (agentvcs remotes), conflicting lessons merged by Claude Code with their evidence. | two members' lessons merged without losing either's probes |
| 5 | **Held-out from the future.** Cases written from work done *after* the memory was compiled, not from the lab's own history. | D0's comparison repeated on them |

## How to help

- **Try it on your own instructions** and tell us what broke: [open an issue](https://github.com/EvolvingAgentsLabs/evolving-agents/issues/new).
- **Bring a case:** a situation where your agent ignored a rule you had written down. Cases with an action checklist
  become the next held-out suite (step 5).
- **Star the repository** if you want to follow it — it is how the lab sees which line people care about.
