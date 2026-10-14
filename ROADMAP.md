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
| ✔ | **Guarded Accept** | an edit is kept only if the memory it produces still holds D0 (≥ 0.86), and that is part of the patch's gate |

## Next — in this order

| | step | how we will know |
|---|---|---|
| 1 | **Your own director in ten minutes.** `python -m director.compile` turns your `CLAUDE.md` and Claude Code's memory into pages; `python -m director.server` opens the interface. *(The command is built; the ten minutes are not yet timed on a stranger's machine.)* | a first user's instructions compile lint-clean and the director plans with them; time and cost from a clean clone, published |
| 2 | **The local lane through lora-kernel.** Tasks in a specialist's domain served on the user's machine; everything else abstains to the frontier. *(Code and tests in; the live run is next.)* | share of tasks served locally on a live session, and that no abstained task is lost |
| 3 | **Cheaper walks.** Today a plan costs about 11× today's session. An index-first walk with a budget per plan. | D0 again: score held within 0.03 at ≤ 3× the cost |
| 4 | **Inside Claude Code, no extra window.** The director as a Claude Code agent and tools, so the walk happens in the session you already use. | D0's cases run inside Claude Code: the same score as the standalone director |
| 5 | **A team's memory.** Lessons shared across machines (agentvcs remotes), conflicting lessons merged by Claude Code with their evidence. | two members' lessons merged without losing either's probes |
| 6 | **Held-out from the future.** Cases written from work done *after* the memory was compiled, not from the lab's own history. | D0's comparison repeated on them |

## How to help

- **Try it on your own instructions** and tell us what broke: [open an issue](https://github.com/EvolvingAgentsLabs/evolving-agents/issues/new).
- **Bring a case:** a situation where your agent ignored a rule you had written down. Cases with an action checklist
  become the next held-out suite (step 6).
- **Star the repository** if you want to follow it — it is how the lab sees which line people care about.
