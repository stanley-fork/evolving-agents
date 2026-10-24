# Roadmap — evolving-agents and the director

<p align="center">
  <img src="docs/img/roadmap.jpg" alt="A railway survey map: seven stations along one line across gentle hills. The first three, on the left, are fully drawn and coloured, joined by solid green track, each with a wax seal; the next four are pencil outlines joined by dashed ochre track, ending near a distant town. A surveyor's level stands where the built track ends." width="100%">
</p>

**In plain words.** Built: the director plans from your team's memory inside Claude Code, learns lessons only with
your yes, keeps confidential work on machines you clear, and learns where you want each kind of task to run.
Measured honestly: with the strongest models it no longer adds general good practice — its value is your own
decisions, privacy and an auditable record. Next: whether an agent improves from one session to the next on
recurring work (director-bench), whether the director can improve its own components, and the same memory on other
agent platforms. The detail of every step follows.

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
| ✔ | **The director with no person present** | `python -m director.headless`: a verifier's verdicts in the journal with a digest of each session; lessons consolidated by Claude Code and accepted by a declared policy (`verified`: only from sessions whose verifier passed; `all`: the ablation), `decided_by` recorded; the person's gate untouched. Built for director-bench; tested offline, not yet run live — [README](packages/director/README.md#without-a-person-directorheadless) |
| ✔ | **The order of [EVOLUTION](docs/EVOLUTION.md), steps 1 and 4–6** | the memory in its own module, kept apart from the agent by a boundary test; plans that wait on the band (Run, Move, Cancel), each move logged for the next reflection; one episodic journal the reflection reads and planning never does; recipes stamped with the memory version they came from, reported stale after an accepted lesson. Built and tested; the measurements that need the private memory, and D16 and D17, are in the [RUNBOOK](packages/director/docs/RUNBOOK.md) |

## Measured flat — not built

| | step | what it showed |
|---|---|---|
| ✘ | **D16 · the lanes as a directory of data** | the same lane facts as data cited by the statements, against prose: routed as well (0.880 vs 0.864, 15 cases) and D0 held (0.929), but 5 clearance violations vs 1 and a lane going down followed at 0.762 / 0.786 against bars of 0.8 / 0.9 — **DIRECTORY LOSES**, prose stays — [brief](packages/director/docs/BRIEF-D16.md) |

## Next — in phases

**Revised 2026-10-22:** phase 2′ (director-bench) added before phase 3; with Opus planning, phases 1 and 2 moved the
director's case from one plan to many sessions. **Revised 2026-10-20.** Everything published so far rests on one number, D0's 0.929, measured on cases from the
lab's own history and judged by the same provider that planned them. Before building on it, the next phase asks
whether it holds outside those two conditions. The phases are ordered so that the cheapest measurement that could
contradict us runs first; a phase that comes back flat stops the ones after it. The longer view (memory, the
directory, managed-agent harnesses) and its own order of steps: [docs/EVOLUTION.md](docs/EVOLUTION.md).

### Phase 1 — does 0.929 hold outside our own history?

| | step | how we will know |
|---|---|---|
| 1a | **A judge from another provider.** D0's plans are already on disk; re-judge them with a non-Claude model, nothing re-planned. Planner and judge are both Claude today, so part of the gap could be Claude preferring Claude's plans. About US$2–5. | the gap between today's session and the director stays ≥ +0.15 under the other judge; below that, self-preference is part of the effect, and that is published — **[ran] 2026-10-20, HOLDS:** gemini-4-argon gives +0.304 (Sonnet +0.339), the same order A < A+ < P ≤ T, US$0.51 ([J0](packages/director/docs/BRIEF-J0.md)) |
| 1b | **Held-out from the future.** Cases written from work done *after* the memory was compiled, not from the lab's own history; a new bank, since D0's, D11's and D12's are spent. About US$30, two arms. | D0's comparison repeated on them: the director ≥ +0.15 over today's session — **[ran] 2026-10-21, HOLDS ON THE FUTURE:** on 14 situations from after the freeze, today's session 0.363 and the director 0.839 (Sonnet; Gemini 0.512 / 0.798), +0.48 and +0.29, US$20 ([F1](packages/director/docs/BRIEF-F1.md)) |
| 1b′ | **The same with Opus planning.** | **[ran] 2026-10-21:** the session alone 0.78 (Opus judge) / 0.857 (Gemini) against Sonnet's session at 0.363 (Sonnet judge); room under Opus only, so the director's arm could at best split the judges ([F1O](packages/director/docs/BRIEF-F1O.md)) |
| 1c | **Your own director in ten minutes.** `python -m director.compile` turns your `CLAUDE.md` and Claude Code's memory into pages; `/plugin install director` or `python -m director.server` runs it. *Ran from a clean clone [ran] 2026-10-14: agentvcs's public 36-line `AGENTS.md` → 2 pages, 17 statements, lint clean, a first plan citing 9 of them — clone 5 s, compile 44 s, plan 24 s, US$1.18 ([D4](packages/director/docs/BRIEF-D4.md)). Not yet: someone else's own `CLAUDE.md` on their machine.* | a first outside user's instructions compile lint-clean and the director plans with them; time, cost and what broke, published |

If 1a or 1b fails, phases 2–4 wait and the failure is the next thing written up.

### Phase 2 — coherence over days, not minutes

Rule recall inside one task is a problem today's session already solves 0.589 of the time; models with native memory
may close it on their own. What no harness does well yet is staying coherent across days of work: why a hypothesis
was dropped on Tuesday, whether a rule decided on day *k* still holds on day *k + n*, after compactions and new
sessions. That is the problem the episodic journal and the gated memory were built for, and it is not measured.

| | step | how we will know |
|---|---|---|
| 2a | **Headroom first.** One real multi-day thread of the lab's own work, its decisions dated in PRs and briefs; about ten probes ("why was X dropped on day *k*?", "is the rule decided on day *k* respected on day *k + n*?"), answered by Claude Code with its own auto-memory and compaction. One arm, about US$5–10. | if that baseline scores ≥ 0.85 there is no room, and phase 2 stops here without building anything — **[ran] 2026-10-21, GO:** on 10 probes from the person's own decisions over eight days, today's session with its auto-memory scored 0.275 (Gemini 0.45); the decisions were written in its memory and not recalled. US$4.80 ([C0](packages/director/docs/BRIEF-C0.md)) |
| 2b | **The director on the same thread.** Only if 2a leaves room: the same probes with the director's journal and gated memory, judged blind. Brief first. | ≥ +0.15 over 2a's baseline — **[ran] 2026-10-21, THE LOOP WORKS:** after one reflection over the thread, 0.725 against 0.275 (Gemini 0.825 / 0.45); the lessons carry it: the walk without them 0.23, C1 − W +0.46 ([C1](packages/director/docs/BRIEF-C1.md)) |
| 2c | **The same with Opus planning** (the person's model since 2026-10-21). | **[ran] 2026-10-21, DOES NOT ADD WITH OPUS:** Opus alone 0.80, with the director 0.85 (+0.05; Gemini +0.025), at 34 % less per plan ([C2](packages/director/docs/BRIEF-C2.md)). Next: D0's rules with an Opus planner, headroom first |
| 2d | **D0 with Opus planning.** | **[ran] 2026-10-21, STOP:** Opus alone follows D0's rules 0.964 (Gemini 0.946); D0's gap was Sonnet's ([D0O](packages/director/docs/BRIEF-D0O.md)). Next: F1 with Opus |

### Phase 2′ — between sessions, against Claude Code's own memory (director-bench)

**Added 2026-10-22.** Phases 1 and 2 left one claim standing with Opus: what lasts beyond a session, and its cost.
[director-bench](https://github.com/EvolvingAgentsLabs/director-bench) measures it in the open, independent of the
director: chained headless sessions on recurring tasks, a verifier per task, only the memory store kept between them.

| | step | how we will know |
|---|---|---|
| 2′a | **DB0 · the pilot.** One synthetic family, no memory / native memory / the director, 3 seeds, 90 sessions, cap US$150. | no-memory must repeat its own errors (RER ≥ 0.15) or there is nothing to measure, and the family is replaced; the seed spread sizes the full run — [brief](packages/director/docs/BRIEF-DB0.md) |
| 2′b | **The full v0.** 4 families × 10 tasks × 5 configurations (no memory, native, director, both, director without verifier gating) × 5 seeds; one family chosen or reviewed by someone outside the lab, all frozen before the director is tuned. | the bench's `DECISION.md`, fixed before it runs: repeated errors −30 % against native memory (95 % CI excluding 0), success ≥ −2 pp, regressions ≤ +1 pp, memory-induced errors ≤ native, tokens ≤ +10 %, and the ablation worse. Go, iterate one cycle on the failing condition, or publish the bench itself as the contribution |

### Phase 3 — evolve components, not only the memory

| | step | how we will know |
|---|---|---|
| 3 | **D17 · component evolution.** The meta-recipes a local model follows change through the same gate as a lesson — the promise of the name "Evolving Agents" the director has not yet kept. After phase 1, because its guard uses the same judge. | [BRIEF-D17](packages/director/docs/BRIEF-D17.md): an evolved recipe set beats its parent on held-out cases and the parent's own bank holds |

### Phase 4 — the same memory on other harnesses

| | step | how we will know |
|---|---|---|
| 4a | **An `anthropic-managed` lane** (P1): Managed Agents behind `submit / status / fetch`, the memory mounted read-only, the walk as a skill, the gate over MCP. Last on purpose: porting a number that has not been checked outside our history only spreads it. | D0 through it within 0.03 of 0.929; cost per plan published |
| 4b | **An `antigravity` lane** (P2), then **one memory package** for every harness | the same bar; P0, P1 and P2 re-run from one artifact ([EVOLUTION §8](docs/EVOLUTION.md#8-the-order), steps 7–9) |

### Alongside the phases

| | step | how we will know |
|---|---|---|
| — | **The local lane through lora-kernel.** Tasks in a specialist's domain served on the user's machine; everything else abstains to the frontier. *Ran live (D4): a tracker read served locally by the specialist with its tools, an abstention detected and sent to the frontier — and the specialist's abstention unreliable outside its measured phrasing (one false, one missed). Next: one gateway session per plan for dependent tasks. Planning itself can now be local too (D15's recipes, in the plugin as `plan_local`, with gemma4:12b).* | share of tasks served locally on a live session, and that no abstained task is lost |
| — | **Cheaper walks.** Today a plan costs about 11× today's session. An index-first walk with a budget per plan. *Two cheaper routes measured: local retrieval with Claude writing the plan, 0.837 vs 0.892 at 17× less (D14); a local model following recipes, 0.806 vs 0.854 at no cost (D15). Neither has yet been run on D0's own suite.* | D0 again: score held within 0.03 at ≤ 3× the cost |
| — | **A team's memory.** Lessons shared across machines (agentvcs remotes), conflicting lessons merged by Claude Code with their evidence. *After phase 1: sharing lessons that have not been checked outside one history multiplies the error.* | two members' lessons merged without losing either's probes |

### Not pursued, and why

- **Distilling the memory into a model's weights.** Measured in lora-kernel and the runtime before it: teaching a
  model the facts did not beat keeping them in notes it reads, and an edited note is answered without retraining
  (17/17). Weights hold how to navigate; notes hold what is true.
- **Merging two forked Agent SDK sessions (the paused plugin's M1).** Merging lives in
  [agentvcs](https://github.com/EvolvingAgentsLabs/agentvcs) now, where Claude Code resolves conflicts with their
  evidence.
- **A separate installer.** `/plugin install director` and `/director:compile` are the two-minute path; phase 1c
  measures whether they are enough.

## How to help

- **Try it on your own instructions** and tell us what broke: [open an issue](https://github.com/EvolvingAgentsLabs/evolving-agents/issues/new).
- **Bring a case:** a situation where your agent ignored a rule you had written down. Cases with an action checklist
  become the next held-out suite (phase 1b).
- **Star the repository** if you want to follow it — it is how the lab sees which line people care about.
