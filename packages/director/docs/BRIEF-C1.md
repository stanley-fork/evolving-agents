# C1: does the director's decision loop bring a thread's decisions back to the plan?

**Written 2026-10-21, before the reflection or any arm runs.** Step 2b of
[ROADMAP](../../../ROADMAP.md#next--in-phases)'s phase 2, after [C0](BRIEF-C0.md) left room: today's session, with the
memory Claude Code keeps by itself, planned by the person's own decisions **0.275** of the time on 10 probes, although
every decision was written in that memory. Director 0.2.0 adds the loop C0 showed missing: every message the person
sends reaches the journal, and `/director:reflect` turns decisions (not only corrections) into statements of the
director's memory, which the walk then reaches. C1 measures that loop on C0's probes.

## The arm — C0 plus the director, after one reflection

**C1 = C0's frozen sandbox, byte for byte** (the workspace `CLAUDE.md`, lora-kernel's, the auto-memory, `MEMORY.md`
in the system prompt; hash checked against C0's freeze) **plus the director** as the plugin puts it into a session:
`.director/memory/` with the lab's live memory (compiled 2026-10-14, two lessons since; copied and hashed) **after one
reflection**, and the walk (`DIRECTOR.md`, pointed at `.director/memory/`) appended to the system prompt as the
SessionStart hook does. Same model (Sonnet 5.5), tools, turns, per-call budget and k = 2.

**The reflection.** The journal it reads is the thread C0's probes were written from: the person's 95 messages from
2026-10-14 to 2026-10-21, as `person` entries (keys scrubbed) — what the capture hook would have recorded. One
reflection over all of it, the conservative case (eight days without one). It runs unattended: a fresh Claude Opus 5.5
session (`claude -p`, `Read, Grep, Glob`) in a directory holding only a copy of the memory and the journal, with
`/director:reflect`'s steps 1–2 for a batch ([`c1/reflect.md`](../c1/reflect.md)); it never sees the probes. Its edits
go through the memory's own code: each edit is applied in order and kept only if it adds no lint problem (the gate's
first check). Two departures from the plugin, said here: **no person accepts** (every lint-clean edit is kept, and the
count is published) and **no positive probe** (TEACHES) runs.

## Judges

D0's `judge.md`, blind: Claude Sonnet 5.5 and gemini-4-argon (private terms replaced and checked before each call).

## What decides it, written before it runs

| C1 − C0 | verdict |
|---|---|
| **≥ +0.15 under both judges** | **THE LOOP WORKS** — then two attribution arms, in this order: the director without the reflection (the walk alone), and a floor with no memory at all |
| ≥ +0.15 under one judge only | **SPLIT**, published with both numbers |
| **< +0.15 under both** | **THE LOOP DOES NOT WORK YET** — read where it fails before changing anything: a decision not proposed by the reflection, proposed but not reached by the walk, or reached and not applied |

Reported beside it: how many edits the reflection proposed and kept, which of the 10 decisions became statements,
per-probe scores, and cost per plan.

## Spend

About US$15 (reflection ~US$4, the arm ~US$10, judges ~US$1); the runner stops at **US$20**. Gemini inside the key's
US$50 ledger.

## What it does not answer

- **Information parity is by construction, not by luck.** The reflection reads the same messages the probe writer
  read; C0's memory was written from them too, by the sessions of those days. C1 asks whether the director's way of
  turning them into memory — and walking it — recalls them better; it does not test decisions nobody wrote down.
- No person accepted the lessons; a person who rejects some would keep fewer.
- c01's threat from C0 holds here too (the memory knows a later decision); C1 is reported with and without it.
