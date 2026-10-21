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
go through the memory's own code: applied as one set and kept unless they add a lint problem (the gate's first
check). **Corrected before any arm ran (2026-10-21):** the first version of the runner kept edits one at a time, so a new
statement linking to another new one was refused (5 of 10, all about the director); the gate applies a proposal's
edits as one set, and so does the runner now, dropping only edits whose problems remain. Two departures from the
plugin, said here: **no person accepts** (every lint-clean edit is kept, and the
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

## Result [ran] 2026-10-21 — **THE LOOP WORKS** (attribution pending)

**The reflection.** One unattended Opus session over the 95 messages (US$1.47, 15 turns) proposed 10 edits — 6
modifications and 4 new statements; gated as one set, all 10 pass lint. Its first gating (one at a time) refused 5
of them; that was corrected before any arm ran (above).

| | Sonnet 5.5 | gemini-4-argon |
|---|---|---|
| C0 — today's session with its auto-memory | 0.275 | 0.450 |
| **C1 — the same, plus the director after one reflection** | **0.725** | **0.825** |
| C1 − C0 | **+0.45** | **+0.375** |

- **Both above +0.15: THE LOOP WORKS.** Per probe (Sonnet) C1 8 wins, C0 1 (c10), 1 tie (c01). Without c01: 0.75
  against 0.25.
- **Where it moved:** c02 0 → 0.75, c03 0.33 → 1, c04 0 → 1, c06 0 → 0.67, c07 0.5 → 1, c08 0 → 1, c05 0.75 → 1.
  On c08 the plans quote the person's cap with its date — the statement the reflection wrote; C0's plans, with the
  same cap in their auto-memory, set their own.
- **Where it did not:** c10 (feature the repository's star count on the site) 0.5 → 0, and c09 0.17 → 0.33 — the
  reflection's statement about the README did not carry the details those probes check.
- Item agreement between judges 0.90 (κ 0.72). Every C1 plan cites director statements (19/19).
- **Cost:** US$0.51 per plan against C0's US$0.15. **One sample missing:** c07 s1 hit the per-call budget twice
  (first pass and its one retry) and is reported as an error; c07 is scored on one sample. Spent US$12.83 of the
  US$20 cap; Gemini US$0.10.

**What is not yet shown.** The brief's attribution arms come next: the director **without** the reflection (does the
walk alone, which also makes the session read more, carry part of the gain?) and a floor with no memory. They need
about US$12 more, past this brief's cap; they wait for the person's decision.

## Attribution arms — written before they run (2026-10-21)

The person raised C1's cap from US$20 to **US$35** on 2026-10-21 for these two arms. Same probes, model, tools, turns,
per-call budget, k and judges.

| arm | what the session gets |
|---|---|
| **W — the walk without the reflection** | C0's sandbox + the director as the hook puts it in, over the live memory **before** the reflection (no lessons from the thread) |
| **F — floor** | nothing: an empty directory and no system prompt beyond Claude Code's own |

How they will be read:

| result (Sonnet) | reading |
|---|---|
| **C1 − W ≥ +0.15** | the **lessons** carry the gain: the reflection put into the memory what the walk then found |
| C1 − W < +0.15 | the **walk** carries most of it: asking the session to walk a memory makes it read more, and it finds the decisions in its own notes — the same split D0's A+ measured for rules |
| F ≥ C0 − 0.05 | the auto-memory adds nothing on these probes: the decisions were not recoverable from it in practice |

W and F run in that order; F is cheap and runs either way.

**A third judge (2026-10-21, at the person's request, before W's and F's scores were seen):** Claude Opus 5.5 judges
every arm — C0, C1, W, F — with the same `judge.md`, blind. It is reported beside Sonnet and Gemini; the verdicts
written above stay on the judges they name.
