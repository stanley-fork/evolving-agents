# C0: does a session already keep a multi-day thread's decisions? (headroom for phase 2)

**Written 2026-10-21, before any probe is written or any arm runs.** Step 2a of
[ROADMAP](../../../ROADMAP.md#next--in-phases)'s phase 2. Phases 1a and 1b showed the director plans by rules written
down before a task ([J0](BRIEF-J0.md), [F1](BRIEF-F1.md)). Phase 2 asks about rules **decided during** the work: over
days of sessions, does the next session still know what was decided, and why, and act on it? The director was built
for that (the journal, gated lessons). Before building or measuring anything of it, C0 asks whether there is anything
to gain: how well does Claude Code already do it, with the memory it keeps by itself?

## The thread

The lab's own work on the director from **2026-10-14 to 2026-10-21**: 95 messages from the person, across sessions in
two projects, eight days. Only the person's messages are extracted (text, dated; no tool output, no assistant turns),
to a private file. That is where the decisions are: "use gemma4:12b, not smaller models", "the UI is Claude Code",
"no more than US$50 with this key", and so on.

## The probes — written by a fresh session that never sees the memory

A fresh Claude Opus 5.5 session (`claude -p`, `Read, Grep, Glob, Write`) reads only the extracted thread and D0's and
F1's situations (to avoid them). It writes **14 candidate probes**, each resting on **one decision the person made on
a day *k***, quoted verbatim with its date, and set on a later day *k + n* (n ≥ 1):

- **mostly tasks in flight** (at least 10 of 14): a situation where that decision must change the plan; 2–3 checklist
  items about actions, as in D0;
- **at most 4 "why" probes**: the plan must state the decision's reason; items judged on meaning, not wording.

**The decision must not be derivable from the probe** (a session that could reason its way to the answer measures
reasoning, not memory; four of the lab's memory experiments came back null for that reason): a cap's number, a model's name, a choice between
options the situation leaves open. Decisions about private projects are excluded; private names are placeholders.

**Screen, in code, before any arm runs:** each quote is found verbatim in a person's message of the date it names; the
probe's day is after it; no private term; word overlap with every D0 and F1 situation below 0.5. The **first 10** that
pass are the bank (`c0/cases.jsonl`, public). Fewer than 8: the writer runs once more; still fewer, C0 stops.

## The arm — today's session, frozen

**C0 = Claude Code as it runs in this workspace today, without the director:** the workspace `CLAUDE.md`,
lora-kernel's `CLAUDE.md`, agents and skills, and the auto-memory directory with `MEMORY.md` in the system prompt —
D0's arm A, rebuilt from today's files and frozen (every file hashed) before the probes are written. It holds what the
sessions of those eight days chose to write down. Same model (Sonnet 5.5), tools (`Read, Grep, Glob`), turns, per-call
budget and k = 2 as D0.

## Judges

D0's `judge.md`, blind: Claude Sonnet 5.5 and gemini-4-argon (private terms replaced and checked before each call).

## What decides it, written before it runs

| C0 under Sonnet | verdict |
|---|---|
| **≥ 0.85** | **STOP — no headroom.** The memory Claude Code keeps by itself already carries the thread's decisions; phase 2b is not built, and that is published |
| **< 0.85** | **GO to 2b**: a brief for the director on the same probes, with a floor arm (no auto-memory) to show the memory, not reasoning, carries the score |

## Spend

About US$6 (probes ~US$4, the arm ~US$1.5, judges ~US$0.5); the runner stops at **US$10**. Gemini inside the key's
US$50 ledger.

## What it does not answer

- It tests a **fresh session on a later day**, not one long session after compaction: the common case here, since
  work resumes in new sessions, but not the only one.
- The auto-memory in C0 was written by the same sessions, with care: this lab writes its decisions down. A team that
  does not would score lower; C0 measures this workspace, not Claude Code in general.
- Ten probes on one thread is small; a STOP or GO near 0.85 is reported with its per-case spread.
