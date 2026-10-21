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

## Result [ran] 2026-10-21 — **GO to 2b**

**The bank.** 14 candidates by a fresh Opus that never saw the memory (decisions from 2026-10-14 to 2026-10-21); all
passed the code screen; the first 10 are [`c0/cases.jsonl`](../c0/cases.jsonl), committed before the arm ran. The arm
was frozen (58 files, hashed) before the probes were written.

| | Sonnet 5.5 | gemini-4-argon |
|---|---|---|
| **C0 — today's session, with its own auto-memory** | **0.275** | 0.450 |
| C0 without c01 (see below) | 0.250 | — |

- **0.275, far below 0.85: there is room. GO to 2b.** Per probe (Sonnet): c05 0.75; c01, c07, c10 0.5; c03 0.33;
  c09 0.17; c02, c04, c06, c08 0.0. Item agreement between judges 0.80 (κ 0.57); Gemini more lenient again.
- **The decisions were written down; they were not recalled.** Every decision has a matching line in the frozen
  memory (a loose search), and the four checked exactly are there verbatim. On c08 the memory holds the person's cap
  of US$50 for the key; both plans set their own cap instead (one "≈ US$75"). The failure is not storage, it is
  bringing the right note to the plan — the failure D0 measured for rules, now for decisions made along the way.
- **Two threats, said before reading further.** (1) c01 is set on 2026-10-17 but the frozen memory already holds a
  later decision (2026-10-21) that contradicts its checklist; it was kept (removing it after seeing the bank would be a
  redesign) and the score is given without it too. (2) The frozen `CLAUDE.md` now tells the session to walk the
  director's memory, which is not in the sandbox; 2 of 20 plans mention it as missing, and the arm read a little more (2.4 turns
  on average, US$0.15 per plan) than D0's arm A did (1.5, US$0.06). Neither moves the verdict: without c01 the score is 0.25.
- 20 plans, 0 errors, 0 unparsed verdicts, 0 private terms or keys sent. **Spent US$4.71** (probes 1.43, plans 3.06,
  judge 0.23) and Gemini US$0.09.

**What 2b has to show.** The director's memory, as it stands, does not hold these decisions either: it was compiled on
2026-10-14 and has learned two lessons since. So 2b is not "walk the memory you have": it is the loop the director was
built for — corrections and decisions noted in the journal as they happen, proposed as lessons by `/director:reflect`,
kept through the gate — measured on these probes against C0, with a floor arm (no memory at all) to show the memory,
not reasoning, carries the score. Its brief comes first.
