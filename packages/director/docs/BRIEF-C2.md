# C2: does the decision loop still help when Opus plans?

**Written 2026-10-21, before any C2 plan runs.** The person decided the same day that Claude Opus 5.5 plans, designs
and judges from now on. [C0](BRIEF-C0.md) and [C1](BRIEF-C1.md) planned with Sonnet 5.5. A stronger planner may
recall the thread's decisions from its own notes without the director; if it does, the loop buys nothing with the
model this lab actually works with. So C2 repeats C0 and C1 with Opus planning — control first.

## Arms — C0's and C1's, byte for byte, planned by Opus

| arm | sandbox | planner |
|---|---|---|
| **O0** | C0's frozen sandbox (workspace `CLAUDE.md`, auto-memory, `MEMORY.md` in the system prompt); hash checked against C0's freeze | **Opus 5.5** |
| **O1** | C1's sandbox: O0 plus the director after the same reflection (memory hash checked against C1's), the walk appended as the hook does | **Opus 5.5** |

Same probes (C0's 10, `c0/cases.jsonl`), tools (`Read, Grep, Glob`), turns and k = 2. The per-call budget rises from
US$0.60 to **US$2.50**: an Opus walk costs more, and a budget cut is an instrument error, not a plan.

Sequence, not a grid: **O0 first** (headroom), O1 only if O0 leaves room.

## Judges

D0's `judge.md`, blind: **Claude Opus 5.5** (the person's choice) and **gemini-4-argon** (another provider, since
Opus now both plans and judges; private terms replaced and checked before each call).

## What decides it, written before it runs

| result (Opus judge) | verdict |
|---|---|
| O0 ≥ 0.85 | **STOP — no headroom with Opus**: Opus alone keeps the thread's decisions; the loop's C1 gain does not transfer |
| O1 − O0 ≥ +0.15 under **both** judges | **THE LOOP HOLDS WITH OPUS** |
| ≥ +0.15 under one judge only | **SPLIT** |
| < +0.15 under both | **THE LOOP DOES NOT ADD WITH OPUS** |

## Spend

Unknown before a smoke test: **one plan per arm** (c01, sample 0; kept as part of the arm) gives the cost per plan,
and the full cap is set by the person from it. Gemini inside the key's US$50 ledger.

## What it does not answer

- The probes were already used with Sonnet; Opus has not seen them, but they are no longer new to the lab.
- Opus judging Opus plans may prefer them; Gemini is there for that, and both are reported.
