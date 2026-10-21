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

## Result [ran] 2026-10-21 — **THE LOOP DOES NOT ADD WITH OPUS**

| arm | Opus 5.5 judge | Gemini judge | cost / plan | turns |
|---|---|---|---|---|
| O0 — Opus, today's session with its auto-memory | **0.80** | 0.84 | US$0.88 | — |
| O1 — Opus + the director after the reflection | **0.85** | 0.87 | **US$0.58** | — |
| O1 − O0 | **+0.05** | **+0.025** | **−34 %** | |

- **O0 = 0.80, below 0.85: there was room, and O1 ran — but it moved +0.05 (Opus) and +0.025 (Gemini), below +0.15
  under both. The verdict is THE LOOP DOES NOT ADD WITH OPUS.**
- **The model is the difference.** With Sonnet planning, the same session scored 0.275 and the director 0.725 (C1);
  with Opus, the session alone already scores 0.80 — Opus finds and uses the decisions in its own auto-memory, which
  Sonnet did not. C1's gain is real for Sonnet and mostly absorbed by Opus.
- **Per probe (Opus judge):** O1 better on c03 (0.67 → 1), c06 (0.83 → 1), c09 (0.5 → 1); worse on c01 (0.5 → 0) —
  the probe flagged in C0, whose checklist the director's memory contradicts with a later decision; equal on the rest.
  Without c01: 0.83 → 0.94, +0.11, still below the bar.
- **What the director did buy with Opus: cost.** O1's plans cost 34 % less (US$0.58 against 0.88): the walk goes to
  the statement instead of reading notes until the decision turns up.
- 40 plans, 0 errors (the per-call budget of US$2.50 cut none), 0 unparsed verdicts. **Spent US$30.49 of the US$50
  cap**; Gemini US$0.20.

**What this changes.** D0, J0, F1, C0 and C1 all planned with Sonnet. The same question now stands for rules written
down before a task: does D0's gap (0.589 → 0.929) survive an Opus planner, or did Sonnet's weaker recall carry it too?
That is the next cheapest falsifier — D0's or F1's arm A planned by Opus, headroom first.
