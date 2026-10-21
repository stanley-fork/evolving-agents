# D0O: does D0's gap survive when Opus plans?

**Written 2026-10-21, before any D0O plan runs.** [C2](BRIEF-C2.md) showed that the decision loop's gain was
Sonnet's: with Opus 5.5 planning, the session alone recalled the thread's decisions (0.80) and the director added
+0.05. Everything else the director publishes also planned with Sonnet — [D0](BRIEF-D0.md) above all: today's
session **0.589**, the director **0.929** on 14 real situations, rules written down before the task. D0O asks the
same question with the model this lab works with.

## Arms — D0's, byte for byte, planned by Opus

| arm | sandbox | planner |
|---|---|---|
| **AO** | D0's arm A (the workspace `CLAUDE.md`, lora-kernel's, the memory files, `MEMORY.md` in the system prompt, as frozen 2026-10-14); hash checked | **Opus 5.5** |
| **TO** | D0's arm T (the same sources compiled into pages, `DIRECTOR.md` as the system prompt); hash checked | **Opus 5.5** |

D0's 14 cases (`d0/cases.jsonl`), tools (`Read, Grep, Glob`), turns and k = 2. Per-call budget **US$2.50** (C2:
none cut at that budget).

Sequence: **AO first** (headroom), TO only if AO leaves room.

## Judges

D0's `judge.md`, blind: **Claude Opus 5.5** and **gemini-4-argon** (private terms replaced and checked before each
call). D0's verdict was Sonnet's; Sonnet is not re-run here.

## What decides it, written before it runs

| result (Opus judge) | verdict |
|---|---|
| AO ≥ 0.85 | **STOP — no headroom with Opus**: Opus follows the written rules without the director; D0's gap was Sonnet's |
| TO − AO ≥ +0.15 under **both** judges | **D0 HOLDS WITH OPUS** |
| ≥ +0.15 under one judge only | **SPLIT** |
| < +0.15 under both | **D0 DOES NOT HOLD WITH OPUS** |

Reported beside it: per-case wins, cost and turns per plan (C2 found the director cheaper with Opus).

## Spend

From C2's measured Opus costs: AO ≈ US$0.9 × 28 ≈ US$25, TO ≈ US$0.6 × 28 ≈ US$17, judges ≈ US$2. The runner stops
at the cap the person sets; nothing runs before that.

## What it does not answer

- D0's cases are the lab's own history and were written before the freeze (F1 is the future bank); a STOP here would
  be followed by F1 with Opus before concluding anything about new situations.
- Opus judging Opus plans may prefer them; Gemini is reported beside it.
