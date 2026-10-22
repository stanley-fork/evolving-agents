# DB0: the director-bench pilot — is there a learning curve to measure, and how noisy is it?

**Written 2026-10-22, before the bench has run a real session.** The lab's results so far measure one planning call
against a memory ([D0](BRIEF-D0.md), [C1](BRIEF-C1.md), [C2](BRIEF-C2.md)). With Opus planning, most of the
director's edge inside one session is gone (D0O: Opus alone 0.964); what is left is what lasts **between** sessions
and what it costs. **director-bench** (its own repository, `EvolvingAgentsLabs/director-bench`, independent of the
director) measures exactly that: chained headless Claude Code sessions on recurring tasks, a binary verifier per task,
the memory store the only thing kept between sessions. Its full v0 is 1,000 sessions. DB0 is the pilot that decides
whether the full run is worth its cost.

## What runs

- **One family**: the bench's synthetic repository (`ledgerkit`, house conventions an agent without memory tends to
  miss: integer cents, a registry, the project's own check command, numbered migrations), 10 tasks plus a retention
  task (task 1 rewritten) and a trap task (a convention that changed). Frozen in the bench's repository before any
  arm runs.
- **Three configurations**, same model string, same Claude Code version, same permissions:
  - **C0** — Claude Code, no memory (the floor);
  - **C1** — Claude Code with its native memory (Auto Memory and its consolidation), isolated by `HOME`;
  - **C2** — Claude Code with the director (`--plugin-dir`, native memory off), consolidating every 2 tasks with
    `python -m director.headless consolidate --policy verified`: lessons drawn only from sessions whose verifier
    passed, accepted by that policy with no person present (`gate.accept_by_policy`; the journal records
    `decided_by: policy:verified`).
- **3 seeds** each (task order permuted by seed, the same across configurations): **90 sessions** plus consolidations.
- Budget per session: 60 turns or 400k tokens, as the bench's spec.

**Model: decided before it runs, recorded as the exact API string.** The brief proposes **Opus 5.5**: it is where the
lab's results leave the least room, so a curve there is the claim that matters; Sonnet is the cheaper rerun.

## What decides it, written before it runs

| Pilot result | verdict |
|---|---|
| C0 repeats few errors: mean repeated-error rate (RER) < 0.15, or fewer than 3 repeated failures per run | **STOP — no headroom.** A session without memory already avoids repeating its own mistakes on this family; the family is replaced (harder conventions) before anything else runs, and that is published |
| C0 has headroom, but the seed-to-seed spread of RER is so wide that 5 seeds could not show a 30 % reduction (bootstrap half-width > 0.3 × C0's RER) | **WIDEN** — the full run's seed count is raised before it is budgeted |
| Otherwise | **GO** — the full v0 (4 families × 10 tasks × 5 configurations × 5 seeds) is budgeted with the pilot's measured cost per session |

The pilot reports C1 and C2 against C0 and against each other, but **decides nothing about the director**: three seeds
on one family cannot carry the bench's win criteria, which stay as its `DECISION.md` fixed them (RER −30 % with a
95 % CI excluding 0, success ≥ −2 pp, regressions ≤ +1 pp, memory-induced errors ≤ C1, tokens ≤ +10 %, and A1 worse
than C2).

## Before it can run

1. **C1's native memory controls** confirmed on the pinned Claude Code version: how Auto Memory is switched on and
   off, and how its consolidation is triggered in a headless run. The bench refuses a configuration with unresolved
   `TODO(confirm)` fields. If consolidation cannot be triggered on demand, C1 runs with what the version does by
   itself, and the report says so.
2. **The family frozen** in the bench's repository (every verifier fails on its snapshot and passes on its reference
   solution; tested), and the leak audit clean: no verifier or reference text in any memory store.
3. **C2's consolidation checked once by hand** on one real session: `outcome`, then `consolidate`, the proposal's diff
   read by a person. Built and tested offline (`tests/test_headless.py`, a fake Claude Code), **not yet run live**.

## Spend

Estimated US$60–150 (90 sessions at roughly US$0.5–1.5 each with Opus on a small repository, plus 45 consolidations);
the runner stops at **US$150**. The estimate is a guess until the first 10 sessions report their cost; the pilot's
real cost per session is one of its results.

## What it does not answer

- Whether the director beats native memory: that is the full run's question, with its pre-registered thresholds.
- Anything about teams, large repositories or one-off tasks (the bench's declared limits).
- Whether a person's yes matters: C2 removes the person from the gate. The product keeps it; the bench measures the
  verifier as the gate instead, and says so.

## First live smoke run [ran] 2026-10-22 — not the pilot; the family has no headroom

Three tasks of `ledgerkit` (01, 02, 05), seed 1, Sonnet 5.5, Claude Code 2.1.308, C0 and C2. A smoke run, not a
measurement: it checks the machinery and the family, and decides nothing about the director.

| | passed | turns per task | tool errors | cost per task |
|---|---|---|---|---|
| C0 — no memory | 3 / 3 | 4, 4, 4 | 0 | US$0.058 |
| C2 — the director (one consolidation after task 2) | 3 / 3 | 5, 5, 5 | 0 | US$0.098 (sessions US$0.063 + the consolidation spread over the 2 tasks it covers) |

- **The family has no headroom.** C0 solved every task first time in four turns with no tool error: the repository is
  small enough to read whole in one turn, and its README names the check command. Nothing repeats, so there is no
  error rate to reduce — the brief's STOP case, met before the pilot spent anything. The family is redesigned before
  the pilot runs: house rules discovered only by failing (enforced by the project's check with terse messages, not
  documented in the README), a repository too large to read whole, and environment quirks that cost turns.
- **The director's loop works live.** `outcome` recorded each verdict; after task 2, `consolidate --policy verified`
  drew one lesson from the two passed sessions (a method for adding a command, each step with its check), accepted by
  policy with `decided_by: policy:verified`; a first proposal with a lint problem was refused. Task 05's session read
  the Index and the lesson before working. The leak audit found no verifier or reference text in the memory.
- **Two fixes from it.** The bench now stops a run when a session never starts, instead of recording failed tasks
  (Claude Code refuses `bypassPermissions` as root unless `IS_SANDBOX=1`, which the first attempt hit). The director
  now reports tokens the way the bench counts a session (cache reads apart) and closes a proposal its policy refuses.
- Spent about US$0.70 in total.
