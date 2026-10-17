# D11: does the 12B's walk lead to good plans, on a bank it was not designed on?

**Written 2026-10-17, before anything runs.** D8–D10 measured *citation recall*, against the statements Sonnet cited
in D0. D10's S10 walk reached 0.45 of that core at 0.65 precision. Its plans were not judged, and its design was
made on D0's cases. D11 changes two things. It uses a **new bank** that no strategy was designed on, and it measures
what matters: **the plan**, judged blind by D0's judge.

## The new bank

A **fresh Claude agent** (Opus 5.5) writes it. It reads only the memory pages (`~/.config/director/d9/memory`) and
`docs/MEMORY-FORMAT.md`, and it is told not to open anything of D0–D10 (cases, plans, briefs, results).

- **Size:** 12 cases, each a task in flight (the request and the state it arrives in), never a quiz.
- **Checklist:** 2–3 items about **actions**. Each case also records the statements it is grounded in. The arms never
  see those statements.
- **What a case needs:**
  - statements from at least 2 pages, as real tasks do;
  - wording that does not reuse the statements' When/Do;
  - a plausible wrong plan for a competent agent that does not know the lab's rules.
- **Where it lives:** `~/.config/director/d11/cases.jsonl`, private. It is written from the private memory, so it
  does not go in the public repo.

## Arms (k = 2 samples per case, same judge for all)

| arm | who plans | what it gets |
|---|---|---|
| **N** | the 12B | the situation only: no memory |
| **B** | the 12B | the situation + BM25's top 15 statements (When, Do, Because) |
| **S9** | the 12B | the situation + the statements its D9 walk picked |
| **S10** | the 12B | the situation + the statements its D10 (S10) walk picked |
| **T** | Sonnet 5.5 | D0's treatment arm as it ran: `claude -p` walks D0's memory sandbox with `DIRECTOR.md`, read-only tools |

**How the 12B plans:**
- model `gemma4:12b-mlx`, temperature 0.3, `think` off;
- one prompt: the lab's statements that apply, as a list citing `[[Page#id]]`, then D0's `ASK` (with "you may read
  files" removed, since it has no tools);
- one walk per sample, at temperature 0, as in D10.

The 12B arms get D9's memory, which has the trajectories. T gets D0's sandbox, as validated. Both have the same 286
statements.

**Judge:** `d0/judge.md`, Sonnet 5.5, blind to the arm. A sample's score is items met / items; an arm's score is the
mean over cases of the per-case mean.

## Verdicts (fixed here)

Here "the walk" is the better of S9 and S10, by mean score.

| verdict | condition |
|---|---|
| **LOCAL PLANNING WORKS** | the walk ≥ T − 0.10 **and** the walk beats N by ≥ 0.15 with more case wins than losses |
| **MEMORY HELPS, GAP REMAINS** | the walk beats N by ≥ 0.15 with more wins than losses, but < T − 0.10 |
| **NO HELP** | otherwise |

**Also reported, not scored:**
- S10 vs S9 (does D10's lower citation precision hurt the plan?);
- the walk vs B (does the 12B's walk add over keyword search?);
- each arm's per-case scores;
- the statements cited;
- cost.

## Cost and stopping

| item | estimate |
|---|---|
| bank (one agent) | ~US$2 |
| T plans: 24 × ≤ US$0.60 | ~US$12 |
| judge: 120 verdicts | ~US$4 |
| 12B arms | US$0 (local) |
| **cap** | **US$25**, read from each call's cost |

One design, no redesign on these data.
