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

## Outcome (2026-10-17) — **MEMORY HELPS, GAP REMAINS**

120 plans and 120 verdicts, 0 unparsed. Cost US$13.48 (T's plans and the judge).

| arm | mean plan score |
|---|---|
| N: 12B, no memory | 0.042 |
| B: 12B + BM25 top 15 | 0.681 |
| S9: 12B + D9 walk | 0.611 |
| **S10: 12B + D10 walk** (the walk) | **0.681** |
| T: Sonnet walking | 0.917 |

| comparison | wins / losses / ties | p | Δ |
|---|---|---|---|
| the walk vs N | 11 / 0 / 1 | 0.001 | +0.64 |
| the walk vs T | 0 / 8 / 4 | 0.008 | −0.24 |
| S10 vs S9 | 5 / 3 / 4 | 0.73 | +0.07 |
| the walk vs B | 5 / 5 / 2 | 1.0 | 0.00 |

Data: `~/.config/director/d11/{cases,plans-*,verdicts}.jsonl`, `report.json` (private).

**Diagnosis, computed after the verdict (not a redesign).** The diagnosis uses the bank's recorded `grounds`, the
statements each case was written from. No arm saw them.

| arm | grounds among the statements it had | statements |
|---|---|---|
| B | 0.72 | 15.0 |
| S9 | 0.65 | 5.6 |
| **S10** | **0.90** | 18.9 |
| T (cited) | 0.95 | 19.0 |

- **Retrieval is nearly closed; writing the plan is the gap.** S10 put 0.90 of the grounds in front of the 12B, close to
  Sonnet's 0.95, with the same number of statements. Yet it plans at 0.68 against 0.92.
- **What the misses look like.** When the 12B has a statement, it applies it partially, or it lets the user's request
  override it. In n06 it had `Statistics#seeds_before_verdict` and wrote "because this is a single draw, I am
  reporting the result as requested". In n09 it had the headroom rule and ran an audit, not the baseline arm.
- **D10's low "precision" did not hurt the plan.** S10 ≥ S9 (5/3).
- **The walk does not beat BM25 at plan level** (5/5), although it retrieves more of the grounds (0.90 vs 0.72). The
  extra statements did not become better plans.
- **Caveats.**
  - The bank was written *from* the statements, so it favours memory arms (N scores 0.04) and may favour keyword
    search.
  - The judge is strict about details (in n06, naming the run directory).
  - This bank is now worn too: the diagnosis above was made on it.

**What this means for the design.** The bottleneck has moved from navigation to execution. A 12B that free-writes a plan
from 19 statements under-applies them. The next step is for code to compose the plan, as it already drives the walk. Each
kept statement's **Do** becomes a step. The 12B answers closed questions per statement: does it apply? does the user's
request conflict with it? what fills its arguments? The step is to be briefed fresh, with a new bank.
