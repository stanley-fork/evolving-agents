# D14: local retrieval + Claude writes the plan, against Sonnet walking with tools

**Written 2026-10-17, before anything runs, and only to run if D13's retrieval is READY or PARTIAL.** This is the
recommended split (the user's decision of 2026-10-17):

- code retrieves locally, at no cost;
- the 12B filters a short list where that is useful;
- Claude writes the plan from what was retrieved.

The question is whether that split plans as well as Sonnet walking the memory with tools (T, D11–D12), and at what cost.

## Arms (D11 + D12 banks, 24 cases, k = 2)

| arm | plan |
|---|---|
| **R** | Sonnet 5.5 via `claude -p`, **no tools**. The prompt holds the situation and the D13 list. It is HYB@20, or HYB→F if D13 found the filter USEFUL. The list is shown as `[[Page#id]]` When / Do / Because, with D11's instruction to follow the ones that apply, cite them, and tell the user about conflicts, then D0's ASK. |
| **T** | already on disk (D11 and D12): Sonnet walking D0's sandbox with Read/Grep/Glob |

The judge is `d0/judge.md`, Sonnet, blind, for R's new plans. T's verdicts are reused.

## Verdicts (fixed here)

| verdict | condition |
|---|---|
| **SPLIT WORKS** | R ≥ T − 0.05 over the 24 cases |
| **SPLIT LOSES** | R < T − 0.05, reported with the per-case losses and whether R had the grounds |

**Also reported:**
- cost per plan, R vs T;
- per-bank means;
- R vs T per case (sign test).

## Cost

48 plans without tools plus 48 verdicts, estimated ≤ US$8. Cap US$15.

## Outcome (2026-10-17) — **SPLIT LOSES, by 0.005**

48 plans; 48 verdicts, 1 unparsed (left out). US$1.96 in total.

| | R: local retrieval (HYB@20) + Sonnet without tools | T: Sonnet walking with tools |
|---|---|---|
| mean over 24 cases | **0.837** | 0.892 |
| D11 bank | 0.792 | 0.917 |
| D12 bank | **0.882** | 0.868 |
| cost per plan | **US$0.028** | US$0.483 (17× more) |

R vs T per case: 6 wins, 6 losses, 12 ties (p = 1.0). The rule needed R ≥ 0.842, and R reached 0.837.

**Where R loses, computed after the verdict.** In its 6 losing cases R's list held 0.48 of the grounds; in the other
18 it held 0.84. The misses are retrieval misses. Given the statements, Sonnet without tools plans as well as Sonnet
walking. So the remaining gap is D13's PARTIAL retrieval (0.77 / 0.74), not the plan writer.

**What it means for the design.** Local retrieval feeding Claude for the plan costs 17× less, and on these banks it
cannot be told apart from walking with tools case by case. By the fixed margin it falls just short. The step that
would close it is retrieval recall, to be briefed fresh on a new bank:
- trajectory expansion of the fused hits, done by code: add the other statements of the trajectory most of the hits
  belong to;
- or a larger k for Claude, which reads 40 statements cheaply.
