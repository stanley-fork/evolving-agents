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
