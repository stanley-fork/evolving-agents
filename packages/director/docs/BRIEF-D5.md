# D5 — can the paired guard fail, and does it refuse an unchanged memory?

**Written 2026-10-15, before anything runs.** The user's go-ahead the same day ("si") to the review's first point: the
guard behind Accept was argued, not validated — its threshold (refuse at 4 net losses) was set after seeing D4's data.

## What and why

A guard is an instrument. Before its verdict is believed, two things must be shown by running it: it **does not refuse
an unchanged memory** (false refusals), and it **does refuse a memory that is broken** (detection). D4's KEPT rests on
neither.

## Arms (D0's 14 cases, Sonnet 5.5 through `claude -p`, read-only tools, the blind judge, k = 1 — the guard as it runs)

| arm | memory | runs |
|---|---|---|
| **A1–A4** | the live memory as it is (291 statements), unchanged | 4 independent arms |
| **B** | the same memory with four rules removed — `Workspace_Rules#ai_base_byte_identical`, `Documentation#mirror_same_commit`, `Experiment_Briefing#write_brief_first`, `Confidentiality#never_name_private_in_public` — and every link and index entry to them | 1 arm |

The removed rules are the ones D0's cases c13 (subtree), c09 (mirror), c03 (brief before run) and c04 (private name)
depend on, so a working guard should see B lose up to four cases.

## Readings (fixed here)

The guard's rule, `paired_verdict`: refuse iff the candidate loses ≥ 4 cases more than it wins.

| | computed on | acceptable |
|---|---|---|
| **false refusals** | the 6 A/A pairs (Ai as candidate vs Aj as current, i < j, both directions counted once) | **≤ 1 of 6** |
| **detection** | B as candidate vs each of A1–A4 | **≥ 3 of 4** refused |

Beside the verdict: each case's spread across A1–A4 (how many of the 14 cases flip at all), each A arm's mean, and how
an absolute floor of 0.86 would have read each arm.

| verdict | condition |
|---|---|
| **VALIDATED** | both acceptable |
| **TOO STRICT** | false refusals > 1 of 6 |
| **BLIND** | detection < 3 of 4 |

If not VALIDATED, the threshold is not changed on these data; the next step is a design change (k = 2 on the cases this
run shows to flip) briefed and run on fresh arms.

Cost ≈ 5 × US$6.6 ≈ US$33, cap US$40. Arms persist per plan and per verdict; a missing plan or verdict is retried twice.

## Result [ran] 2026-10-15 — **BLIND**

Five arms, D0's 14 cases, k = 1, the blind judge; US$33.48. The broken copy was checked before it ran: exactly the four
statements gone (291 → 287), no link left to them, lint clean — a first version of the copier deleted 13 (its link
regex ate newlines and glued the next statements' headers onto a line), which would have made detection easy.

| arm | A1 | A2 | A3 | A4 | **B** (4 rules removed) |
|---|---|---|---|---|---|
| mean | 0.929 | 0.893 | 0.857 | 0.893 | **0.786** |
| an absolute floor of 0.86 refuses | no | no | **yes** | no | yes |

- **False refusals: 0 of 6** A/A pairs (net losses −1 to +1) — acceptable.
- **Detection: 0 of 4** — against A1, A2, A3, A4 the broken memory lost 3 cases each and won 1, 2, 3 and 2: net losses
  2, 1, 0, 1, never the 4 the rule needs. **Verdict by the table fixed above: BLIND.**
- **Why, read where it happens.** 7 of the 14 cases change between identical arms (c02, c06, c08, c10, c11, c12, c13).
  B did lose what the removed rules carry where only the memory knows them — c09 (mirror) and c13 (subtree) at 0, and
  c07 — but won back noisy cases; c03 (brief before run) and c04 (private name) did not fall at all: the model applies
  those without the memory (D2's lesson again). Exploratory, not a verdict: counting only losses on cases stable across
  the other arms does not separate either — B 2, the unchanged A3 also 2.
- **A judge reply was dropped once** for an unescaped quote in its free-text evidence; the parser now reads `id` and
  `met` item by item when the JSON is invalid (tested), and the arm was completed without re-paying its plans. A plan
  over its US$0.60 budget was retried.

**What this changes.** D4's KEPT means the guard saw no damage; this run shows it would not have seen a four-rule break
either. The threshold is not changed on these data (the brief's rule). The guard stays as a check against gross damage
and is described as such. The brief named "k = 2 on the cases that flip" as the next design; with 7 of 14 cases
flipping that is most of the suite at twice the price, so the next design is **targeted** instead, briefed and run on fresh arms before it is believed: re-run
only the cases whose plans cite the pages an edit touches (the citations are in the cached baseline's plans), with
k = 3 on those, plus the edit's own positive probe.
