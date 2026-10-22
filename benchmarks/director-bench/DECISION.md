# DECISION — pre-registered criteria

Fixed on 2026-10-22, **before any run that counts** (amended the same day, still before any
run that counts, to define RER over first-attempt CI feedback with the review round). Thresholds do not change after that; any
change needs a dated entry in the log below explaining why, made before seeing the affected results.

## Primary comparison

The director configuration (C2) against the native-memory baseline (C1), same pins, same
families, same seeds. 95 % confidence intervals come from a bootstrap that resamples seeds
(within each family, paired across configs; `bench report --boot-seed 0 --n-boot 2000`).

## Definitions fixed with the criteria

- **RER** (repeated-error rate): per task session, the failures are the unique normalized
  signatures of the **first attempt's CI feedback**, the final verifier failures of failed tasks
  and the tool errors of every round; RER = failures whose signature already appeared in an
  earlier session of the same run / all failures, pooled over seeds per family.
- **Review round**: a failed first attempt is resumed once (`feedback.rounds: 1`) with the
  verifier's CI/reviewer-style feedback; success is the final pass within the shared budget.
  The round exists because rules that live in the repository are learned by reading, not by
  failing: in the DB0 live checks (both ledgerkit versions, C0, Sonnet 5.5) a strong agent read
  the repo's checker before writing and never repeated an error. The recurring knowledge a memory
  is for — what remote CI enforces and reviewers ask for — reaches the agent only as feedback.
  First-pass rate and rounds per task are reported alongside but are not win conditions.

## Win conditions (all five, vs C1)

| # | condition | rule |
|---|-----------|------|
| 1 | Repeated-error rate (RER) | relative reduction (C1 − C2) / C1 ≥ **30 %**, and its 95 % CI excludes 0 |
| 2 | Success rate within budget | C2 − C1 ≥ **−2 pp** (point estimate) |
| 3 | Regression rate (vs C0, same seed) | C2 ≤ C1 + **1 pp** (point estimate) |
| 4 | Memory-induced error on the trap task | C2 ≤ C1 (point estimate) |
| 5 | Tokens per successful task (incl. prorated consolidation) | C2 ≤ C1 × **1.10** (point estimate) |

Mechanism condition: **A1 significantly worse than C2** in RER or in memory-induced error
(95 % CI of A1 − C2 entirely above 0).

Conditions 2–5 are non-inferiority guards and are judged on point estimates; their CIs are
reported alongside. Condition 1 and the mechanism are judged on intervals.

## Rule

- **Go** — all five conditions and the mechanism hold → proceed to weeks 5–8.
- **Iterate** — RER improves (condition 1 holds) but another condition fails → one more 2-week
  cycle on that condition, thresholds unchanged.
- **Pivot** — RER does not improve vs C1 → publish director-bench itself as the main
  contribution, with C0/C1 as reference results.

`bench report` prints an advisory verdict with this rule; the decision is recorded here by people.

## Decision log

| date | runs (ids / pins) | outcome | notes |
|------|-------------------|---------|-------|
| | | | |
