# D6 — a targeted guard: re-plan only the cases that relied on what an edit changed

**Written 2026-10-15, before anything runs.** The user's go-ahead ("si") after D5 found the paired guard BLIND (0/4
detection at k = 1 over D0's 14 cases, 7 of which flip between identical arms).

## The design

An edit can only break a case that **relied on what it changed**. The current memory's plans cite the statements they
followed (`[[Page#id]]`), so:

1. **changed** = the statements an edit removes or modifies (an added statement is not checked here — it is checked
   by its own positive probe, a separate step);
2. **relevant cases** = the D0 cases whose baseline plans cite a changed statement, read from the cached baseline
   (D5's four arms of the unchanged live memory: k = 4 per case, US$0 now);
3. the candidate re-plans **only those cases, k = 3**;
4. **refuse** iff some relevant case was reliable before (baseline mean ≥ 0.75) and fails now (candidate mean ≤ 1/3).

Computed from D5's plans before writing this (free): `Workspace_Rules#ai_base_byte_identical` is cited only by c13;
`Documentation#mirror_same_commit` by c04, c08, c09; `Experiment_Briefing#write_brief_first` by c02, c03, c06, c08, c10,
c12, c14. Statement-level citation, not page-level: the mirror's page is cited by 11 of 14 cases.

## Validation (D0's cases, Sonnet 5.5 via `claude -p`, read-only tools, the blind judge — as the guard runs)

| arm | memory | cases (k = 3) | expected |
|---|---|---|---|
| **B1** | live memory minus the subtree rule | c13 | refused |
| **B2** | live memory minus the mirror rule | c04, c08, c09 | refused |
| **A′1, A′2** | the live memory, unchanged, run as if it were a candidate | c04, c08, c09, c13 | not refused |
| **Bc** (descriptive) | live memory minus the brief-before-run rule — a rule D5 showed the model applies anyway | c02, c03, c06, c08, c10, c12, c14 | reported, not scored |

| verdict (fixed here) | condition |
|---|---|
| **VALIDATED** | B1 **and** B2 refused, **and** neither A′ refused |
| **TOO STRICT** | an A′ refused |
| **BLIND** | B1 or B2 not refused |

The baseline is reused, not re-run: D5's arms are of this exact memory (checked by content hash). Each broken copy is
checked to remove exactly its statement and every link to it. Cost ≈ 57 plans ≈ US$27, cap US$35. One design, no
redesign on these data: if not VALIDATED, it is reported and the next step is briefed fresh.

**Clarified before anything ran (2026-10-15), and why.** The dry check of the runner (no model calls) found that
removing a statement also edits the Links line of every statement that pointed to it, and the first `changed_statements`
counted those as modified — B2's relevant set grew from 3 cases to 10. The design means **substance**: a statement is
changed if it is removed or its When, Do or Because changes; a change to Links alone is navigation, and a dangling link
is the lint's business. With that, the relevant sets are the ones listed above, and the runner asserts them.

## Result [ran] 2026-10-15 — **VALIDATED**, with thin margins

Baseline: D5's four arms of the unchanged live memory (content hash checked). k = 3 on each arm's relevant cases;
57 plans, US$28.34.

| arm | relevant cases: baseline mean → candidate mean | guard |
|---|---|---|
| **B1** (subtree rule removed) | c13 0.75 → **0.333** | **refused** |
| **B2** (mirror rule removed) | c09 1.0 → **0.333**; c04 1.0 → 1.0; c08 0.875 → 1.0 | **refused** (c09 only) |
| **A′1** (unchanged) | c04 1.0, c08 1.0, c09 1.0, c13 0.667 | passed |
| **A′2** (unchanged) | c04 1.0, c08 1.0, c09 1.0, c13 0.667 | passed |
| Bc (descriptive: brief-before-run rule removed) | c12 0.875 → **0.333**; the six others held or rose | would refuse |

**Detection 2/2, false refusals 0/2 → VALIDATED** by the table fixed above. The guard saw exactly the case that relied on
the removed rule and left the others alone; per Accept it costs the relevant cases × 3 plans (B1 US$1.4, B2 US$4.2)
instead of D0's whole suite twice (US$13–14).

**Read before relying on it.**
- **Thin margins.** B1's c13 landed exactly on the failure line (1/3) from a baseline exactly on the reliability line
  (0.75); against A′, c13's separation is one sample (0.333 vs 0.667). Two breaks and two unchanged arms are not a rate.
- **The control was refused**, on c12 — a case D5 showed flipping (1, 1, 0.5, 1). Either the brief rule does carry c12
  (its plans cite it), or this is a false refusal on a noisy case; the A′ arms measured false refusals only on four
  stable cases. Noisy relevant cases should get more samples before the guard trusts a drop.
- **An added statement is not checked** by this guard (nothing relied on it yet); that needs its own positive probe.
