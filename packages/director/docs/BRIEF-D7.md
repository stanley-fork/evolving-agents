# D7 — a positive probe: does a lesson actually change what the director does?

**Written 2026-10-15, before anything runs.** The user's "continua" after D6: the targeted guard checks that an edit breaks
nothing that relied on what it changed, but **an added statement is checked by nothing** — and an edit can pass every guard
while teaching nothing.

## The design

At reflection, the director proposes with each edit **two probe situations** — a task in flight where the lesson should
change the plan, worded apart from the episode (another file, tool or goal) — each with action checklist items in D0's
format. At Accept, the probes run (D0's runner and blind judge, Sonnet 5.5, read-only tools, k = 2) on **the current memory
and the candidate**:

| verdict | condition |
|---|---|
| **TEACHES** | candidate mean ≥ 0.67 **and** candidate − current ≥ 0.33 |
| **REDUNDANT** | current already ≥ 0.67 — the memory or the model already does it |
| **NO EFFECT** | otherwise |

Accept keeps an edit only if it TEACHES (and the targeted guard passes); REDUNDANT and NO EFFECT are shown to the user.
A probe whose situation shares an 8-word run with the episode is flagged as a leak.

## Validation — lessons whose effect D2 already measured on held-out probes

Current memory = the compiled memory **before** D2's edits (D2's T0 memory). Inputs = D2 redesign's five real episodes
(`d2/episodes2.jsonl`), reflected again now with probe generation; the candidate = that memory plus the edit.

| lesson | what D2's held-out probes showed (T0c → T1b) | the probe should say |
|---|---|---|
| A — private data stays out of the repo | 0 → 1.0, 1.0 | **TEACHES** |
| B — edit the generator, not its output | 0 → 0.75, 1.0 | **TEACHES** |
| I — status surfaces change together | 0 → 1.0, 1.0 | **TEACHES** |
| F — scrub local paths from pictures | screened at 1.0 before any lesson (the model knows it) | **REDUNDANT** |
| E — the clone's real path | 0 → 0, 0.5 (learned least) | descriptive |

| D7 verdict (fixed here) | condition |
|---|---|
| **VALIDATED** | A, B and I TEACH, and F is REDUNDANT |
| **MISSES** | one of A, B, I does not TEACH |
| **NOT SPECIFIC** | F is not REDUNDANT |

Cost ≈ five reflections (Opus 5.5, ≈ US$4.5) + 5 lessons × 2 probes × k = 2 × 2 memories = 40 plans (≈ US$19) + the judge;
cap US$35. One design; no redesign on these data.

## Result [ran] 2026-10-15 — **VALIDATED**, and the probe is the lesson's own exam

| lesson | D2's held-out probes | positive probe: current → candidate | verdict | expected |
|---|---|---|---|---|
| A | 0 → 1.0, 1.0 | 0.083 → 0.75 | **TEACHES** | TEACHES ✔ |
| B | 0 → 0.75, 1.0 | 0.625 → 1.0 | **TEACHES** | TEACHES ✔ |
| I | 0 → 1.0, 1.0 | 0.0 → 1.0 | **TEACHES** | TEACHES ✔ |
| F | known before any lesson | 0.917 → 1.0 | **REDUNDANT** | REDUNDANT ✔ |
| E | 0 → 0, 0.5 (learned least) | 0.625 → 1.0 | TEACHES | descriptive |

No generated probe shares an 8-word run with its episode. US$25.68. **Verdict by the table fixed above: VALIDATED.**

**Read before relying on it.**
- **E is the warning.** D2's independent probes asked for the fact (the clone's real path) and the edit generalised it
  into a procedure ("find the clone by its remote"); the probes the reflection wrote test the procedure, and pass. The
  writer of the lesson writes its exam: the probe catches lessons that change nothing (NO EFFECT) or that the memory
  already holds (REDUNDANT); it does not catch a lesson that is the wrong lesson. That stays the person's Accept.
- **An account limit was recorded as results once.** claude.ai's session limit hit mid-run: lesson F's candidate plans
  and lesson I's reflection came back as errors at US$0 and were scored INCOMPLETE / NO PROBE. They were rerun after the
  reset (F re-reflected from scratch, its stale arm deleted); the runner and the reflection now stop on an account limit
  instead of recording it.

Wired into Accept: a lesson is kept only if the targeted guard passes **and** its probe TEACHES; REDUNDANT, NO EFFECT and
NO PROBE leave the memory as it was and say why. `teaches` is part of the agentvcs patch's gate.
