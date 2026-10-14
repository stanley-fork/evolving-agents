# D2 — does the director learn from a task, without breaking what it knew?

**Written 2026-10-14, before anything runs.** The user's instruction the same day: "seguí con D2".

## What and why

D0 showed that a session walking the compiled operational memory plans the lab's way (0.929 vs 0.589; the
organisation carries it, A+ 0.768). D2 is the half of the director that **learns**: after a task, it reads
what happened and the correction it received, and proposes edits to its memory; an edit is kept only if a
gate says it helps where it should and costs nothing where it should not, and every kept edit is a versioned,
gated patch in agentvcs.

## Lessons (real incidents of 2026-10-14, absent from the memory)

`d2/episodes.jsonl` — each an incident as it happened plus the correction, the input the director learns from:

| | lesson | in memory before D2 |
|---|---|---|
| L1 | a scripted edit (`sed`, regex replace) that matches nothing exits 0; verify it landed before relying on it | no |
| L2 | a command handed to a runner that executes in another directory needs absolute paths; an all-fail result is read first as a harness problem | a neighbour only (workflow `working-directory`) |
| L3 | `claude -p --bare` skips `CLAUDE.md` auto-discovery; an arm that must see project instructions uses `--restricted` and proves they load with a probe | no |

## Probes — written now, never shown to the learning step

`d2/probes.jsonl`: **6 situations, 2 per lesson**, worded and set apart from the episode (another file, another
tool, another goal), with action checklists in D0's format and D0's blind judge.

## Steps (in sequence)

1. **T0** — today's compiled memory on the 6 probes (k = 2). **Headroom gate: T0 ≥ 0.85 → STOP** (the
   lessons are not new to the director; reshape).
2. **Reflect** — Claude Opus 5.5 with the memory (read-only) and one episode at a time proposes edits as
   JSON (`add` / `modify` a statement, the index situations that reach it). Applied deterministically to a
   copy; **lint gate**: ids unique, links resolve, every new statement reachable from `Index.md`. An edit that
   fails lint is dropped and counted.
3. **T1** — the edited memory on the same probes.
4. **Guard** — T1 on D0's 14 cases.
5. **agentvcs** — memory v0 and v1 as manifests (one `prompt` dimension per page), a run whose steps are the
   scored probe samples, the v0 → v1 patch proposed with the lessons as rationale, gated by a suite that reads
   the scores, applied only if it passes; `verify` and `blame` on the probe metric.

## Verdict (fixed here)

| | condition |
|---|---|
| **STOP — no headroom** | T0 ≥ 0.85 |
| **LEARNS** | T1 − T0 ≥ +0.30 on the probes **and** T1 wins more probes than it loses **and** the guard holds: T1 on D0 ≥ 0.86 (T's 0.929 less one item in fourteen cases' worth of slack) |
| **LEARNS, BREAKS** | the probe gain holds and the guard fails — the edits were kept by a gate that could not see the damage |
| **DOES NOT LEARN** | otherwise |

Same model, tools, turns, budget per call and judge as D0's arm T. Cap **US$40** for D2, read from every
call's `total_cost_usd`. Plans, edits and verdicts stay private; scores, the edit count and the lint results
are published. Stopping: one redesign at most, with the user's sign-off.

## Not in D2

Reflection inside the running interface (D2b, after this passes), the user accepting or rejecting an edit by
hand, lane learning (D3), more than three lessons.

## T0 [ran] 2026-10-14 — 0.792: GO by the letter, **void by arithmetic**; stopped before reflecting

12/12 plans, 0 errors, 12/12 verdicts. Per probe: p1, p2 (L1), p3 (L2), p5 (L3) at **1.0**; p6 (L3) 0.75;
**p4 (L2, a gate command's working directory) 0.0**.

**The instrument error, found before any treatment ran:** the headroom gate (T0 < 0.85) did not check that
the pass rule could be met. With T0 = 0.792 the largest possible gain is +0.208, below the +0.30 the brief
asks: LEARNS was unreachable by construction. A headroom gate must be stated against the effect the verdict
requires — `1 − T0 ≥ the required gain` — not against a fixed level. Recorded as written; nothing reran.

**What T0 says on its own.** Two of the three lessons (verify a scripted edit; `--bare` drops CLAUDE.md)
are general engineering the model already applies without any memory — they are not what the director's
memory is for. Its value in D0 was lab-specific knowledge the model cannot have (the subtree, the mirrors,
the stealth rule). Lessons for D2 have to be of that kind.

**Stopped here** (US$5.59 spent). The one redesign the brief allows needs the user's sign-off.

## Redesign 1 (of 1), the user's sign-off 2026-10-14 — written before it runs

**Lessons the model cannot already know** (`d2/episodes2.jsonl`), all real and absent from the memory (grep):
A — where the director's private data lives and how cases carry private names (placeholders);
B — agentvcs conformance cases come from `gen.py` + the reference, CI checks three CLIs, never hand-edit;
E — the local clone of evolving-agents is `evolving-agents-readme`, and its branch must be `main` before branching;
F — pictures for public pages are read back and scrubbed of local paths and account names;
I — when a project's built / not-built state changes, the org profile and the website home change with it.
**Probes** (`d2/probes2.jsonl`): 10, two per lesson, apart from the episodes.

**Headroom screen, then a fresh control** (the fix for T0's error, and for regression to the mean):
1. **S** — today's memory on the 10 probes, k = 1. Keep the probes scoring ≤ 0.5. Fewer than 4 kept → **STOP**.
2. **T0c** — today's memory again on the kept probes, k = 2, fresh samples: the control.
3. **Reflect** on all five episodes (the learner does not know which will be probed), lint gate as before.
4. **T1b** — the edited memory on the kept probes, k = 2. **Guard** — the edited memory on D0's 14 cases.

| verdict (fixed here) | condition |
|---|---|
| **LEARNS** | T1b − T0c ≥ ½ · (1 − T0c) — half the headroom the control leaves — **and** T1b wins more kept probes than it loses **and** guard ≥ 0.86 |
| **LEARNS, BREAKS** | the gain holds, the guard fails |
| **DOES NOT LEARN** | otherwise |

Cap for D2 raised to **US$45** by the user (≈ US$38 more).

## Redesign 1 [ran] 2026-10-14 — **LEARNS**

- **S (screen, k = 1):** 8 of 10 probes at 0 — every lab-specific lesson (A, B, E, I); F's two probes at 1.0
  (scrubbing local paths from a picture is, again, general practice the model already has).
- **T0c (fresh control, k = 2, the 8 kept):** **0.0** on all 16 samples.
- **Reflect (Claude Opus 5.5, read-only, one episode at a time):** 9 edits — 4 new statements, 5 modified —
  0 rejected, 0 lint problems, all five episodes kept; US$4.1. The learned statements generalise the incident
  (e.g. *edit the generator, not its output*; *find a clone by its remote*) and keep it as the **Because**.
- **T1b (k = 2):** **0.781** — A 1.0 · 1.0, B 0.75 · 1.0, I 1.0 · 1.0, **E 0 · 0.5**.
- **Guard (D0's 14 cases, k = 2):** **0.964** (T was 0.929; only c10 below 1.0, at 0.5). One judge reply did not
  parse and was re-judged once (2/2), as the runner's resume does.

T1b − T0c = **+0.781** against a required +0.5; wins/losses **7 : 0**, exact one-sided sign test **p = 0.0078**;
guard 0.964 ≥ 0.86 → **LEARNS** by the table fixed before the run. D2 total **US$41.99** of US$45.

**Where it learned least, read where it happens:** lesson E. The reflection generalised "the clone of
evolving-agents is `evolving-agents-readme`" into "find a clone by its remote" — a better rule, which the probes
did not reward: asked for the exact first commands, the director wrote the generic search, or guessed the
path. Generalising a fact into a procedure loses the fact when the probe wants the fact.

**Versioned in agentvcs [ran]** (`d2/version.py`, store private): memory v0 and v1b as manifests (one `prompt`
dimension per page, 6 dimensions changed), a run with the control's 16 probe scores under v0, the v0 → v1b
patch with the lessons as rationale, **gated** by a suite reading this report (gain over the bar +0.28, guard
0.964 — passed), **applied**, the edited memory's 16 scores under v1b; `verify` valid; **`blame` attributes
+0.781 to exactly that patch.** v1b is now the director's live memory.

*Published-text note:* probe `qA2` ran with a short description of the private project in its situation; in
the published `probes2.jsonl` it reads "a private project", because the stealth rule forbids an identifiable
description of private work in a public repository. The original text stays in the private directory.
