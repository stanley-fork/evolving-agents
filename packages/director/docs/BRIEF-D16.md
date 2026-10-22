# D16: does a directory of lanes, as data, route as well as routing rules in prose?

**Written 2026-10-19, before anything is built or run.** Step 2 of [`docs/EVOLUTION.md`](../../../docs/EVOLUTION.md) §8.
Today the facts that decide where a task runs are spread over three places: `lanes.json` (the commands and
`confidential_ok`, private), lora-kernel's domains (in the director's config), and statements in the memory written
in prose (D3's `Task_Lanes#public_text_on_frontier`, the confidentiality rules). D16 asks whether the same facts,
moved into **one directory of lanes as data** that the statements cite, route tasks at least as well, keep D0 where it
is, and do something prose cannot: change routing when a fact about a lane changes, without a lesson.

Nothing runs until the user authorises the spend (cap below). The run needs the lab's private memory and lane
configuration, which live on the user's machine, not in this repository.

> **Run: to be done by the user on their machine, the night of 2026-10-19.** None of the run's inputs exists yet, so
> the run starts by preparing them, in this order, before any plan is made:
>
> 1. the **fact sheet**: the lab's lanes as about 30 facts, private, `~/.config/director/d16/facts.md`;
> 2. the **bank**: 18 cases by a fresh agent from the fact sheet alone, `~/.config/director/d16/cases.jsonl`, then read
>    once to drop any case its facts do not decide;
> 3. **arm P**: a copy of the live memory whose lane statements carry the facts in prose; **arm D**: the same copy with
>    those statements citing a `directory.json` written from the facts;
> 4. the **parity check** by a second fresh agent; a missing fact stops the run until it is fixed;
> 5. a runner, `packages/director/d16/run_d16.py`, in the shape of `d15/`: plan each case through each arm, read the
>    lanes from the plan's JSON, write `report.json`. Then measurements 1–4 and the verdict, against the table below,
>    unchanged.
>
> The result is added here as `## Outcome`, flat if it is flat.

## The directory (what step 2 would build)

`.director/directory.json` (shareable) plus an optional private overlay `~/.config/director/directory.json` (merged
over it; the commands stay in `lanes.json`). One entry per lane, whatever runs it: the session, a headless Claude Code,
a local model, a lora-kernel specialist, a command lane, a managed agent.

```json
{"entries": [
  {"id": "local", "kind": "local-model",
   "description": "Gemma on llama.cpp, on the user's machine",
   "good_for": ["internal text: summaries, notes, classification"],
   "avoid_for": ["public-facing text", "multi-step work with tools"],
   "confidential_ok": true, "region": "user's machine", "cost": "free",
   "health": "up", "fallback": "frontier",
   "reliability": [{"claim": "abstains outside its measured phrasing", "value": "unreliable",
                    "evidence": "packages/director/docs/BRIEF-D4.md"}]}
]}
```

| field | meaning |
|---|---|
| `kind` | `session`, `claude-p`, `local-model`, `specialist`, `command`, `managed` |
| `good_for`, `avoid_for` | kinds of work, in the words a request would use |
| `domains` | a specialist's domains (lora-kernel) |
| `confidential_ok` | may receive a confidential task; enforced in code by `lanes.py`, whatever the plan says |
| `region`, `cost` | where the work runs; `free`, `low`, `high` |
| `health`, `fallback` | `up`, `down`, `unknown`; the lane to use instead |
| `reliability` | claims about the lane, each with its evidence (a brief, a run) |

Statements stop describing lanes and cite entries instead: *When a draft will be read outside the lab, Do plan it on
the lane the directory marks `good_for` public-facing text*, with the entry named. Lint checks that every cited entry
exists. The directory is the fourth memory of EVOLUTION §4: its configuration is the person's, and a reliability
change carries its evidence.

## Arms

Both arms plan with the same planner and settings as D0's arm T: Claude Sonnet walking the memory with Read, Grep
and Glob, k = 2. Each plan is JSON whose every task names one lane from the same set of six:

| lane | in brief |
|---|---|
| `frontier` | this session (Claude Code) |
| `isolated` | headless Claude Code (`claude -p`), read-only tools |
| `local` | a local model on the user's machine |
| `tracker` | a lora-kernel specialist, one domain, its own tools |
| `private-exec` | a command lane on the lab's own machine, `confidential_ok` |
| `cloud-agent` | a managed agent (Managed Agents or Antigravity), not `confidential_ok`, a provider region |

- **P (prose, the control):** today's form. Lane names and one-line descriptions in the prompt, as `lanes_list`
  gives them; every other fact as statements in the memory, written in prose.
- **D (directory):** the directory rendered into the prompt; the same statements rewritten to cite entries; no lane
  fact left in prose.

**Information parity.** Both arms are written from one **fact sheet** (plain English, about 30 facts: each lane's
use, limits, clearance, region, cost, health, fallback, and the reliability claims with their evidence). A second
fresh agent checks that every fact appears in both arms and that neither carries one the other lacks. A fact missing
from either arm stops the run until it is fixed.

## The bank

A **fresh Claude agent** (Opus 5.5) writes 18 routing cases from the fact sheet alone. It never sees either arm,
D3's probes or this brief's verdicts. Three cases in each of six kinds:

1. public-facing text (D3's preference);
2. internal text (D3's control);
3. confidential material: a client's file, or an input under a confidential folder;
4. work in a specialist's domain;
5. a lane whose reliability claim should keep a task away from it;
6. bulk or long work where cost or a sandbox decides.

Each case lists its acceptable lanes (more than one where the facts allow it) and one line naming the facts that
decide it. Before the run, the user or a second agent reads the bank and drops any case whose answer is not decided
by the facts.

## Measurements

1. **Lane accuracy.** For each case, the fraction of the plan's tasks in an acceptable lane, averaged over k = 2.
   Read from the plan's JSON; no judge.
2. **Clearance violations.** Tasks with confidential input planned on a lane without `confidential_ok`. `lanes.py`
   would refuse each one at delegation, so a violation wastes a step and is counted against the arm.
3. **Following the data (arm D only).** Arm D re-plans the cases whose acceptable lanes include `local` or
   `tracker`, with one fact changed in the directory and nothing else: `local` marked `down`, then `tracker` marked
   `down`. A task that was on the lane now down should move to that entry's `fallback`; every other task should stay
   where it was. Arm P has no counterpart without a new lesson, which is the point being measured.
4. **D0 holds.** D0's 14 situations planned through arm D's memory, judged blind by D0's judge, k = 2.

## Verdicts (fixed here)

| verdict | condition |
|---|---|
| **DIRECTORY ROUTES BETTER** | D's lane accuracy ≥ P's + 0.10, with more case wins than losses; and every other condition of DIRECTORY HOLDS |
| **DIRECTORY HOLDS** | D ≥ P − 0.05 and D has no more violations than P; and D0 through D ≥ 0.899 (0.929 − 0.03); and in measurement 3 ≥ 0.8 of the affected tasks move to the fallback while ≥ 0.9 of the unaffected stay put |
| **DIRECTORY LOSES** | otherwise: the directory is not built; prose stays; the result is published as flat |
| *no headroom* | no stop: if P already routes ≥ 0.95, DIRECTORY HOLDS is the bar, because the directory's case is that a fact changes once, as data, not that it routes better |

Also reported, not scored:

- accuracy per kind of case;
- P vs D per case (sign test);
- cost per plan;
- how many words of the prompt each arm spends on lanes.

## Cost

72 plans for the two arms (18 cases × k = 2 × 2 arms), about 20 for measurement 3 and 28 for D0, plus D0's judge.
At D15's US$0.48 per plan with tools, about US$60. **Cap US$75.**

## What this will and will not show

It will show whether lane facts kept as data route as well as the same facts in prose, under information parity, on
cases written from the facts and not from either arm. It will not show which lanes are good: the directory's claims
are inputs here, not results. Nor will it show routing on real requests over time; the journal (step 5) will. And the
bank is written by Claude from a fact sheet Claude can read, which favours both arms equally but not a stranger's
phrasing.

## Outcome [ran] 2026-10-19 — **DIRECTORY LOSES** by the table; it routes as well as prose, but does not yet earn the rest

The inputs, in the order of the Run box: a fact sheet of 30 facts on six lanes; a bank of 18 cases by a fresh Opus agent
from the facts alone, then read once by a second fresh agent, who dropped 3 cases the facts did not decide (one listed
`frontier` for a check meant to run without the session's context; one excluded the untested managed agent where two
others sent work only there; one set a cost threshold the facts do not) — **15 cases**; arms P and D built from a copy of
the live memory, both lint clean. **The parity check stopped the run twice** before any plan: arm D had weakened three
facts (the local model's size and failure, a confidential task waiting when its lane is down, the tracker's mid-session
failure), and then lacked "frontier model"; each arm carried rationale sentences the facts lack. Fixed in the inputs, not
the instrument; the third check found nothing missing, and one shared extra (the release-note incident, in both arms).

| measurement | P (prose) | D (directory) | bar | |
|---|--:|--:|---|---|
| 1 · lane accuracy, 15 cases × k = 2 | 0.864 | **0.880** | D ≥ P − 0.05 | ✔ (2 wins each, 11 ties, p 0.69) |
| 2 · clearance violations | **1** | 5 | D ≤ P | ✘ |
| 3 · following the data: affected tasks to the fallback | — | 0.762 | ≥ 0.8 | ✘ |
| 3 · following the data: unaffected tasks stay | — | 0.786 | ≥ 0.9 | ✘ |
| 4 · D0 through D's memory, blind judge | — | **0.929** | ≥ 0.899 | ✔ |

Cost **US$55.92** of the US$75 cap (bank and read-through 1.85, parity 3.84 over three checks, P 13.98, D 14.02,
measurement 3 7.90, D0 13.96 and its judge). US$0.47 per plan in both arms. The lanes cost 64 prompt words + 1,032 memory
words in P, 624 + 445 in D.

**Read before relying on it — where each failure happens.**
- **Violations.** P's one and two of D's five are the same harmless act counted by the rule: in a confidential case, a
  step that reads a script with no client data, or reports counts to the user, planned on `frontier`. D's other three
  are one plan (internal-3, sample 1) that marked internal text `confidential: true` and then put it on `frontier` — a
  plan contradicting itself, which `lanes.py` would refuse at delegation. Counting only the case's confidentiality,
  it is P 1, D 2: still a loss on this bar.
- **Measurement 3 aligns tasks by position**, as written before the run, so a re-plan with fewer tasks counts its missing
  tasks as moved: internal-3 went from 3 tasks to 1. Not scored, read as sets: the down lane left **14 of 15** plans, and
  **11 of 12** unaffected lanes stayed. The one real failure is where it matters most: with `local` down, confidential-1
  (a client file) put one step on `frontier` in one of two samples, where the directory says a confidential task waits for
  a cleared lane (`private-exec`).
- **Bulk is where both arms fail** (P 0.229, D 0.267): plans keep most steps of a long sandboxed job on `frontier` and
  hand only the run to `cloud-agent`; the bank accepts only `cloud-agent`. The cases or the planner — not separated here.
- Every other kind scores 1.0 in both arms: public, internal, reliability, specialist.

**What it says.** Lane facts kept as data route as well as the same facts in prose (0.880 vs 0.864) and leave D0 where
it was (0.929); what the brief asked beyond that — no extra violations, and a fact change followed as data — did not
clear its bar, on one instrument choice (position alignment) and one real routing error under a down lane. By the table:
**the directory is not built**; prose stays. One measurement, k = 2; not re-run, no redesign on these data.
