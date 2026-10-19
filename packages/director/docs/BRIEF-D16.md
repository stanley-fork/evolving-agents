# D16: does a directory of lanes, as data, route as well as routing rules in prose?

**Written 2026-10-19, before anything is built or run.** Step 2 of [`docs/EVOLUTION.md`](../../../docs/EVOLUTION.md) §8.
Today the facts that decide where a task runs are spread over three places: `lanes.json` (the commands and
`confidential_ok`, private), lora-kernel's domains (in the director's config), and statements in the memory written
in prose (D3's `Task_Lanes#public_text_on_frontier`, the confidentiality rules). D16 asks whether the same facts,
moved into **one directory of lanes as data** that the statements cite, route tasks at least as well, keep D0 where it
is, and do something prose cannot: change routing when a fact about a lane changes, without a lesson.

Nothing runs until the user authorises the spend (cap below). The run needs the lab's private memory and lane
configuration, which live on the user's machine, not in this repository.

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
