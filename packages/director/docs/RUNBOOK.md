# Runbook: what runs on the lab's machine

**Written 2026-10-19, at the hand-off from the cloud session to the user's machine; updated 2026-10-22** with what ran
since (D16, phase 1, phase 2, the Opus checks, PG) and plugin 0.2.4. What is still open is in §2–§5. Everything that could be built and
tested without the lab's private memory is merged on `main`: EVOLUTION steps 1, 4, 5 and 6 are built; steps 2 and 3
have their briefs. What is left needs the private memory (`~/.config/director/`), Ollama, or the user's spend, and is
listed here in the order it should run. Each item names its brief, its cap and where its result goes.

## 0. Before anything

```bash
git pull origin main
cd packages/director && python -m pytest tests -q          # 98 passed, 3 skipped (plugin 0.2.4, 2026-10-22)
claude plugin test . && claude plugin validate .           # the band: 8 pass
```

- **The 3 skipped tests need the real agentvcs CLI** (the Rust one, from its own repository) on PATH or in `$AGENTVCS`.
  They cover the versioned Accept. The frozen Python copy in `packages/agentvcs/` is out of date with the current CLI:
  with it installed, those 3 fail identically on `main` and on every branch since. Run them with the real CLI once.
- **Reinstall the plugin** so Claude Code picks up the band, plans and journal: `/plugin install director
  --marketplace EvolvingAgentsLabs/evolving-agents` (or `/plugin` → update). Not yet checked: that a Claude Code
  without plugin UI modules ignores the `modules` key in `hooks/hooks.json` and still runs the SessionStart hook. If
  you have an older Claude Code at hand, open a session with the plugin and confirm the walk still reaches the context.

## 1. D16, the directory of lanes — **done 2026-10-19: DIRECTORY LOSES** (see the brief's Outcome)

[BRIEF-D16](BRIEF-D16.md), **cap US$75**. Its "Run" box lists the inputs to prepare first, in order: the fact sheet,
the bank (18 cases, then a read-through), arms P and D, the parity check, and `d16/run_d16.py` in `d15/`'s shape. Then
measurements 1–4 against the verdict table, unchanged. The result goes in the brief as `## Outcome`, flat if flat, and
in EVOLUTION §8 step 2.

## 1b. Phase 1 of the roadmap, before D17 — **done 2026-10-20/21**

What ran, each with its brief and result:

| step | verdict | in one line |
|---|---|---|
| 1a · another provider's judge | **HOLDS** ([J0](BRIEF-J0.md)) | Gemini gives +0.30 (Sonnet +0.34) |
| 1b · held-out from the future | **HOLDS ON THE FUTURE** ([F1](BRIEF-F1.md)) | session 0.36, director 0.84 (Sonnet); +0.29 under Gemini |
| 2a · headroom for multi-day coherence | **GO to 2b** ([C0](BRIEF-C0.md)) | today's session keeps 0.275 of a thread's decisions |
| 2b · the director on the same thread | **THE LOOP WORKS** ([C1](BRIEF-C1.md)) | 0.275 → 0.725 with Sonnet; the lessons carry it |
| 2c · the same with Opus | **DOES NOT ADD** ([C2](BRIEF-C2.md)) | Opus alone 0.80, with the director 0.85; −34 % per plan |
| 2d · D0 with Opus | **STOP** ([D0O](BRIEF-D0O.md)) | Opus alone follows D0's rules at 0.964 |
| F1 with Opus | **ROOM under Opus, none under Gemini** ([F1O](BRIEF-F1O.md)) | the director's arm could at best split the judges |
| portability | **STOP, no headroom** ([PG](BRIEF-PG.md)) | Gemini follows the plain notes at 0.875 (floor 0.345) |

**What it means for what is left:** the measured gap is Sonnet's; with Opus planning, what the director adds is cost
and what lasts beyond one session (gated, versioned lessons that travel as plain notes). EVOLUTION §2 has the longer
reading. The plan as first written follows.

[ROADMAP](../../../ROADMAP.md#next--in-phases) now runs in phases, and phase 1 comes first: does D0's 0.929 hold
outside the lab's own history and outside one provider's judgement? Each step needs a short brief first.

- **1a · another provider's judge** (about US$2–5): re-judge D0's plans already on disk with a non-Claude model,
  same rubric, nothing re-planned. Needs that provider's key on the machine. Falsified if the director's gap over
  today's session falls below +0.15.
- **1b · held-out from the future** (about US$30): a new bank from work done after the memory was compiled; D0's two
  arms. Same bar.
- **2a · headroom for multi-day coherence** (about US$5–10, one arm, may run beside 1b): probes on decisions dated
  across one real multi-day thread, answered by Claude Code with its own memory. Phase 2 stops if it scores ≥ 0.85.

D17 below waits for 1a and 1b: its guard uses the same judge.

## 2. D17, component evolution on the recipes — **not run; decide first**

The person decides whether it still runs: its case was the local model's recipes (no cost per plan), which is the
kind of edge that survives Opus, so it may still pay; but its guard and judge were written for Sonnet's numbers. If it
runs, the brief stands as written.

[BRIEF-D17](BRIEF-D17.md), **cap US$40**, needs Ollama with `gemma4:12b-mlx`. Episodes come from D15's runs on disk
(`~/.config/director/d15/`). Arms R0, R1, RG and T on a new bank, and R1 and RG again on D15's bank. The result goes in
the brief and in EVOLUTION §8 step 3. If the verdict is GATED EVOLUTION TEACHES, the gate is generalised to component
kinds next (the brief's last section).

## 3. Measuring what is already built

These steps are built and tested, but not measured. Each needs a brief first (lab rule), short, since the method is
borrowed:

| step | measure | borrowed from |
|---|---|---|
| 4 · plans on the band (plugin since 0.2.x) | after one move made **on the band** and one `/director:reflect`, public-facing drafts go to the frontier and internal text stays local | D3: same probes, same verdict table; only the move's origin changes |
| 5 · the journal | a lesson drafted by `/director:reflect` from `journal_read` passes D2's held-out probes as well as one drafted from the session alone | D2's probes and judge |
| 6 · stamped views | none needed: it is a lint (tests show a stale recipe set reported after an accepted lesson) | — |

## 4. Housekeeping on the live memory

- **Stamp the live recipes once:** `python -m director.recipes stamp ~/.config/director/<memory>` (or the project's
  `.director/memory`), so the status line can tell when they go stale.
- **Build the guard's baseline** for the current live memory if `guard` reports "no baseline for this memory": the
  baseline is stored under the memory's version, so every accepted lesson needs it carried forward (Accept does this
  when the guard is enabled).
- **Lanes:** if you use `lanes.json`, check that each lane's `confidential_ok` is what you mean; the band's Move
  follows it.

## 5. Still open, needing a person

- ROADMAP step 1: a stranger's own `CLAUDE.md` compiled on their machine (time and cost from a clean clone).
- EVOLUTION steps 7–9 (Managed Agents and Antigravity lanes, the memory package): roadmap phase 4, after phase 1 —
  porting a number not yet checked outside our history only spreads it.
