# Runbook: what runs on the lab's machine

**Written 2026-10-19, at the hand-off from the cloud session to the user's machine.** Everything that could be built and
tested without the lab's private memory is merged on `main`: EVOLUTION steps 1, 4, 5 and 6 are built; steps 2 and 3
have their briefs. What is left needs the private memory (`~/.config/director/`), Ollama, or the user's spend, and is
listed here in the order it should run. Each item names its brief, its cap and where its result goes.

## 0. Before anything

```bash
git pull origin main
cd packages/director && python -m pytest tests -q          # 73 passed, 3 skipped expected
claude plugin test . && claude plugin validate .           # the band: 6 pass
```

- **The 3 skipped tests need the real agentvcs CLI** (the Rust one, from its own repository) on PATH or in `$AGENTVCS`.
  They cover the versioned Accept. The frozen Python copy in `packages/agentvcs/` is out of date with the current CLI:
  with it installed, those 3 fail identically on `main` and on every branch since. Run them with the real CLI once.
- **Reinstall the plugin** so Claude Code picks up the band, plans and journal: `/plugin install director
  --marketplace EvolvingAgentsLabs/evolving-agents` (or `/plugin` → update). Not yet checked: that a Claude Code
  without plugin UI modules ignores the `modules` key in `hooks/hooks.json` and still runs the SessionStart hook. If
  you have an older Claude Code at hand, open a session with the plugin and confirm the walk still reaches the context.

## 1. D16, the directory of lanes (tonight)

[BRIEF-D16](BRIEF-D16.md), **cap US$75**. Its "Run" box lists the inputs to prepare first, in order: the fact sheet,
the bank (18 cases, then a read-through), arms P and D, the parity check, and `d16/run_d16.py` in `d15/`'s shape. Then
measurements 1–4 against the verdict table, unchanged. The result goes in the brief as `## Outcome`, flat if flat, and
in EVOLUTION §8 step 2.

## 2. D17, component evolution on the recipes

[BRIEF-D17](BRIEF-D17.md), **cap US$40**, needs Ollama with `gemma4:12b-mlx`. Episodes come from D15's runs on disk
(`~/.config/director/d15/`). Arms R0, R1, RG and T on a new bank, and R1 and RG again on D15's bank. The result goes in
the brief and in EVOLUTION §8 step 3. If the verdict is GATED EVOLUTION TEACHES, the gate is generalised to component
kinds next (the brief's last section).

## 3. Measuring what is already built

These steps are built and tested, but not measured. Each needs a brief first (lab rule), short, since the method is
borrowed:

| step | measure | borrowed from |
|---|---|---|
| 4 · plans on the band | after one move made **on the band** and one `/director:reflect`, public-facing drafts go to the frontier and internal text stays local | D3: same probes, same verdict table; only the move's origin changes |
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
- EVOLUTION steps 7–9 (Managed Agents and Antigravity lanes, the memory package): after D16, since the directory is
  where a managed lane's region and clearance are recorded.
