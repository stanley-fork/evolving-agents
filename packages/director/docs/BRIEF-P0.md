# P0 — does the director's walk survive being a Claude Code plugin?

**Written 2026-10-17, before anything runs.** The user asked whether the director — and the repository it lives in —
should be a Claude Code plugin (later a Codex one) rather than a harness of its own. Everything D0–D7 measured was
measured with the director *driving* Claude Code: `DIRECTOR.md` appended to the system prompt (`claude -p
--append-system-prompt`). A plugin cannot append to the system prompt; it reaches the session through a hook's
`additionalContext` or a skill the model chooses to load. If that channel loses the walk, D0's 0.929 does not carry
over and the plugin would ship a claim it has not earned.

## The one comparison

| arm | what the model gets | runs |
|---|---|---|
| **T** (D0, 2026-10-14, [ran]) | D0's T sandbox (the compiled pages under `memory/`) + `DIRECTOR.md` as appended system prompt | already paid: 28 plans, mean **0.929** |
| **P — plugin** | **the same sandbox, byte for byte**, no appended system prompt; the session loads the plugin (`--plugin-dir packages/director`), whose `SessionStart` hook puts `DIRECTOR.md`'s text into the context | 14 cases × k = 2 |

Only the channel changes, and one line of the text. The hook's text is the shipped `DIRECTOR.md` (asserted at prepare
and in `tests/test_plugin.py`). T ran with an earlier draft that differs in one line: it pointed to the memory's format
as `docs/MEMORY-FORMAT.md`, a path absent from the sandbox; the shipped text points to `memory/README.md`, which the
sandbox holds (the same file). Found by the parity assertion at prepare [ran]; recorded in `arm-P/sources.json`. If
P comes out below T, this line is one of the two things the attribution arm (T re-run today, below) separates.

Everything else is D0's: `claude -p --restricted`, tools `Read, Grep, Glob`, Claude Sonnet 5.5 through the local Claude
Code, `--max-turns 12`, `--max-budget-usd 0.60` per call, D0's 14 cases and its blind judge (Sonnet 5.5, no tools).

**Headroom** is D0's: arm A (today's session) scored 0.589, so the suite can see a loss.

**The channel is checked where it happens, not inferred from the score.** A smoke call [ran] 2026-10-17 (Haiku,
`--restricted --plugin-dir`, US$0.02) showed the hook firing (`hook_response SessionStart:startup` carrying
`DIRECTOR.md`) and the session naming `memory/Index.md` as its first file. In the run, every P call records whether
its plan cites statements (`[[Page#id]]`, a format only the walk teaches); the stream is not kept per call, so the
hook firing is asserted by the smoke and by citations, not by every transcript.

## Verdict (fixed here)

| verdict | condition |
|---|---|
| **HOLDS** | P mean ≥ 0.85 **and** T − P ≤ 0.10 |
| **WEAKER** | P mean ≥ 0.85 but T − P > 0.10 — the plugin helps, less than the harness |
| **LOST** | P mean < 0.85 — the channel does not carry the walk; the director stays a harness of its own |

Beside the verdict: per-case scores against T's, cost and turns per plan, plans citing statements (T: reported in
D0).

## What this does not settle

- **Drift.** T ran three days ago, with the earlier draft. If P comes out *below* T, T re-run today with the shipped
  text is the attribution arm, bought only then (≈ US$13): it separates the channel from drift and the one line. If P
  holds, neither can have helped P.
- **A skill instead of a hook.** The hook fires every session; a skill fires when the model picks it. Not measured.
- **The other half of the plugin.** Reflect, propose and accept (MCP tools) are the director's own functions
  (`learn.apply_edits`, `learn.lint`, `guard.positive_probe`, `memory_vcs.LiveStore`), already measured in D2–D7; the
  plugin calls them, and `tests/test_plugin.py` checks the wiring, not their effect. The targeted guard is not in the
  plugin: its cases are this lab's (D0), and a third-party project has none yet.
- **Codex.** Nothing here runs on Codex.

## Cost and stopping

28 plans at T's rate (US$0.45 each) ≈ US$13, plus the judge ≈ US$0.5. **Cap US$20**, read from each call's
`total_cost_usd`; results persist per call (`<private>/d0/arm-P/`), resumable. One design: no redesign on these data.
An account limit stops the runner without recording a result (since D7).

## Result [ran] 2026-10-17 — **HOLDS**: the walk survives the plugin channel

| | T (D0, harness) | P (plugin) |
|---|---|---|
| mean, 14 cases × k = 2 | **0.929** | **0.929** |
| plans citing statements | 28 / 28 | 27 / 28 |
| turns per plan · cost per plan | 9.8 · US$0.45 | 7.8 · US$0.46 |
| errors · unparsed verdicts | 0 · 0 | 0 · 0 |

T − P = **0.0**; per case, T is higher on 2 (c06 1.0 → 0.75, c09 1.0 → 0.5) and P on 2 (c10 0.5 → 0.75, c12 0.5 →
1.0); the other 10 tie. A (today's session) stays at 0.589. **By the table fixed above: HOLDS.** US$12.91 for the plans
+ US$0.36 for the judge, of the US$20 cap.

**Read before relying on it.**
- **The equal means are not a copy.** None of P's 28 plans is identical to any of T's; four cases moved, two each way.
  A k = 2 suite moves by 0.25 a case on one sample — the cases that moved are within that noise, in both directions.
- **One plan did not walk** (c11, one sample, no `[[…]]`); it still met its checklist. The hook fired in every call
  that walked — citations are the format only the walk teaches.
- **Two things changed at once and neither cost anything:** the channel, and the one line of text (above). P holds,
  so the attribution arm (T re-run today) is not bought.
- **Not measured:** a skill instead of the hook; another project's memory (the quickstart on third-party
  `CLAUDE.md`s, NEXT item 3); Codex.

**Consequence.** The plugin is the director's distribution: `/plugin install director --marketplace
EvolvingAgentsLabs/evolving-agents`. The harness (`director.server`) stays for what a plugin cannot hold — the local
lane as a peer and the two-lane interface.
