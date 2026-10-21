# F1: does the director's gap hold on situations from after its memory was compiled?

**Written 2026-10-21, before any case is written or any arm runs.** Step 1b of
[ROADMAP](../../../ROADMAP.md#next--in-phases)'s phase 1. Step 1a ([J0](BRIEF-J0.md)) showed that D0's gap is not a
Claude judge's preference. D0's other weakness remains: its 14 situations were written from the lab's history
**up to** the day the memory was compiled, by a session that knew the rules. A memory can look good on the past
it was built from. F1 asks the same question on situations that had not happened yet when the memory was frozen.

## The two arms — D0's, frozen

The sources and the compiled memory are **D0's, byte for byte**, as prepared on 2026-10-14: arm A's sandbox (the
workspace `CLAUDE.md`, lora-kernel's `CLAUDE.md`, agents and skills, the memory files, `MEMORY.md` in the system
prompt) and arm T's sandbox (the same sources compiled into pages, `DIRECTOR.md` as the system prompt). The runner
copies them and asserts every file's hash equals D0's. Nothing learned after 2026-10-14 reaches either arm, so both
see the same rules and neither has seen the situations. Same model (Sonnet 5.5), tools (`Read, Grep, Glob`), turns,
per-call budget and k = 2 as D0.

Sequence, not a grid: **A first** (headroom), T only if A leaves room. A+ and P are not run.

## The cases — written by a fresh session, screened by code

**Source of situations:** the lab's public record from **2026-10-15 to 2026-10-20** — the 28 pull requests opened in
that window across the organisation's public repositories (titles and descriptions; collected 2026-10-21). Only
things that happened after the freeze.

**Source of rules:** arm A's sandbox only — the rules as they stood on 2026-10-14.

**Writer:** a fresh Claude Opus 5.5 session (`claude -p`, tools `Read, Grep, Glob, Write`), given the record, A's
sandbox, D0's 14 situations (to avoid repeating them) and D0's case rules: a task in flight, never a quiz; 2–3
checklist items about **actions**, never wording; private names as placeholders. It writes **18 candidates**. For
each checklist item it gives the source file and a **verbatim quote** of the rule that should change the plan, and
for each case the pull request it comes from. It never sees the compiled memory.

**Screen, in code, in the writer's order, before any arm runs:**
1. **Parity:** every item's quote is found verbatim (whitespace-normalised) in the file it names in A's sandbox —
   so the rule was written down before the freeze, and both arms have it.
2. **After the freeze:** the pull request it cites exists in the record (opened 2026-10-15 or later).
3. **No private names:** no term of the private list appears in the case.
4. **Not a D0 case:** word overlap of its situation with every D0 situation below 0.5 (Jaccard on words of 4+
   letters).

The **first 14** candidates that pass are the bank, `f1/cases.jsonl` (public, placeholders only). Fewer than 12
passing: the writer runs once more with the failures listed; still fewer, F1 stops and says so.

## Judges

D0's `judge.md`, unchanged, blind to the arm: **Claude Sonnet 5.5** (D0's judge) and **gemini-4-argon** (J0's), the
latter with private terms replaced and checked before every call, as in J0.

## What decides it, written before it runs

| result | verdict |
|---|---|
| A ≥ 0.85 under Sonnet | **STOP — no headroom**: on these situations today's session already plans by the rules; T is not run |
| T − A ≥ +0.15 under **both** judges | **HOLDS ON THE FUTURE** |
| T − A ≥ +0.15 under one judge only | **SPLIT** — published as such, with both numbers |
| T − A < +0.15 under both | **DOES NOT HOLD ON THE FUTURE** — D0's gap was partly the past it was compiled from; phases 2–4 wait |

Reported beside it: per-case wins, both judges' means and item agreement, cost per plan, and how many T plans cite
the statements they follow.

## Spend

Claude, through `claude -p`: about US$25 (cases ~US$4, arm A ~US$4, arm T ~US$13, Sonnet judge ~US$3); the runner
stops at **US$30**. Gemini: under US$0.50, inside the key's US$50 ledger. Nothing runs on a local model.

## Privacy

The record is public. The sandboxes, plans, verdicts and the private-term list stay in `~/.config/director/f1/`;
the cases file and the scores are what is published.

## What it does not answer

- The writer is still Claude, and knows what a rule is for; it does not know which arm will be tested, and it never
  sees the compiled memory. A bank written by a person would be stronger; that is step 1c's outside user.
- Six days is a short future. A second future bank, from work after this one, would repeat the check.
- Rules created after the freeze are excluded by construction (screen 1), so F1 measures whether old rules transfer
  to new situations, not whether the memory keeps up with new rules — that is phase 2's question.

## Result [ran] 2026-10-21 — **HOLDS ON THE FUTURE**

**The bank.** 18 candidates by a fresh Opus (US$3.05, 38 turns); all 18 passed the code screen; the first 14 are
[`f1/cases.jsonl`](../f1/cases.jsonl), committed before any arm ran. Arms copied from D0 with every file's hash equal
(A 57 files, T 21).

| arm | Sonnet 5.5 | gemini-4-argon | item agreement | κ | cost / plan | turns | plans citing statements |
|---|---|---|---|---|---|---|---|
| A — today's session (frozen 2026-10-14) | **0.363** | 0.512 | 0.724 | 0.45 | US$0.06 | 1.5 | 0 / 28 |
| T — the director (frozen 2026-10-14) | **0.839** | 0.798 | 0.931 | 0.78 | US$0.50 | 8.6 | 28 / 28 |

- **Headroom:** A 0.363, far below 0.85 — GO to T.
- **T − A: +0.476 under Sonnet, +0.286 under Gemini**, both above +0.15. Per case (Sonnet) T 11 wins, A 2, 1 tie.
  **HOLDS ON THE FUTURE.**
- **Against D0:** both arms score lower on situations from after the freeze (A 0.589 → 0.363, T 0.929 → 0.839), and the
  gap is wider under Sonnet (+0.34 → +0.48) and a little narrower under Gemini (+0.30 → +0.29). The director transfers;
  today's session loses more on new situations than the director does.
- **Where T still misses (Sonnet):** f08 *headroom before coherence* 0.25; f01 *read the path before blaming the model*,
  f03 *account-limit errors scored as plans*, f12 *the whole gate before the PR* 0.5. All four rest on rules in the
  workspace `CLAUDE.md` / lora-kernel's `CLAUDE.md` about instruments; the walk reached the page or not — that split is
  not measured here.
- **Gemini is again more lenient**, and more on A (0.363 → 0.512) than on T (0.839 → 0.798), which narrows its gap;
  agreement with Sonnet is lower on A (κ 0.45) than on T (κ 0.78).
- **Length:** T's plans are longer, as in D0 (4,495 vs 2,832 characters); a judge's length preference is still not
  ruled out by either judge.
- **One instrument error:** T's f11 sample 0 hit the per-call budget mid-walk (US$0.69, empty plan); retried once with
  the same budget as the runner resumes, it planned (US$0.49) and was judged like the rest. No other errors; 0 unparsed
  verdicts; 0 private terms sent.
- **Spent:** Claude US$19.99 of the US$30 cap (cases 3.05; A 1.73, its judge 0.31; T 14.50 with the cut call, its judge 0.40); Gemini US$0.32 (key total
  US$0.84 of 50).

**What it rules out, and what not.** D0's gap is not the past the memory was compiled from: on situations that had not
happened yet, judged by two providers, the director keeps a gap of +0.29 to +0.48. Still open: a bank written by
someone other than Claude (step 1c), length as a shared judge bias, and new rules the frozen memory never had — that is
phase 2.
