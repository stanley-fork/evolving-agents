# J0: does D0's gap hold under a judge from another provider?

**Written 2026-10-20, before any call is made.** Step 1a of [ROADMAP](../../../ROADMAP.md#next--in-phases)'s phase 1.

## Why

Every number the director publishes rests on D0: today's session **0.589**, the director **0.929**, a gap of
**+0.34** on 14 real situations ([D0](BRIEF-D0.md)). In D0 the plans were written by Claude and judged by Claude
(Sonnet 5.5). Part of that gap could be a Claude judge preferring the plans of a Claude that walked a memory written
by Claude. D0 also noted that the director's plans are longer (4,473 vs 2,645 characters on average) and the judge
saw them. Neither is measured. J0 is the cheapest measurement that could contradict D0: nothing is re-planned, only
judged again.

## What runs

- **The plans:** D0's, already on disk. Arm A (today's session) and arm T (the director): 14 cases × k = 2 = 28
  plans each. Arms A+ and P are judged only if A and T hold (the sequence, not the grid).
- **The judge:** Google's **`gemini-4-argon`**, through the Gemini API with the user's own key. Chosen over
  `gemini-3.1-pro-preview` because it costs no more (US$2 / 10 per million tokens in / out through 2026-11-30,
  against 2 / 12; it doubles from 2026-12-01, so a re-run after that date costs twice as much). Default settings,
  JSON output, one verdict per plan, one retry if the verdict does not parse.
- **The prompt:** D0's own [`judge.md`](../d0/judge.md), unchanged — the case, its checklist, the plan, blind to the
  arm. Scoring is D0's: a plan's score is items met / items; an arm's is the mean over cases of the mean over samples.
- **Privacy:** before anything leaves the machine, every term in a private list (names of private work) is replaced
  by a fixed marker (`[PRIVATE-n]`) in the situation, the checklist and the plan, the same marker each time; the
  prompt is then checked for any term left, and a leak stops the run before the call. Dry run on all 112 prompts:
  0 leaks. The list, the key, the plans and the verdicts stay in the private directory; only scores are published.
- **Code:** [`d0/rejudge.py`](../d0/rejudge.py), standard library only. One line per call; each verdict appended to
  disk as it lands; resumable.

## Spend

Estimated US$2–4 for A and T (about 1,300 tokens in and a few thousand out, thinking included, per call). The run
stops itself at **US$10**, and every call made with this key is appended to a private ledger that refuses any call
once the key's total reaches **US$50** (the user's cap for all tests with this key, 2026-10-20).

## What decides it, written before it runs

| result | verdict |
|---|---|
| T − A under Gemini **≥ +0.15** (D0's own bar) | **HOLDS** — the gap is not a Claude judge's preference; A+ and P are judged next |
| T − A under Gemini **< +0.15** | **SELF-PREFERENCE IS PART OF THE EFFECT** — published as such; phases 2–4 wait, and the next step is to find which part (length, Claude's phrasing, or the rules) |

Reported beside the verdict, without changing it: per-case wins, both judges' means for each arm, and item-level
agreement between Gemini and Sonnet on the same plans (raw and Cohen's κ). A judge that agrees with Sonnet item by
item and still moves the gap would say the disagreement is concentrated on one arm.

## What it does not answer

- The cases are still the lab's own history: that is step 1b, not this one.
- One other provider is one other judge, not a consensus; a HOLDS from Gemini rules out Claude-on-Claude preference,
  not every bias a judge can have (length among them).
- The marker replacing a private name may read differently to the judge than the name did; the checklist items
  judge actions, so this should not move a verdict, but it is a change to the input and is said here.
