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

## Result [ran] 2026-10-20 — **HOLDS**

| arm | Sonnet 5.5 (D0) | gemini-4-argon (J0) | item agreement | κ |
|---|---|---|---|---|
| A — today's session | 0.589 | **0.696** | 0.857 | 0.69 |
| A+ — told to walk its files | 0.768 | 0.857 | 0.911 | 0.71 |
| P — the plugin | 0.929 | 0.946 | 0.982 | 0.85 |
| T — the director | 0.929 | **1.000** | 0.929 | — (Gemini met every item) |

- **T − A under Gemini: +0.304** (Sonnet: +0.339), above the +0.15 bar; per case T 7 wins, A 0, 7 ties. **HOLDS.**
- **The order is the same under both judges:** A < A+ < P ≤ T. The attribution arm keeps its meaning: T − A+ is
  +0.143 under Gemini (Sonnet +0.161), T 2 wins, A+ 0.
- **Gemini is more lenient, on both sides.** Items Gemini met and Sonnet did not: 7 in A (cases c03, c12, c13), 4 in T
  (c10, c12); the reverse, 1 and 0. c12 is the case D0 had noted for the judge's strictness (the right command, then
  waiting for the user's go). The leniency lifts A more than T only because T is at the ceiling under Gemini: the gap
  can show no larger there, so +0.304 is a lower bound for this judge.
- κ is undefined for T because Gemini never said "not met" there; raw agreement is 0.929.
- 112 verdicts, 0 unparsed, 0 retries needed; private terms replaced before every call, 0 leaks. **Spent US$0.51**
  (key total US$0.51 of 50).

**What it rules out, and what not.** A Claude judge preferring the plans of a Claude that walked Claude's memory does
not explain D0's gap: another provider's judge, blind to the arm, gives the same order and a gap twice the bar. It does
not rule out what any judge may share (length: T's plans are still longer), and the cases are still the lab's own
history — step 1b.
