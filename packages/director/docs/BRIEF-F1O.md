# F1O: is there room for the director on new situations when Opus plans?

**Written 2026-10-21, before any F1O plan runs.** With Opus 5.5 planning, the session alone already followed D0's
written rules 0.964 ([D0O](BRIEF-D0O.md)) and kept a thread's decisions 0.80 ([C2](BRIEF-C2.md)). The one advantage
not yet tested with Opus is [F1](BRIEF-F1.md)'s: situations from **after** the memory was frozen, where Sonnet's
session scored only 0.363 and the director 0.839. F1O asks the cheapest question first — is there room?

## The arm

**AO = F1's arm A, byte for byte** (D0's frozen sources, hash checked), planned by **Opus 5.5**, on F1's 14 cases,
k = 2, tools `Read, Grep, Glob`, per-call budget US$2.50. Judges: **Opus 5.5** and **gemini-4-argon** (private terms
replaced before each call).

TO (F1's arm T with Opus) is **not** part of this brief: if AO leaves room, its cap and its run are the person's next
decision.

## What decides it, written before it runs

| AO (Opus judge) | verdict |
|---|---|
| **≥ 0.85** | **STOP — no headroom with Opus**: on new situations too, Opus alone plans by the rules written before them; no advantage measured so far survives Opus except cost |
| **< 0.85** | **ROOM** — F1's TO with Opus is worth running; its brief section and cap come next |

## Spend

From D0O's AO (US$0.49 per plan): about US$14 for 28 plans plus about US$1 of judges. **Cap US$20**, set by the person
on 2026-10-21. Gemini inside the key's US$50 ledger.

## Result [ran] 2026-10-21 — **ROOM under Opus, none under Gemini**

| arm | Opus 5.5 judge | Gemini judge | cost / plan | turns |
|---|---|---|---|---|
| F1's A, Sonnet planning (F1) | 0.363 | 0.512 | US$0.06 | 1.5 |
| **AO — F1's A, Opus planning** | **0.780** | **0.857** | US$0.43 | 4.8 |

- **AO = 0.780 under Opus: below 0.85, ROOM** by the rule written above. 17 of 28 plans meet every item. Weakest:
  f01 *read the path before blaming the model* 0.17; f08, f10, f12 0.5.
- **Under Gemini AO is 0.857**, at the ceiling: a +0.15 gain would need more than 1.0. **TO cannot meet "≥ +0.15
  under both judges"**; the best it can return is SPLIT. Said here before deciding whether to spend on it.
- 28 plans, 0 errors, 0 unparsed verdicts. **Spent US$12.86 of the US$20 cap**; Gemini US$0.14.

**Reading across the Opus runs:** D0O 0.964, C2 0.80 (session) / 0.85 (director), F1O 0.78–0.86. With Opus planning,
the session already follows written rules and recalls decisions most of the time; the room left is small and the two
judges disagree about how much there is. What is still measured in the director's favour with Opus is cost (C2,
−34 % per plan).
