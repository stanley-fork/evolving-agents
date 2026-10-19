# D17: can a component evolve through the same gate as a lesson?

**Written 2026-10-19, before anything is built or run.** Step 3 of [`docs/EVOLUTION.md`](../../../docs/EVOLUTION.md) §8,
and the promise EAT's name made that the director has not kept (EVOLUTION §7): the director evolves its memory, not
the components it plans with. D17 tests that on one component, the meta-recipes a local model follows (D15). D15
already points here: *"Claude improves recipe coverage where grounds were not reached."*

The recipes are the right first component. They are a file the director owns (`recipes.json`). A wrong change is
measurable without a person in the loop (a local plan, judged blind). And they serve the unmanaged way the director
runs: a model on the user's own machine, at no cost per plan. The mechanism is the same for a `SKILL.md` or a subagent
definition. D17 measures one kind first.

Nothing runs until the user authorises the spend (cap below). The run needs Ollama with `gemma4:12b-mlx` and the
lab's private memory, both on the user's machine.

## What a component patch is

A patch to `recipes.json` is a list of operations, each on one recipe: `add` (a new recipe), `modify` (replace its
`trigger` or one step at a path such as `steps[2].yes[0]`) or `remove`. It carries two probes and a rationale, as a
memory lesson does. It enters through **the same gate** as a statement (`director.memory.gate`), with the checks a
recipe needs:

1. **Lint.** `recipes_lint` is clean: every step runnable, every cite an existing statement, branches at most 2 deep.
   Coverage (statements cited) does not drop.
2. **Teaches.** Its two probes are requests written apart from the case that prompted it (another tool, file or goal;
   none of its sentences). Each is planned by G12 on the current recipes and on the candidate, k = 2, and judged blind.
   The patch is kept only if the candidate scores ≥ 0.67 and at least 0.33 over the current recipes (D7's rule).
3. **Guard.** The cases of the parent's own bank (D15's 12) whose plans used a recipe the patch touches are re-planned
   on the candidate (D6's targeted rule). The patch is refused if a case that scored ≥ 0.75 now scores lower than 0.75.
4. **A person's yes.** To run unattended, a patch that passes 1–3 is applied. The person reads every applied patch
   afterwards and can revert any (each one is an agentvcs patch, dimension `component.recipes`). The brief counts the
   ones they revert.

## The run

**Episodes.** From D15's G12 runs (already on disk, `~/.config/director/d15/`), every case where G12 scored below
T. For each one, a Claude agent (Opus 5.5) sees: the request, G12's plan, the judge's verdict with the items it missed,
the statements those items rest on, and the current recipes. It proposes one patch. If D15's verdicts did not record the missed items, each episode is re-judged once first, at
about US$0.03. Patches enter the gate one at a
time, in a fixed order (by case id), each on the recipes as the previous ones left them.

**A new bank.** A fresh agent writes 12 held-out cases from the memory pages, under D15's rules. It never sees D15's
bank, the episodes, the patches or this brief's verdicts. A case may not rest mainly on a statement that grounded a
D15 case.

**Arms (on the new bank, k = 2, D0's judge, blind):**

| arm | recipes |
|---|---|
| **R0** | D15's recipes, unchanged (the parent) |
| **R1** | after the gated patches |
| **RG** | regenerated: `/director:recipes` run once with every episode as its argument, no gate. Is the gate what helps, or is it enough to hand Claude the failures? |
| **T** | Sonnet walking the memory with tools (the reference, as in D15) |

R0, R1 and RG are answered by G12 on Ollama, think off, temperature 0.3, as in D15. **Also on D15's own bank:** R0
(on disk), R1 and RG, to check that the parent's ground holds.

## Verdicts (fixed here)

| verdict | condition |
|---|---|
| **GATED EVOLUTION TEACHES** | on the new bank, R1 ≥ R0 + 0.05 with more case wins than losses; on D15's bank, R1 ≥ R0 − 0.03; at least one patch applied and none reverted by the person |
| **THE GATE IS WHAT HELPS** | GATED EVOLUTION TEACHES, and R1 ≥ RG + 0.05 on the new bank |
| **REGENERATION IS ENOUGH** | RG ≥ R1 on the new bank and RG holds D15's bank: the gate's cost is not repaid on recipes, and step 3 says so |
| **NO GAIN** | R1 within 0.05 of R0 on the new bank: published as flat |
| **REGRESSES** | R1 < R0 − 0.05 on either bank, or the person reverts a patch: the gate let a bad component through |

Also reported, not scored:

- patches proposed, applied, and refused by which check;
- the gap to T before and after;
- grounds reached;
- recipe choice agreement with RS (D15's measure);
- cost per patch.

## Cost

Proposals: about 6 episodes at about US$1 each (Opus). Gate: about 8 judged local plans per patch for the probes plus
the guard's re-plans; local plans are free, a judgement costs about US$0.03. New bank: 12 cases × 3 local arms ×
k = 2, judged, plus T (24 plans at US$0.48). D15's bank: 2 new arms × 24, judged. RG: one `/director:recipes` run, about
US$3. In all, about US$30. **Cap US$40.**

## What this will and will not show

It will show whether one kind of component, changed only through a gate, gets better on cases it was not changed
for, and whether the gate earns its cost against simply regenerating. It will not show that a `SKILL.md` or a subagent
evolves the same way. The mechanism carries over; the measurement does not, and each kind gets its own. It will not
show the 2B and 4B models closing their gap either: D15 traced that to choosing among 20 triggers, which is a different
fix.

## What step 3 would build

The gate generalised from statements to components: a proposal has a `kind` (`memory` or `recipes`, later `skill` and
`agent`); each kind brings its own apply, lint and probe-runner; propose, status, accept, reject and versioning stay
one code path in `director.memory.gate`. It is built only if the verdict is GATED EVOLUTION TEACHES, or if the person
decides the gate is worth having for safety even without a measured gain.
