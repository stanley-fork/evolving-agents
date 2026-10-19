---
description: Write (or rewrite) the meta-recipes a local model follows to plan from this memory, then lint them
argument-hint: "[what to change, e.g. 'cover the uncited statements' or a request the local plan got wrong]"
allowed-tools: Read, Glob, Write, Bash(python3 *)
---

You maintain the director's **meta-recipes**, which turn the memory into plans that a small local model can follow.
The model never writes a plan. Code runs your recipe, and the model only:

- picks at most 2 recipes, by their one-line `trigger`;
- answers your yes/no questions.

Every word of the plan comes from your `do` steps. In D15 (`docs/BRIEF-D15.md`), gemma4:12b following recipes like
these planned at 0.806, against 0.854 for Sonnet walking the memory with tools.

1. Find the memory as the SessionStart hook does: `$DIRECTOR_MEMORY`, `.director/memory/` or `memory/`. Read its
   pages (statements with When / Do / Because / Links) and `Index.md`. If `recipes.json` exists, read it too. You are
   editing it, not starting over, unless asked to start over: $ARGUMENTS
2. Write `recipes.json` in that folder: a JSON array of at most 20 recipes, one per **type of request** this project
   gets.

   ```json
   {"id": "merge_pr", "trigger": "Committing, opening or merging a pull request, or cleaning branches",
    "steps": [
      {"do": "Merge only with authorisation given in this session and checks green; otherwise leave it to the user and say why.", "cite": "Git_And_PRs#merge_is_users_call"},
      {"ask": "Does the request say a CI check failed or is still pending?",
       "yes": [{"do": "Read the failing check's log before deciding anything.", "cite": "..."}],
       "no": []}
    ]}
   ```

   **`trigger`:**
   - at most 20 words, concrete, phrased the way a request would read;
   - triggers clearly distinct from each other.

   **`do`:**
   - one concrete action (a command, a file, a refusal, a question to the user), self-contained, faithful to its
     statement's Do and conditions;
   - at most 45 words;
   - always with a `cite` that is an existing `Page#id`.

   **`ask`:**
   - a yes/no question about a **fact readable in the request text**, such as "Does the request ask to publish to a
     public repo?";
   - never a question of judgement ("does this rule apply?"): small models answer yes to every one of those;
   - make a rule an unconditional `do` whenever it is safe to state each time the recipe applies.

   **Shape:**
   - branches at most 2 deep, about 8 asks and 25 reachable `do` steps per recipe;
   - safety, refusals and asking the user come first;
   - where the user's request conflicts with a rule, the `do` tells the user and follows the rule.

   **Coverage:** a statement that no recipe cites can never reach a plan. Cover the niche ones too.
3. Call the `recipes_lint` tool (MCP `director`), or run
   `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python3 -m director.recipes lint <memory folder>`. Fix every problem, then
   report the recipes, their triggers, and the coverage (statements cited / statements).
   Once they lint clean, stamp them with the memory version they were written from:
   `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python3 -m director.recipes stamp <memory folder>`. When a lesson is accepted
   later, the memory changes, `recipes_lint` reports the recipes `stale` and the status line says so: run this command
   again.
4. Tell the user that `plan_local` now plans with these recipes on their local model (Ollama, `DIRECTOR_LOCAL_MODEL`,
   default `gemma4:12b-mlx`). A plan that comes out wrong is a reason to run this command again with that request
   as the argument.
