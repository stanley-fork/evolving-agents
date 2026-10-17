---
description: Compile this project's instructions (CLAUDE.md, AGENTS.md, auto-memory, agents, skills) into the director's memory of linked atomic statements
argument-hint: "[extra files or folders to include]"
allowed-tools: Bash(python3 *), Bash(git *), Read, Glob
---

Compile this project's instructions into the director's memory. From the next session on, the director's
SessionStart hook has every plan walk it.

1. Find the sources, and read nothing the user did not agree to:
   - `CLAUDE.md` / `AGENTS.md` at the project root and in its subdirectories (not under `node_modules`, `.git`,
     `.director`);
   - Claude Code's auto-memory for this project, if it exists:
     `~/.claude/projects/<the project's absolute path with every non-alphanumeric character replaced by "-">/memory`;
   - any extra files or folders named here: $ARGUMENTS
2. Show the user the list and tell them: the compile sends these files to Claude (Opus by default), takes about 15
   minutes and costs about US$5 for ~60 files; the memory is written to `.director/memory/`. Ask before going on.
3. If this is a git repository and `.director/` is not ignored, add `.director/` to `.git/info/exclude` (local, never
   committed) unless the user says the memory should be shared through the repository.
4. Run, in the background, from the project root:

   ```
   PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python3 -m director.compile \
       --claude-md <each CLAUDE.md / AGENTS.md> --memory-dir <the auto-memory dir, if any> --extra <each extra> \
       --out .director/memory --config .director/server.json --workdir .
   ```

5. Report what it printed: pages, statements, cost, and the lint problems (if any, show them; they are the compile's,
   and `memory_lint` re-checks them). Tell the user the walk starts with their next session, and that
   `/director:reflect` after a task turns a correction into a gated edit of this memory.
