"""The director's operational memory, apart from the agent that plans with it: the format, editing and lint, the gate
a lesson passes, and its versioning. Standard library only; it never calls a model.

    memory.format   where the memory is (`find`), where its state lives (`state_dir`); pages as data:
                    statements, fields, ids
    memory.edit     apply edits to a copy (`apply_edits`), lint (`lint`, `new_problems`)
    memory.gate     a lesson's way in: propose → check → accept or reject, one directory per proposal
    memory.vcs      each accepted edit as a gated agentvcs patch (`LiveStore`)

The agent's side (planning, reflection, lanes, the interfaces) imports from here; the memory never imports from it.
The format itself is in `docs/MEMORY-FORMAT.md`; the memory model this module serves is in the repository's
`docs/EVOLUTION.md` §4.
"""

from __future__ import annotations

from .edit import apply_edits, index_add, lint, new_problems
from .format import LINK, STMT, field, find, render, state_dir, statement_ids, statements

__all__ = ["LINK", "STMT", "apply_edits", "field", "find", "index_add", "lint", "new_problems", "render",
           "state_dir", "statement_ids", "statements"]
