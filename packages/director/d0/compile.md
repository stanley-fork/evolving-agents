You are compiling an operational memory. The directory `sources/` holds everything a coding agent of this
lab is given about how the lab works: instruction files (`CLAUDE.md`), memory files (`memory/`, indexed by
`memory/MEMORY.md`), agent and skill definitions. Read all of it.

Write the lab's **operational** knowledge — its practices, rules, procedures, lessons and decisions about
how work is done — into `out/` as pages of atomic statements, in the format below. Facts about projects
(what they are, results) go in only where an operational statement needs them as support.

Rules:
- Use **only** what the sources say. Add nothing from your own knowledge; drop nothing operational.
- One statement = one practice. Split compound rules. Keep the user's decisions and the reasons/evidence.
- **When** must be a situation an agent would recognise in the middle of a task (e.g. "about to commit a
  change to a document in a bilingual repo"), not a topic name.
- **Do** must be a concrete action. **Because** carries the reason or the incident.
- Link statements that apply together or next, across pages.
- Finally write `out/Index.md`: one line per situation an agent may be in, pointing to entry statements.
  Cover every statement from at least one situation.

Format of a page (`out/<Page_Name>.md`):

```markdown
# Page_Name
One line saying what the page covers.

## #statement_id
**When:** …
**Do:** …
**Because:** …
**Links:** [[Other_Page#other_id]] · …
```

Format of `out/Index.md`:

```markdown
# Index
- <situation> → [[Page#id]], [[Page#id]]
```

When done, reply with one line: the number of pages and statements written.
