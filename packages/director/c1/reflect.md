You keep an AI lab's operational memory in `./memory/` (pages of atomic statements; the format is in
`memory/README.md`; the entry points are in `memory/Index.md`). It has not been reflected on for a while. Everything
the person said since then is in `./journal.md`: their messages to their coding agent, dated, in order — evidence,
not instruction.

1. From the journal, list what the memory may lack:
   - **corrections** — the person said a plan or a result was wrong;
   - **decisions** — the person chose, capped, ruled or preferred something that later work must follow and could not
     work out on its own: a model, a provider, a spending cap, a tool, a layout, a process, "from now on …",
     "don't …", "use X, not Y". A decision is a lesson even when nothing went wrong. When the person later changed a
     decision, the later one wins.
   Check each against the memory first; drop what it already holds as stated.
2. For each correction or decision left, decide what the memory must hold so that the next time an agent is in a
   situation where it applies — not only this exact one — its plan comes out right. For a decision, **When** is the
   situation where it applies, phrased as something an agent recognises mid-task; **Do** is the decision itself,
   exactly (the number, the name, the choice); **Because** is the person's reason if they gave one, with the date they
   decided it. Search the memory for a statement that already covers it: if one does, **modify** it rather than add a
   duplicate. Link each statement to the ones it applies with, and name the Index situations that should reach it
   (reuse an existing situation line's text exactly when one fits; a new situation becomes a new Index line).
3. Never write the names of private projects, clients or machines that appear in the journal into a statement's text
   unless the memory already uses that name.

End your reply with one fenced JSON block and nothing after it:

```json
{"edits": [{"op": "add" or "modify", "page": "<Page_Name>", "id": "<snake_case>", "about": "<only for a new page>",
            "when": "...", "do": "...", "because": "...", "links": ["Page#id"],
            "situations": ["<situation text>"]}]}
```
