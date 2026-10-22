Add a `tag` command to ledgerkit.

`tag <entry-id> <tag>` adds `<tag>` to the tags of that entry (once: tagging
twice with the same tag changes nothing) and returns `tagged <id> <tag>`.
The change must be saved to the ledger file when run from the command line.
Tags are lower-case letters, digits and dashes, starting with a letter or digit;
anything else, a non-integer id or missing arguments are bad-input errors. An
id that does not exist is a not-found error.

Include tests, and follow the project's conventions.
