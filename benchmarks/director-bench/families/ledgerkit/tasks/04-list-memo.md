Add two options to the existing `list` command of ledgerkit.

- `--memo <text>`: show only the entries whose memo contains `<text>`
  (case-insensitive).
- `--limit <n>`: show only the last `<n>` entries (after any filtering); `<n>`
  must be a positive integer, anything else is a bad-input error.

Both combine with each other and with the optional account argument
(`list bank --memo rent --limit 2`). The line format is unchanged; when nothing
matches the output is `no entries`, as for an empty listing. An option without
a value is a bad-input error.

Include tests, and follow the project's conventions.
