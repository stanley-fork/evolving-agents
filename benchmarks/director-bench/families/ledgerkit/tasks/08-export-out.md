Add an `--out <file>` option to the existing `export` command of ledgerkit.

With `--out <file>`, `export` writes the exported text to `<file>` instead of
returning it, and returns `wrote <n> entries to <file>` (n = number of entries
exported). The file must contain exactly the text that `export` with the same
`--format` would have returned. It works with every `--format`. `--out` without
a value is a bad-input error.

Include tests, and follow the project's conventions.
