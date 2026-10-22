Add a `count` command to ledgerkit.

`count <account>` returns the number of entries recorded for that account, as a
bare integer (for example `2`; `0` when the account has no entries). Calling it
without an account is a bad-input error.

Include tests, and follow the project's conventions.
