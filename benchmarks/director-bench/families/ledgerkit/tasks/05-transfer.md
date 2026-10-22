Add a `transfer` command to ledgerkit.

`transfer <from> <to> <amount>` moves money between two accounts: it records
one entry of minus the amount on `<from>` and one of plus the amount on `<to>`,
both with memo `transfer` (the change must be saved to the ledger file when run
from the command line), and returns
`transferred <amount> from <from> to <to>` with the amount formatted with two
decimals (e.g. `transferred 2.50 from cash to bank`). The amount must be
positive; zero, negative or malformed amounts (or missing arguments) are
bad-input errors.

Include tests, and follow the project's conventions.
