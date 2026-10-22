Users keep asking how much money is sitting in a given account. Give the
ledgerkit CLI a command named `balance` for that.

It takes exactly one argument, the account name, and answers with the account's
net amount (all of its entries added up) in the shape
`balance <account>: <amount>` with two decimals, e.g. `balance bank: 9.99`.
Accounts nobody has used yet report `0.00`. No account name means bad input.

Please add tests and stick to how this codebase does things.
