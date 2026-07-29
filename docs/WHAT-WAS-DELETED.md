# What was deleted, and what replaced it

On 2026-07-28 this repository stopped being a framework and became a plugin for
the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview).
10,165 lines went with that decision.

Recording it here rather than leaving it silent, because this organisation has a
documented habit of large deletions landing without a trace and every reader
afterwards trusting a description that stopped being true months earlier.

Everything below is recoverable from git history at the commit that removed it.

## Deleted, because the SDK does it

| Removed | Lines | What does it now |
|---|---:|---|
| `evolving_agents/tools/` | 5,077 | `Read` · `Write` · `Edit` · `Bash` · `Glob` · `Grep` · `WebSearch` · `WebFetch` |
| `evolving_agents/agents/` | 1,222 | Subagents |
| `evolving_agents/agent_bus/` | 1,050 | MCP for discovery, subagents for delegation |
| `evolving_agents/providers/` | 948 | The SDK is the runtime |
| `evolving_agents/workflow/` | 802 | The agent loop |
| `evolving_agents/adapters/` | 530 | — (OpenAI-specific shims, no longer relevant) |
| `evolving_agents/monitoring/` | 459 | `total_cost_usd` on the result message, plus hooks |
| `evolving_agents/firmware/` | 77 | Permissions, and `PreToolUse` returning `permissionDecision: "deny"` |

### On `firmware/`, specifically

It is worth quoting, because for a year it was this organisation's argument
about what governance is:

```python
self.base_firmware = """
You are an AI agent operating under strict governance rules:
...
- Never use dangerous imports (os, subprocess, etc.)
"""
```

That is governance by **asking**. A model that ignores the paragraph is not
prevented from anything.

The SDK's answer is a `PreToolUse` hook returning `permissionDecision: "deny"`,
which stops the call before it runs regardless of what the model decided. The
gap between those two is the whole reason this repository was rewritten — and
the SDK closing it is exactly why we no longer ship our own version.

## Kept, in `legacy/eat/`

Not because it runs — it needs a MongoDB Atlas cluster — but because it is the
ancestry of what this repository ships today:

| Kept | Lines | Became |
|---|---:|---|
| `core/` | 1,406 | the base abstractions everything else was built on |
| `smart_library/` | 860 | skills, loaded from `.claude/` by the SDK |
| `evolution/` | 337 | `packages/agentvcs` — the evolution loop is the direct ancestor of the merge |
| `auditing/` | 274 | signed commits and provenance in `packages/agentvcs` |
| `utils/`, `config.py`, `templates/` | 260 | supporting code |

`evolution/` is the one to read if you only read one. 337 lines that tried to
close the loop between an agent changing and that change being kept, with
nothing to verify the change was an improvement. `avcs_freeze` refusing to
promote an agent whose eval fails is the same idea with the missing half added.

## Also removed

`examples/` moved to `legacy/eat/examples/` — every one of them imports a
deleted subsystem and expects a MongoDB connection string.
