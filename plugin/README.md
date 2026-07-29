# Evolving Agents — a plugin for the Claude Agent SDK

The SDK already runs the loop, ships the tools, manages context, and enforces
permissions. This plugin does not re-implement any of that. It adds the part the
SDK leaves open: **what happens to an agent's history once it has more than one.**

## What the SDK gives you, and where it stops

`fork` branches a session. Nothing merges the branches back. There is no diff
between two sessions, nothing refuses to promote an agent whose eval fails, and
the transcript is plain JSONL that anyone can hand-edit. Sessions also persist
the *conversation*, not the agent — the skills, subagents, model and goal that
produced it are not versioned alongside it.

| | Agent SDK | This plugin |
|---|---|---|
| Branch a session | `fork` | — |
| Compare two sessions | — | `avcs_diff` |
| Rejoin two branches | — | `avcs_merge`, with a `--reconcile` seam for the non-code dimensions |
| Refuse to ship a regression | — | `avcs_freeze`, which fails unless the eval passes |
| Prove a transcript wasn't edited | — | Ed25519-signed commits |
| Keep the full trace past compaction | summarised away | archived by the `PreCompact` hook |

## Install

```python
from claude_agent_sdk import query, ClaudeAgentOptions

options = ClaudeAgentOptions(
    plugins=[{"type": "local", "path": "/path/to/evolving-agents/plugin"}],
)
```

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

const messages = query({
  prompt: "…",
  options: { plugins: [{ type: "local", path: "/path/to/evolving-agents/plugin" }] },
});
```

Confirm it loaded by reading `plugins` on the init system message.

## What it installs

**An MCP server** (`agentvcs`) exposing the history as tools the agent can call
on itself mid-loop: `avcs_log`, `avcs_diff`, `avcs_merge`, `avcs_freeze`,
`avcs_rollback`, `avcs_replay`, `avcs_branch`, `avcs_recall`, `avcs_eval`, and
more. Zero runtime dependencies — standard library only, and there is a test
that fails if that ever stops being true.

**Three hooks**, none of which can block your agent:

- `PreCompact` copies the transcript to `.agentvcs/sdk/transcripts/` before
  compaction summarises it. This is the only point where fidelity is lost and
  cannot be recovered afterwards.
- `SubagentStop` records which subagents ran, which is the swarm dimension a
  session transcript flattens.
- `SessionStart` notes the session so later commits can be traced back to it.

Every hook exits 0 on any input. A full disk, a read-only checkout or a
malformed event is recorded and skipped, never raised into your session.

## Where the data goes

`.agentvcs/sdk/` in your working directory — `events.jsonl`, `swarm.jsonl`, and
`transcripts/`. Plain files, readable without this plugin installed.

## Scope

This plugin targets the Agent SDK and nothing else. The organisation's work on
decode-time constraint (`token-trie`) and on model-internals research
(`sleep-harness`) needs logit and weight access that an API-backed SDK does not
expose; those live in their own repositories and are not part of this.
