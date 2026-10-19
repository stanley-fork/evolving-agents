// The director's band and status line. The status line names the memory and what waits for a decision; the band
// above the prompt shows a plan waiting to run (each task's lane; Run / Move / Cancel), a proposed lesson (its edits,
// its check, Diff / Accept / Reject) and the delegations not yet brought back. Both read director/ui_state.py and stay silent where the project has no director memory.
import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { Snapshot } from '../types'

const snap = atom({ plugin: 'director', key: 'snap' } as const, null)
const diff = atom({ plugin: 'director', key: 'diff' } as const, null)
const busy = atom({ plugin: 'director', key: 'busy' } as const, null)
const message = atom({ plugin: 'director', key: 'message' } as const, null)
const notes = atom({ plugin: 'director', key: 'notes' } as const, [] as string[])

const TOOL_PREFIX = 'mcp__plugin_director_director__'

export function statusText(s: Snapshot | null): string | undefined {
  if (!s || !s.memory) return undefined
  if (s.error) return `director · ${s.error.slice(0, 60)}`
  const n = s.statements ?? 0
  const parts = [`director · ${n} statement${n === 1 ? '' : 's'}`, `${s.accepted ?? 0} learned`]
  const waiting = s.proposals?.length ?? 0
  if (waiting) parts.push(`${waiting} to decide`)
  const plans = s.plans?.length ?? 0
  if (plans) parts.push(`${plans} plan${plans === 1 ? '' : 's'} waiting`)
  const out = s.delegations?.length ?? 0
  if (out) parts.push(`${out} delegated`)
  if (s.recipes_stale) parts.push('recipes stale')
  return parts.join(' · ')
}

type Result = { error?: string } & Record<string, unknown>

let poll: { cancel: () => void } | null = null

async function run($: EngineInterface, args: string[]): Promise<Result> {
  const root = await $.session.root()
  try {
    const r = await $.process.run(['python3', `${$.plugin.root}/director/ui_state.py`, ...args], {
      cwd: root,
      env: { CLAUDE_PROJECT_DIR: root },
      timeoutMs: args[0] === 'fetch' ? 600_000 : 150_000,
    })
    return JSON.parse(r.stdout.trim().split('\n').pop() || '{}') as Result
  } catch (err) {
    return { error: `ui_state.py failed: ${String(err).slice(0, 200)}` }
  }
}

async function refresh($: EngineInterface): Promise<void> {
  const s = (await run($, ['status'])) as unknown as Snapshot
  await update($, snap, () => s)
  $.ui.status(statusText(s))
  const checking = (s.proposals ?? []).some(p => p.check === 'RUNNING')
  if (checking && !poll) {
    poll = $.clock.every(20_000, () => void refresh($))
  } else if (!checking && poll) {
    poll.cancel()
    poll = null
  }
}

// An action from a button: run it, say how it went, tell the session on the next prompt, and re-read the state.
async function act($: EngineInterface, args: string[], note: string | ((r: Result) => string)): Promise<void> {
  const what = args.join(' ')
  await update($, busy, () => what)
  const r = await run($, args)
  const text = r.error ? `${what}: ${r.error}` : `${what}: done`
  await update($, message, () => text)
  if (!r.error) await update($, notes, xs => [...xs, typeof note === 'string' ? note : note(r)])
  await update($, busy, () => null)
  $.ui.toast(`director: ${text}`)
  await refresh($)
}

// What the session needs to hear after Run: where each task went, and which are its own to do now.
export function runSummary(r: Result): string {
  const results = (r.results ?? []) as { id: string; lane: string; status?: string; error?: string }[]
  return results
    .map(x =>
      x.error
        ? `${x.id} failed on ${x.lane} (${x.error})`
        : x.lane === 'frontier'
          ? `${x.id} is yours to do now`
          : `${x.id} delegated to ${x.lane}`,
    )
    .join('; ')
}

async function showDiff($: EngineInterface, id: string): Promise<void> {
  const r = await run($, ['diff', id])
  await update($, diff, () => ({ id, text: String(r.diff ?? r.error ?? '') }))
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    const started = await next(e)
    await refresh($)
    return started
  })

  on('turn.complete', async ($, e, next) => {
    const done = await next(e)
    await refresh($)
    return done
  })

  on('tool.call', async ($, e, next) => {
    const result = await next(e)
    if (e.tool.startsWith(TOOL_PREFIX)) void refresh($)
    return result
  }).catch(($, e, next) => next(e))

  // What the person did on the band reaches the model with their next prompt, so the session's picture stays true.
  on('prompt.submit', async ($, e, next) => {
    const pending = await read($, notes)
    if (!pending.length || e.command) return next(e)
    await update($, notes, () => [])
    return next({ ...e, context: [...(e.context ?? []), `On the director's band, the person: ${pending.join('; ')}.`] })
  }).catch(($, e, next) => next(e))

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const s = await read($, snap)
    const proposals = s?.proposals ?? []
    const delegations = s?.delegations ?? []
    const plans = s?.plans ?? []
    if (e.props.hasSurvey || !s?.memory || (!proposals.length && !delegations.length && !plans.length)) return next(e)

    const { Box, Button, Text, Code } = $.ui.resolve(e)
    const shown = await read($, diff)
    const working = await read($, busy)
    const said = await read($, message)
    const p = proposals[0]
    const plan = plans[0]

    return (
      <Box flexDirection="column">
        {plan && (
          <Box flexDirection="column">
            <Text bold>Director · a plan to run{plans.length > 1 ? ` (1 of ${plans.length})` : ''}</Text>
            {plan.request ? <Text>{plan.request}</Text> : null}
            {plan.tasks.map(t => (
              <Box key={`task-${plan.id}-${t.id}`} flexDirection="row" gap={1}>
                <Text>
                  <Text bold>{t.lane}</Text> {t.task}
                  {t.confidential ? <Text dimColor> (confidential)</Text> : null}
                </Text>
                {t.movable ? (
                  <Button
                    key={`move-${plan.id}-${t.id}`}
                    label="Move"
                    onPress={() =>
                      act($, ['plan-move', plan.id, t.id], r => `moved task ${t.id} from ${t.lane} to ${String(r.to)}`)
                    }
                  />
                ) : null}
              </Box>
            ))}
            <Box flexDirection="row" gap={1}>
              <Button
                key={`run-${plan.id}`}
                label="Run"
                variant="primary"
                onPress={() => act($, ['plan-run', plan.id], r => `ran plan ${plan.id}: ${runSummary(r)}`)}
              />
              <Button
                key={`cancel-${plan.id}`}
                label="Cancel"
                onPress={() => act($, ['plan-cancel', plan.id], `cancelled plan ${plan.id} (nothing was delegated)`)}
              />
            </Box>
          </Box>
        )}
        {p && (
          <Box flexDirection="column">
            <Text bold>Director · a lesson to decide{proposals.length > 1 ? ` (1 of ${proposals.length})` : ''}</Text>
            {p.rationale ? <Text>{p.rationale}</Text> : null}
            <Text dimColor>{p.edits.join(', ')}</Text>
            <Text dimColor>
              {p.check === null ? 'not checked' : `check: ${p.check}`}
              {p.lint ? ` · ${p.lint} lint problems` : ''}
            </Text>
            {!p.acceptable ? <Text color="warning">Cannot be accepted: {p.why_not}</Text> : null}
            {shown && shown.id === p.id ? <Code source={shown.text || '(no changes)'} format="diff" /> : null}
            <Box flexDirection="row" gap={1}>
              <Button
                key={`diff-${p.id}`}
                label={shown && shown.id === p.id ? 'Hide diff' : 'Diff'}
                onPress={() =>
                  shown && shown.id === p.id ? update($, diff, () => null) : showDiff($, p.id)
                }
              />
              {p.acceptable ? (
                <Button
                  key={`accept-${p.id}`}
                  label="Accept"
                  variant="primary"
                  onPress={() => act($, ['accept', p.id], `accepted proposal ${p.id} (now the live memory)`)}
                />
              ) : null}
              <Button
                key={`reject-${p.id}`}
                label="Reject"
                onPress={() => act($, ['reject', p.id], `rejected proposal ${p.id} (the memory is unchanged)`)}
              />
            </Box>
          </Box>
        )}
        {delegations.map(d => (
          <Box key={`lane-${d.id}`} flexDirection="row" gap={1}>
            <Text>
              {d.lane}: {d.id} <Text dimColor>{d.status}</Text>
            </Text>
            <Button
              key={`refresh-${d.id}`}
              label="Status"
              onPress={() => act($, ['refresh', d.id], `asked lane ${d.lane} for the status of ${d.id}`)}
            />
            {d.done ? (
              <Button
                key={`fetch-${d.id}`}
                label="Fetch"
                onPress={() => act($, ['fetch', d.id], `fetched delegation ${d.id} from lane ${d.lane}`)}
              />
            ) : null}
          </Box>
        ))}
        {working ? <Text dimColor>working: {working}…</Text> : said ? <Text dimColor>{said}</Text> : null}
      </Box>
    )
  })
}
