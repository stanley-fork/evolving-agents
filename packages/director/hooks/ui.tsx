// The director's band and status line. The status line names the memory and what waits for a decision; the band
// above the prompt shows a plan waiting to run (each task's lane; Run / Move / Cancel), a proposed lesson (its edits,
// its check, Diff / Accept / Reject) and the delegations not yet brought back. Both read director/ui_state.py and stay silent where the project has no director memory.
import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { Snapshot, Step } from '../types'

const snap = atom({ plugin: 'director', key: 'snap' } as const, null)
const diff = atom({ plugin: 'director', key: 'diff' } as const, null)
const busy = atom({ plugin: 'director', key: 'busy' } as const, null)
const message = atom({ plugin: 'director', key: 'message' } as const, null)
const notes = atom({ plugin: 'director', key: 'notes' } as const, [] as string[])
const steps = atom({ plugin: 'director', key: 'steps' } as const, [] as Step[])
const verbose = atom({ plugin: 'director', key: 'verbose' } as const, true)

const TOOL_PREFIX = 'mcp__plugin_director_director__'

export function statusText(s: Snapshot | null, live: Step[] = [], now = Date.now()): string | undefined {
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
  if (s.to_reflect) parts.push(`${s.to_reflect} to reflect`)
  const lanes = laneText(live, s, now)
  return lanes ? `${parts.join(' · ')} │ ${lanes}` : parts.join(' · ')
}

const MEMORY_PATH = /\/\.director\/memory\/([^/]+?)\.md$/

// One tool call, seen live: which lane it belongs to and a short label. The director's own steps are its MCP tools
// and the walk (a read of one of its memory's pages); everything else the session does is the frontier lane, and a
// plan made on the local model (plan_local) or a task sent to a lane other than the frontier is the local lane.
export function classify(tool: string, args: Record<string, unknown>): { lane: Step['lane']; label: string } {
  const str = (k: string) => (typeof args[k] === 'string' ? (args[k] as string) : '')
  const path = str('file_path') || str('path')
  const page = path.match(MEMORY_PATH)
  if (page) return { lane: 'director', label: `walk → ${page[1]}` }
  if ((tool === 'Grep' || tool === 'Glob') && path.includes('/.director/memory'))
    return { lane: 'director', label: `search memory: ${str('pattern')}`.slice(0, 80) }
  if (tool.startsWith(TOOL_PREFIX)) {
    const name = tool.slice(TOOL_PREFIX.length)
    if (name === 'plan_local') return { lane: 'local', label: `plan_local: ${str('request')}`.slice(0, 80) }
    if (name === 'delegate') return { lane: 'local', label: `→ ${str('lane')}: ${str('task')}`.slice(0, 80) }
    const detail = str('rationale') || str('kind') || str('id') || str('request')
    return { lane: 'director', label: `${name}${detail ? `: ${detail}` : ''}`.slice(0, 80) }
  }
  const base = (f: string) => f.split('/').pop() || f
  const label =
    tool === 'Bash'
      ? str('description') || str('command').split('\n')[0]
      : tool === 'Read' || tool === 'Edit' || tool === 'Write'
        ? `${tool} ${base(path)}`
        : tool === 'Grep' || tool === 'Glob'
          ? `${tool} ${str('pattern')}`
          : tool === 'Agent'
            ? `subagent: ${str('description')}`
            : tool.startsWith('mcp__')
              ? tool.split('__').slice(-1)[0]
              : tool
  return { lane: 'frontier', label: `${tool === 'Bash' ? '$ ' : ''}${label}`.slice(0, 80) }
}

function ago(step: Step, now: number): string {
  const secs = Math.round(((step.ms ?? now - step.t) as number) / 1000)
  return step.state === 'running' ? `${secs}s…` : step.state === 'error' ? 'failed' : `${secs}s`
}

// The status line's live half: what each lane is doing now, or did last.
export function laneText(live: Step[], s: Snapshot | null, now = Date.now()): string {
  const last = (lane: Step['lane']) => [...live].reverse().find(x => x.lane === lane)
  const f = last('frontier')
  const l = last('local')
  const out = (s?.delegations ?? []).find(d => !d.done)
  const local = l ? `${l.label} (${ago(l, now)})` : out ? `${out.lane}: ${out.id} (${out.status})` : 'idle'
  if (!f && !l && !out) return ''
  return `frontier ▶ ${f ? `${f.label} (${ago(f, now)})` : 'idle'} │ local ▶ ${local}`
}

type Result = { error?: string } & Record<string, unknown>

let poll: { cancel: () => void } | null = null
let tick: { cancel: () => void } | null = null

// The status line again from what is already known (no subprocess): called on every step and, while a step runs,
// every second, so the elapsed time moves.
async function redraw($: EngineInterface): Promise<void> {
  const live = await read($, steps)
  $.ui.status(statusText(await read($, snap), live))
  const running = live.some(x => x.state === 'running')
  if (running && !tick) tick = $.clock.every(1_000, () => void redraw($))
  else if (!running && tick) {
    tick.cancel()
    tick = null
  }
}

async function record($: EngineInterface, step: Step): Promise<void> {
  await update($, steps, list => [...list, step].slice(-40))
  await redraw($)
}

async function finish($: EngineInterface, t: number, state: Step['state']): Promise<void> {
  await update($, steps, list => list.map(x => (x.t === t ? { ...x, state, ms: Date.now() - x.t } : x)))
  await redraw($)
}

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
  $.ui.status(statusText(s, await read($, steps)))
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
    // A surface without slash commands still gets the status line and the band.
    await $.command
      .register({
        name: 'director-activity',
        description: "Show or hide the director's live activity above the prompt (the status line always shows the lanes)",
      })
      .catch(() => undefined)
    await refresh($)
    return started
  })

  on('command.run', { command: 'director-activity' }, async $ => {
    const on_ = !(await read($, verbose))
    await update($, verbose, () => on_)
    return { text: `director activity above the prompt: ${on_ ? 'shown' : 'hidden'}` }
  })

  on('turn.complete', async ($, e, next) => {
    const done = await next(e)
    await refresh($)
    return done
  })

  // Every tool call, live: recorded on its lane when it starts and closed when it ends, so the status line and the band
  // show what each lane is doing at every moment. The director's own tools also re-read its state.
  on('tool.call', async ($, e, next) => {
    const { lane, label } = classify(e.tool, e as unknown as Record<string, unknown>)
    const t = Date.now()
    await record($, { t, lane: e.agentId && lane === 'frontier' ? 'frontier' : lane, label: e.agentId ? `[subagent] ${label}` : label, state: 'running' })
    try {
      const result = await next(e)
      await finish($, t, 'deny' in result && result.deny ? 'error' : result.isError ? 'error' : 'done')
      if (e.tool.startsWith(TOOL_PREFIX)) void refresh($)
      return result
    } catch (err) {
      await finish($, t, 'error')
      throw err
    }
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
    const live = await read($, steps)
    const showActivity = (await read($, verbose)) && (live.length > 0 || (s?.recent ?? []).length > 0)
    if (e.props.hasSurvey || !s?.memory || (!proposals.length && !delegations.length && !plans.length && !showActivity))
      return next(e)

    const { Box, Button, Text, Code } = $.ui.resolve(e)
    const shown = await read($, diff)
    const working = await read($, busy)
    const said = await read($, message)
    const p = proposals[0]
    const plan = plans[0]

    const now = Date.now()
    const directorSteps = live.filter(x => x.lane === 'director').slice(-5)
    const journal = (s?.recent ?? []).slice(-3)
    const laneLine = (lane: Step['lane']) => {
      const x = [...live].reverse().find(y => y.lane === lane)
      return x ? `${x.label} · ${ago(x, now)}` : 'idle'
    }

    return (
      <Box flexDirection="column">
        {showActivity && (
          <Box flexDirection="column">
            <Text bold>Director · activity</Text>
            {directorSteps.map(x => (
              <Text key={`step-${x.t}`} dimColor={x.state !== 'running'}>
                {x.state === 'running' ? '▶ ' : x.state === 'error' ? '✗ ' : '✓ '}
                {x.label}
              </Text>
            ))}
            {journal.map(j => (
              <Text key={`journal-${j.t}-${j.kind}`} dimColor>
                · {j.kind}: {j.what}
              </Text>
            ))}
            <Text>
              <Text bold>frontier</Text> ▶ {laneLine('frontier')}
            </Text>
            <Text>
              <Text bold>local</Text> ▶ {laneLine('local')}
            </Text>
          </Box>
        )}
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
