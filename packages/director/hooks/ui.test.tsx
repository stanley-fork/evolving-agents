import type { On } from 'claude-code'
import { describe, expect, test } from 'claude-code/testing'

import { classify, laneText, runSummary, statusText } from './ui'

const SNAP = {
  memory: '/proj/.director/memory',
  statements: 286,
  accepted: 3,
  proposals: [
    {
      id: 'p20261018-120000',
      rationale: 'check the README case count before drafting',
      edits: ['add working#count-before-draft'],
      lint: 0,
      check: 'TEACHES',
      probes: 2,
      acceptable: true,
      why_not: '',
    },
  ],
  delegations: [{ id: 'summarise-logs', lane: 'local', status: 'done', done: true }],
}

const BAND = {
  component: 'AbovePrompt',
  props: { hasSurvey: false, isWorking: false, maxRows: 20, bodyColumns: 80, scroll: { offset: 0, bodyRows: 19 }, view: {} },
} as const

// ui_state.py stands behind $.process.run: the test answers it with a canned state and records what was asked.
function fake(on: On, state: object, calls: string[][]) {
  on('session.root', () => ({ value: '/proj' }))
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', () => ({ value: undefined }))
  on('process.run', ($, e) => {
    const args = e.argv.slice(2)
    calls.push([...args])
    const out = args[0] === 'status' ? state : args[0] === 'diff' ? { id: args[1], diff: '@@ -1 +1 @@\n-a\n+b' } : { ok: true }
    return {
      value: { exitCode: 0, stdout: JSON.stringify(out), stderr: '', isStdoutTruncated: false, isStderrTruncated: false },
    }
  })
}

describe('status line', () => {
  test('silent without a memory, counts with one', async () => {
    expect(statusText(null)).toBe(undefined)
    expect(statusText({ memory: null })).toBe(undefined)
    expect(statusText(SNAP)).toBe('director · 286 statements · 3 learned · 1 to decide · 1 delegated')
    expect(statusText({ memory: '/m', statements: 10, accepted: 0, proposals: [], delegations: [] })).toBe(
      'director · 10 statements · 0 learned',
    )
  })
})

const PLAN = {
  memory: '/proj/.director/memory',
  statements: 10,
  accepted: 0,
  proposals: [],
  delegations: [],
  plans: [
    {
      id: 'plan-20261019-120000',
      request: 'ship 0.6',
      tasks: [
        { id: 'draft-note', task: 'draft the release note', lane: 'local', confidential: false, movable: true },
        { id: 'read-contract', task: 'summarise the contract', lane: 'private-exec', confidential: true, movable: false },
      ],
    },
  ],
}

describe('plans on the band', () => {
  test('the status line counts a waiting plan, and Run is summarised for the session', async () => {
    expect(statusText(PLAN)).toBe('director · 10 statements · 0 learned · 1 plan waiting')
    expect(statusText({ ...PLAN, plans: [], recipes_stale: true })).toBe(
      'director · 10 statements · 0 learned · recipes stale',
    )
    expect(statusText({ ...PLAN, plans: [], to_reflect: 3 })).toBe('director · 10 statements · 0 learned · 3 to reflect')
    expect(
      runSummary({
        results: [
          { id: 'draft-note', lane: 'frontier', status: 'in_session' },
          { id: 'sum', lane: 'local', status: 'submitted' },
          { id: 'x', lane: 'cloud', error: 'refused' },
        ],
      }),
    ).toBe('draft-note is yours to do now; sum delegated to local; x failed on cloud (refused)')
  })

  test('each task shows its lane; Move only where another lane is allowed; Run and Move go through ui_state', async ($, on) => {
    const calls: string[][] = []
    fake(on, PLAN, calls)
    await $.session.start({ cwd: '/proj', surface: 'terminal', isInteractive: true })
    for (const surface of ['terminal', 'desktop', 'mobile'] as const) {
      const ui = await $.ui.mount({ plugin: 'director', surface, ...BAND })
      expect(await ui.find({ type: 'Text', text: /a plan to run/ })).toBeDefined()
      expect(await ui.find({ key: 'move-plan-20261019-120000-draft-note' })).toBeDefined()
      expect(await ui.find({ key: 'move-plan-20261019-120000-read-contract' })).toBe(undefined)
      await ui.unmount()
    }
    const ui = await $.ui.mount({ plugin: 'director', surface: 'terminal', ...BAND })
    await ui.press({ key: 'move-plan-20261019-120000-draft-note' })
    expect(calls).toContainEqual(['plan-move', 'plan-20261019-120000', 'draft-note'])
    await ui.press({ key: 'run-plan-20261019-120000' })
    expect(calls).toContainEqual(['plan-run', 'plan-20261019-120000'])
    await ui.unmount()
  })
})

describe('band', () => {
  test('shows the lesson and the delegation, and Accept goes through ui_state.py', async ($, on) => {
    const calls: string[][] = []
    fake(on, SNAP, calls)
    await $.session.start({ cwd: '/proj', surface: 'terminal', isInteractive: true })
    for (const surface of ['terminal', 'desktop', 'mobile'] as const) {
      const ui = await $.ui.mount({ plugin: 'director', surface, ...BAND })
      expect(await ui.find({ type: 'Text', text: /a lesson to decide/ })).toBeDefined()
      expect(await ui.find({ type: 'Text', text: /check: TEACHES/ })).toBeDefined()
      expect(await ui.find({ key: 'fetch-summarise-logs' })).toBeDefined()
      await ui.unmount()
    }
    const ui = await $.ui.mount({ plugin: 'director', surface: 'terminal', ...BAND })
    await ui.press({ key: 'diff-p20261018-120000' })
    expect(await ui.find({ type: 'Code' })).toBeDefined()
    await ui.press({ key: 'accept-p20261018-120000' })
    expect(calls).toContainEqual(['accept', 'p20261018-120000'])
    await ui.unmount()
  })

  test('says why a lesson cannot be accepted, and offers no Accept', async ($, on) => {
    const blocked = {
      ...SNAP,
      delegations: [],
      proposals: [{ ...SNAP.proposals[0]!, check: 'REDUNDANT', acceptable: false, why_not: 'its check says REDUNDANT' }],
    }
    fake(on, blocked, [])
    await $.session.start({ cwd: '/proj', surface: 'terminal', isInteractive: true })
    const ui = await $.ui.mount({ plugin: 'director', surface: 'terminal', ...BAND })
    expect(await ui.find({ type: 'Text', text: /Cannot be accepted/ })).toBeDefined()
    expect(await ui.find({ key: 'accept-p20261018-120000' })).toBe(undefined)
    await ui.unmount()
  })

  test('draws nothing of its own where the project has no memory', async ($, on) => {
    fake(on, { memory: null }, [])
    let engineDrew = false
    on('ui.render', ($, e) => {
      engineDrew = true
      const { Text } = $.ui.resolve(e)
      return <Text>engine</Text>
    })
    await $.session.start({ cwd: '/proj', surface: 'terminal', isInteractive: true })
    const ui = await $.ui.mount({ plugin: 'director', surface: 'terminal', ...BAND })
    expect(engineDrew).toBe(true)
    expect(await ui.find({ type: 'Text', text: /Director/ })).toBe(undefined)
    await ui.unmount()
  })
})


describe('live activity', () => {
  test('a tool call lands on its lane with a short label', () => {
    expect(classify('Read', { file_path: '/p/.director/memory/Headroom.md' })).toEqual({ lane: 'director', label: 'walk → Headroom' })
    expect(classify('Grep', { pattern: 'brief', path: '/p/.director/memory' }).lane).toBe('director')
    expect(classify('mcp__plugin_director_director__memory_propose', { rationale: 'cap the key' })).toEqual({
      lane: 'director',
      label: 'memory_propose: cap the key',
    })
    expect(classify('mcp__plugin_director_director__plan_local', { request: 'draft the note' }).lane).toBe('local')
    expect(classify('mcp__plugin_director_director__delegate', { lane: 'tower', task: 'correct ch. 1' })).toEqual({
      lane: 'local',
      label: '→ tower: correct ch. 1',
    })
    expect(classify('Bash', { command: 'pytest -q', description: 'Run the tests' })).toEqual({
      lane: 'frontier',
      label: '$ Run the tests',
    })
    expect(classify('Edit', { file_path: '/repo/README.md' })).toEqual({ lane: 'frontier', label: 'Edit README.md' })
  })

  test('the status line shows what each lane is doing, and stays as it was with nothing live', () => {
    const now = 100_000
    const live = [
      { t: 90_000, lane: 'frontier' as const, label: '$ Run the tests', state: 'running' as const },
      { t: 80_000, lane: 'local' as const, label: 'plan_local: draft', state: 'done' as const, ms: 4_000 },
    ]
    expect(laneText(live, null, now)).toBe('frontier ▶ $ Run the tests (10s…) │ local ▶ plan_local: draft (4s)')
    expect(statusText({ memory: '/m', statements: 10, accepted: 0 }, live, now)).toBe(
      'director · 10 statements · 0 learned │ frontier ▶ $ Run the tests (10s…) │ local ▶ plan_local: draft (4s)',
    )
    expect(statusText({ memory: '/m', statements: 10, accepted: 0 }, [], now)).toBe('director · 10 statements · 0 learned')
    expect(laneText([], { memory: '/m', delegations: [{ id: 'ch-1', lane: 'tower', status: 'running', done: false }] }, now)).toBe(
      'frontier ▶ idle │ local ▶ tower: ch-1 (running)',
    )
  })
})
