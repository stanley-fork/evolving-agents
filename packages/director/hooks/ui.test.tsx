import type { On } from 'claude-code'
import { describe, expect, test } from 'claude-code/testing'

import { statusText } from './ui'

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
