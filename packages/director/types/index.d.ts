export type Proposal = {
  id: string
  rationale: string
  edits: string[]
  lint: number
  check: string | null
  probes: number
  acceptable: boolean
  why_not: string
}

export type Delegation = { id: string; lane: string; status: string; done: boolean }

export type Snapshot = {
  memory: string | null
  statements?: number
  accepted?: number
  proposals?: Proposal[]
  delegations?: Delegation[]
  error?: string
}

declare module 'claude-code' {
  interface PluginState {
    director: {
      snap: Snapshot | null
      diff: { id: string; text: string } | null
      busy: string | null
      message: string | null
      notes: string[]
    }
  }
}
