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

export type PlanTask = { id: string; task: string; lane: string; confidential: boolean; movable: boolean }

export type Plan = { id: string; request: string; tasks: PlanTask[] }

export type Snapshot = {
  memory: string | null
  statements?: number
  accepted?: number
  proposals?: Proposal[]
  delegations?: Delegation[]
  plans?: Plan[]
  recipes_stale?: boolean
  to_reflect?: number
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
