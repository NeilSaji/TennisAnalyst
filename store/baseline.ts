import { create } from 'zustand'
import type { Baseline } from '@/lib/supabase'

interface BaselineStore {
  baselines: Baseline[]
  activeBaseline: Baseline | null
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
  setActive: (id: string) => Promise<void>
  rename: (id: string, label: string) => Promise<void>
  remove: (id: string) => Promise<void>
}

function pickActive(list: Baseline[]): Baseline | null {
  // Prefer the active row; otherwise most recent. "Active" is per shot_type in
  // principle, but for the store we surface a single active baseline for the UI.
  const active = list.find((b) => b.is_active)
  if (active) return active
  return list[0] ?? null
}

export const useBaselineStore = create<BaselineStore>((set, get) => ({
  baselines: [],
  activeBaseline: null,
  loading: false,
  error: null,

  refresh: async () => {
    set({ loading: true, error: null })
    try {
      const res = await fetch('/api/baselines', { credentials: 'same-origin' })
      if (!res.ok) {
        const text = await res.text()
        set({ loading: false, error: text || `HTTP ${res.status}` })
        return
      }
      const { baselines } = (await res.json()) as { baselines: Baseline[] }
      set({
        baselines,
        activeBaseline: pickActive(baselines),
        loading: false,
      })
    } catch (err) {
      set({ loading: false, error: err instanceof Error ? err.message : 'Failed to load baselines' })
    }
  },

  setActive: async (id: string) => {
    const res = await fetch(`/api/baselines/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ isActive: true }),
    })
    if (!res.ok) {
      set({ error: (await res.text()) || `HTTP ${res.status}` })
      return
    }
    await get().refresh()
  },

  rename: async (id: string, label: string) => {
    const res = await fetch(`/api/baselines/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ label }),
    })
    if (!res.ok) {
      set({ error: (await res.text()) || `HTTP ${res.status}` })
      return
    }
    await get().refresh()
  },

  remove: async (id: string) => {
    const res = await fetch(`/api/baselines/${id}`, {
      method: 'DELETE',
      credentials: 'same-origin',
    })
    if (!res.ok) {
      set({ error: (await res.text()) || `HTTP ${res.status}` })
      return
    }
    await get().refresh()
  },
}))
