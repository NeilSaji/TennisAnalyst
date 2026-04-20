'use client'

import { useEffect, useMemo } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import BaselineCard from '@/components/BaselineCard'
import { useBaselineStore } from '@/store/baseline'
import { useUser } from '@/hooks/useUser'

export default function BaselinePage() {
  const { baselines, loading, error, refresh, setActive, rename, remove } = useBaselineStore()
  const { user, loading: authLoading } = useUser()
  const router = useRouter()

  useEffect(() => {
    if (authLoading) return
    if (!user) {
      router.replace('/login?next=/baseline')
      return
    }
    refresh()
  }, [authLoading, user, refresh, router])

  const { active, history } = useMemo(() => {
    const active = baselines.filter((b) => b.is_active)
    const history = baselines.filter((b) => !b.is_active)
    return { active, history }
  }, [baselines])

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      <div className="mb-8 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black text-white mb-2">Your Baselines</h1>
          <p className="text-white/50">
            Your best-day swings, saved for long-term comparison. Upload a new swing and
            we&apos;ll show you what&apos;s improved.
          </p>
        </div>
        <Link
          href="/baseline/compare"
          className="shrink-0 px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white font-semibold text-sm transition-colors"
        >
          Compare new swing
        </Link>
      </div>

      {error && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 mb-6">
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {loading && baselines.length === 0 && (
        <div className="rounded-xl border border-white/10 bg-white/[0.02] p-8 text-center">
          <p className="text-white/50 text-sm">Loading baselines...</p>
        </div>
      )}

      {!loading && baselines.length === 0 && !error && (
        <div className="rounded-xl border border-white/10 bg-white/[0.02] p-12 text-center">
          <div className="text-5xl mb-3">🎾</div>
          <p className="text-white font-semibold mb-2">No baselines yet</p>
          <p className="text-white/50 text-sm mb-6">
            Analyze a swing and mark it as your baseline to start tracking progress.
          </p>
          <Link
            href="/analyze"
            className="inline-block px-6 py-3 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl transition-colors"
          >
            Analyze a swing
          </Link>
        </div>
      )}

      {active.length > 0 && (
        <section className="space-y-4 mb-8">
          <h2 className="text-sm font-semibold text-white/70 uppercase tracking-wide">Active</h2>
          <div className="space-y-3">
            {active.map((b) => (
              <BaselineCard
                key={b.id}
                baseline={b}
                onRename={rename}
                onDelete={remove}
              />
            ))}
          </div>
        </section>
      )}

      {history.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-sm font-semibold text-white/70 uppercase tracking-wide">History</h2>
          <div className="space-y-3">
            {history.map((b) => (
              <BaselineCard
                key={b.id}
                baseline={b}
                onSetActive={setActive}
                onRename={rename}
                onDelete={remove}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
