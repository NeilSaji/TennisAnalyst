'use client'

import { useEffect, useState } from 'react'

export interface BestShotPanelProps {
  sessionId: string
  segmentCount: number
}

type BestShot = { bestIndex: number; reasoning: string }

export default function BestShotPanel({ sessionId, segmentCount }: BestShotPanelProps) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [best, setBest] = useState<BestShot | null>(null)

  useEffect(() => {
    if (segmentCount < 2) return
    let cancelled = false
    setLoading(true)
    fetch(`/api/segments/${encodeURIComponent(sessionId)}/best-shot`, {
      method: 'POST',
      credentials: 'same-origin',
    })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`)
        return r.json() as Promise<BestShot>
      })
      .then((body) => {
        if (cancelled) return
        setBest(body)
        setError(null)
      })
      .catch((e: Error) => {
        if (cancelled) return
        setError(e.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [sessionId, segmentCount])

  if (segmentCount < 2) return null

  return (
    <div className="rounded-2xl border border-cyan-500/30 bg-gradient-to-br from-cyan-500/10 to-cyan-500/[0.02] p-5">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-base">✨</span>
        <h3 className="text-xs font-semibold text-white/80 uppercase tracking-wide">AI pick</h3>
      </div>
      {loading && (
        <p className="text-white/60 text-sm">Analyzing your {segmentCount} shots…</p>
      )}
      {error && !loading && (
        <p className="text-red-300 text-sm">Couldn&apos;t pick a best shot: {error}</p>
      )}
      {best && !loading && (
        <p className="text-white text-sm leading-relaxed">
          <span className="font-semibold text-cyan-300">Shot {best.bestIndex + 1}</span>
          {' — '}
          {best.reasoning}
        </p>
      )}
    </div>
  )
}
