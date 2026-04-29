'use client'

import { useEffect, useState, use } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import VideoCanvas from '@/components/VideoCanvas'
import { useUser } from '@/hooks/useUser'
import type { Baseline } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'

const ALL_VISIBLE: Record<JointGroup, boolean> = {
  shoulders: true,
  elbows: true,
  wrists: true,
  hips: true,
  knees: true,
  ankles: true,
}

export default function BaselineWatchPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const { user, loading: authLoading } = useUser()
  const router = useRouter()
  const [baseline, setBaseline] = useState<Baseline | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (authLoading) return
    if (!user) {
      router.replace(`/login?next=/baseline/${id}`)
      return
    }
    let cancelled = false
    setLoading(true)
    fetch(`/api/baselines/${encodeURIComponent(id)}`)
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).error || `HTTP ${r.status}`)
        return r.json()
      })
      .then((body: { baseline: Baseline }) => {
        if (cancelled) return
        setBaseline(body.baseline)
        setError(null)
      })
      .catch((e: Error) => {
        if (cancelled) return
        setError(e.message)
      })
      .finally(() => {
        if (cancelled) return
        setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [authLoading, user, id, router])

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      <div className="mb-6">
        <Link
          href="/baseline"
          className="text-sm text-white/50 hover:text-white transition-colors"
        >
          ← Baselines
        </Link>
      </div>

      {loading && (
        <div className="rounded-xl border border-white/10 bg-white/[0.02] p-8 text-center">
          <p className="text-white/50 text-sm">Loading baseline...</p>
        </div>
      )}

      {error && !loading && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4">
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {baseline && !loading && (
        <>
          <div className="mb-6 flex items-start justify-between gap-4">
            <div className="min-w-0">
              <h1 className="text-2xl font-black text-white truncate">{baseline.label}</h1>
              <div className="flex gap-2 items-center text-xs text-white/50 mt-1">
                <span className="capitalize">{baseline.shot_type}</span>
                <span>·</span>
                <span>{baseline.keypoints_json?.frame_count ?? 0} frames</span>
                {baseline.is_active && (
                  <>
                    <span>·</span>
                    <span className="text-emerald-400 font-medium">Active</span>
                  </>
                )}
              </div>
            </div>
            <Link
              href="/baseline/compare"
              className="shrink-0 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-white/70 hover:text-white text-xs font-medium transition-colors"
            >
              Compare new swing
            </Link>
          </div>

          <VideoCanvas
            src={baseline.blob_url}
            framesData={baseline.keypoints_json?.frames ?? []}
            visible={ALL_VISIBLE}
            showSkeleton
            showRacket
            showAngles
            shotType={baseline.shot_type}
          />
        </>
      )}
    </div>
  )
}
