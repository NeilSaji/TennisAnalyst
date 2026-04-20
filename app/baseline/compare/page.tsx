'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import JointTogglePanel from '@/components/JointTogglePanel'
import { useBaselineStore } from '@/store/baseline'
import { usePoseExtractor } from '@/hooks/usePoseExtractor'
import { useUser } from '@/hooks/useUser'
import type { PoseFrame, Baseline } from '@/lib/supabase'

const ComparisonLayout = dynamic(() => import('@/components/ComparisonLayout'), { ssr: false })
const MetricsComparison = dynamic(() => import('@/components/MetricsComparison'), { ssr: false })
const LLMCoachingPanel = dynamic(() => import('@/components/LLMCoachingPanel'), { ssr: false })

export default function BaselineComparePage() {
  const { baselines, activeBaseline, loading, error, refresh } = useBaselineStore()
  const { extract, progress, isProcessing } = usePoseExtractor()
  const { user, loading: authLoading } = useUser()
  const router = useRouter()

  const [todayObjectUrl, setTodayObjectUrl] = useState<string | null>(null)
  const [todayFrames, setTodayFrames] = useState<PoseFrame[]>([])
  const [dragging, setDragging] = useState(false)
  const [selectedBaselineId, setSelectedBaselineId] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Gate behind auth; load baselines once signed in.
  useEffect(() => {
    if (authLoading) return
    if (!user) {
      router.replace('/login?next=/baseline/compare')
      return
    }
    refresh()
  }, [authLoading, user, refresh, router])

  // Derive the effective selection without storing a default — the defaulting
  // logic falls back to the active baseline when the user hasn't explicitly picked one.
  const selectedBaseline: Baseline | null = useMemo(() => {
    if (selectedBaselineId) {
      const picked = baselines.find((b) => b.id === selectedBaselineId)
      if (picked) return picked
    }
    return activeBaseline
  }, [baselines, selectedBaselineId, activeBaseline])

  const baselineFrames = selectedBaseline?.keypoints_json?.frames ?? []

  // Clean up any object URL on unmount / replacement
  useEffect(() => {
    return () => {
      if (todayObjectUrl) URL.revokeObjectURL(todayObjectUrl)
    }
  }, [todayObjectUrl])

  const processTodayVideo = async (file: File) => {
    if (!file.type.startsWith('video/')) return
    const result = await extract(file)
    if (!result) return
    // Swap in the new object URL, revoke the old
    if (todayObjectUrl) URL.revokeObjectURL(todayObjectUrl)
    setTodayObjectUrl(result.objectUrl)
    setTodayFrames(result.frames)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) processTodayVideo(file)
  }

  const hasToday = todayObjectUrl && todayFrames.length > 0
  const canCompare = selectedBaseline && baselineFrames.length > 0 && hasToday

  if (!loading && baselines.length === 0 && !error) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center">
        <div className="text-5xl mb-3">🎾</div>
        <h1 className="text-2xl font-black text-white mb-2">No baseline saved yet</h1>
        <p className="text-white/50 mb-6">
          Analyze a swing first, then mark it as your baseline. You&apos;ll then be able to
          see how future swings stack up.
        </p>
        <Link
          href="/analyze"
          className="inline-block px-6 py-3 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl transition-colors"
        >
          Analyze a swing
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black text-white mb-2">Beat Your Last Swing</h1>
          <p className="text-white/50">
            Your best day on the left. Today&apos;s swing on the right.
          </p>
        </div>
        <Link
          href="/baseline"
          className="shrink-0 text-sm text-white/60 hover:text-white transition-colors"
        >
          Manage baselines →
        </Link>
      </div>

      {error && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 mb-6">
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Baseline selector if multiple exist */}
          {baselines.length > 1 && (
            <div className="flex gap-2 flex-wrap">
              {baselines.map((b) => (
                <button
                  key={b.id}
                  onClick={() => setSelectedBaselineId(b.id)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                    b.id === selectedBaseline?.id
                      ? 'bg-emerald-500 text-white'
                      : 'bg-white/10 text-white/60 hover:bg-white/20 hover:text-white'
                  }`}
                >
                  {b.label} · {b.shot_type}
                </button>
              ))}
            </div>
          )}

          {/* Upload zone for today's swing */}
          {!hasToday && (
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => !isProcessing && fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
                dragging
                  ? 'border-emerald-400 bg-emerald-500/10'
                  : 'border-white/20 hover:border-white/40 hover:bg-white/5'
              } ${isProcessing ? 'cursor-not-allowed opacity-80' : ''}`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) processTodayVideo(file)
                }}
              />
              {isProcessing ? (
                <div className="space-y-3">
                  <p className="text-white">Processing today&apos;s swing... {progress}%</p>
                  <div className="w-full bg-white/10 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="h-full bg-emerald-400 rounded-full transition-all"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              ) : (
                <div>
                  <div className="text-4xl mb-3">🎬</div>
                  <p className="text-white font-medium">Upload today&apos;s swing</p>
                  <p className="text-white/40 text-sm mt-1">MP4, MOV, WebM</p>
                </div>
              )}
            </div>
          )}

          {/* Side-by-side comparison */}
          {canCompare && selectedBaseline && (
            <div className="rounded-2xl border border-white/10 overflow-hidden bg-black">
              <div className="p-3 border-b border-white/5 flex items-center justify-between">
                <p className="text-sm text-white/60">
                  <span className="text-emerald-300">{selectedBaseline.label}</span>
                  <span className="text-white/30"> vs </span>
                  <span className="text-white">Today</span>
                </p>
                <button
                  onClick={() => {
                    if (todayObjectUrl) URL.revokeObjectURL(todayObjectUrl)
                    setTodayObjectUrl(null)
                    setTodayFrames([])
                  }}
                  className="text-xs text-white/40 hover:text-white"
                >
                  Upload different video
                </button>
              </div>
              <div className="p-3">
                <ComparisonLayout
                  userBlobUrl={selectedBaseline.blob_url}
                  userFrames={baselineFrames}
                  proFrames={todayFrames}
                  proVideoUrl={todayObjectUrl ?? ''}
                  proName="Today"
                />
              </div>
            </div>
          )}

          {canCompare && (
            <MetricsComparison
              userFrames={baselineFrames}
              proFrames={todayFrames}
              shotType={selectedBaseline?.shot_type ?? null}
            />
          )}

          {canCompare && selectedBaseline && (
            <LLMCoachingPanel
              proSwing={null}
              compareMode="baseline"
              frames={baselineFrames}
              compareFrames={todayFrames}
              baselineLabel={selectedBaseline.label}
            />
          )}
        </div>

        <div className="space-y-4">
          <JointTogglePanel />

          {selectedBaseline && (
            <div className="rounded-xl bg-white/5 border border-white/10 p-4">
              <h3 className="text-sm font-semibold text-white mb-2">Comparing against</h3>
              <p className="text-white font-medium truncate">{selectedBaseline.label}</p>
              <p className="text-white/50 text-xs capitalize mt-0.5">
                {selectedBaseline.shot_type} ·{' '}
                {new Date(selectedBaseline.created_at).toLocaleDateString()}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
