'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { upload } from '@vercel/blob/client'
import JointTogglePanel from '@/components/JointTogglePanel'
import SwingSelector from '@/components/SwingSelector'
import { useBaselineStore } from '@/store/baseline'
import { usePoseExtractor } from '@/hooks/usePoseExtractor'
import { useUser } from '@/hooks/useUser'
import { detectSwings } from '@/lib/jointAngles'
import { extractPoseViaRailway, RailwayExtractError } from '@/lib/poseExtractionRailway'
import type { PoseFrame, Baseline } from '@/lib/supabase'

// Mirrors UploadZone's flag. When true the today's-clip extraction
// goes through Railway (RTMPose + YOLO crop) which produces meaningfully
// better tracing on small-in-frame subjects than browser MediaPipe.
// Browser MediaPipe stays as the fallback if Railway is unreachable
// or returns no result.
const USE_RAILWAY_EXTRACT =
  process.env.NEXT_PUBLIC_USE_RAILWAY_EXTRACT === 'true'

function safeBlobFilename(name: string): string {
  return (
    name
      .split(/[\\/]/)
      .pop()!
      .replace(/[^a-zA-Z0-9._-]/g, '_')
      .slice(0, 100) || 'upload'
  )
}

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
  // Index (1-based) of the swing within today's clip the user wants to
  // compare against the baseline. Multi-swing clips would otherwise
  // confuse phase detection — detectSwingPhases finds ONE peak across
  // the whole video, so only one of N swings gets clean phase sync.
  // Slicing to one swing at a time fixes that.
  const [selectedTodaySwing, setSelectedTodaySwing] = useState<number | null>(null)
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

  // Detect swings within today's uploaded clip. Multi-swing clips
  // (e.g. "I just hit 12 forehands and uploaded the rally") get
  // segmented so each swing can be compared independently.
  const todaySwings = useMemo(() => detectSwings(todayFrames), [todayFrames])
  const hasMultipleTodaySwings = todaySwings.length > 1

  // The frames actually fed into ComparisonLayout / MetricsComparison /
  // the LLM. When the user picks a specific swing, slice to it. When
  // there's only one swing, use the whole clip (detectSwings returns
  // the whole array as a single segment in that case anyway). Slicing
  // is what lets phase sync work on every swing the user picks: phases
  // re-detect cleanly on a 30-frame slice in a way they cannot on a
  // 3000-frame multi-swing rally.
  const todayFramesForCompare = useMemo<PoseFrame[]>(() => {
    if (selectedTodaySwing != null && todaySwings[selectedTodaySwing - 1]) {
      return todaySwings[selectedTodaySwing - 1].frames
    }
    return todayFrames
  }, [selectedTodaySwing, todaySwings, todayFrames])

  // Clean up any object URL on unmount / replacement
  useEffect(() => {
    return () => {
      if (todayObjectUrl) URL.revokeObjectURL(todayObjectUrl)
    }
  }, [todayObjectUrl])

  const processTodayVideo = async (file: File) => {
    if (!file.type.startsWith('video/')) return

    let frames: PoseFrame[] | null = null

    // Railway path: upload the file to Vercel Blob first (Railway pulls
    // from a URL, not a multipart body), then create a pending session
    // row, then queue + poll. On any failure fall through silently to
    // the browser MediaPipe path so the compare flow never dies on a
    // bad day for the server.
    if (USE_RAILWAY_EXTRACT) {
      try {
        const blobPath = `videos/${Date.now()}-${safeBlobFilename(file.name)}`
        const blob = await upload(blobPath, file, {
          access: 'public',
          handleUploadUrl: '/api/upload',
          contentType: file.type,
        })
        const sessRes = await fetch('/api/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ blobUrl: blob.url, shotType: selectedBaseline?.shot_type ?? 'unknown' }),
        })
        if (sessRes.ok) {
          const { sessionId } = await sessRes.json()
          const result = await extractPoseViaRailway({
            sessionId,
            blobUrl: blob.url,
          })
          frames = result.frames
        }
      } catch (err) {
        if (err instanceof RailwayExtractError) {
          console.info('[baseline/compare] Railway path unavailable, falling back to browser:', err.reason)
        } else {
          console.error('[baseline/compare] Railway path errored, falling back to browser:', err)
        }
        frames = null
      }
    }

    // Browser MediaPipe fallback (or primary path when the feature flag
    // is off). Always runs the local extractor so we still produce an
    // object URL for inline playback even when Railway succeeded.
    const localResult = await extract(file)
    if (!localResult) return
    if (todayObjectUrl) URL.revokeObjectURL(todayObjectUrl)
    setTodayObjectUrl(localResult.objectUrl)

    // Prefer Railway frames (better quality) when available; fall back
    // to the browser MediaPipe frames otherwise.
    setTodayFrames(frames ?? localResult.frames)

    // Reset the per-swing selection on a new upload so the previous
    // clip's selection doesn't leak into a clip that may not have
    // that many swings.
    setSelectedTodaySwing(null)
  }

  // When today's frames change (new upload), default to swing 1 if
  // there are multiple swings so phase sync engages immediately
  // instead of waiting for the user to discover the selector.
  useEffect(() => {
    if (todaySwings.length > 1 && selectedTodaySwing === null) {
      setSelectedTodaySwing(1)
    } else if (todaySwings.length <= 1 && selectedTodaySwing !== null) {
      setSelectedTodaySwing(null)
    }
  }, [todaySwings.length, selectedTodaySwing])

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
              {hasMultipleTodaySwings && (
                <div className="px-3 pt-3">
                  <SwingSelector
                    allFrames={todayFrames}
                    selectedIndex={selectedTodaySwing}
                    onSelect={(seg) => setSelectedTodaySwing(seg.index)}
                  />
                </div>
              )}
              <div className="p-3">
                <ComparisonLayout
                  userBlobUrl={selectedBaseline.blob_url}
                  userFrames={baselineFrames}
                  proFrames={todayFramesForCompare}
                  proVideoUrl={todayObjectUrl ?? ''}
                  userName="Baseline"
                  proName="Today"
                />
              </div>
            </div>
          )}

          {canCompare && (
            <MetricsComparison
              userFrames={baselineFrames}
              proFrames={todayFramesForCompare}
              shotType={selectedBaseline?.shot_type ?? null}
              userLabel="Baseline"
              proLabel="Today"
            />
          )}

          {canCompare && selectedBaseline && (
            <LLMCoachingPanel
              compareMode="baseline"
              frames={baselineFrames}
              compareFrames={todayFramesForCompare}
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
