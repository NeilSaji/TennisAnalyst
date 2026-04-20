'use client'

import { useState, useMemo } from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import JointTogglePanel from '@/components/JointTogglePanel'
import SwingSelector from '@/components/SwingSelector'
import { usePoseStore, useJointStore } from '@/store'
import { detectSwings } from '@/lib/jointAngles'
import { useUser } from '@/hooks/useUser'
import { useProfile } from '@/hooks/useProfile'
import type { SwingSegment } from '@/lib/jointAngles'
import type { PoseFrame } from '@/lib/supabase'

const VALID_BASELINE_SHOTS = new Set(['forehand', 'backhand', 'serve', 'volley', 'slice'])

// Dynamic import to avoid SSR issues with MediaPipe WASM
const UploadZone = dynamic(() => import('@/components/UploadZone'), { ssr: false })
const VideoCanvas = dynamic(() => import('@/components/VideoCanvas'), { ssr: false })
const LLMCoachingPanel = dynamic(() => import('@/components/LLMCoachingPanel'), { ssr: false })

export default function AnalyzePage() {
  const { framesData, blobUrl, localVideoUrl, sessionId, shotType } = usePoseStore()
  const { visible, showSkeleton, showTrail } = useJointStore()
  const [done, setDone] = useState(false)
  const [allFrames, setAllFrames] = useState<PoseFrame[]>([])
  const [selectedSwing, setSelectedSwing] = useState<number | null>(null)
  const [baselineStatus, setBaselineStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [baselineError, setBaselineError] = useState<string | null>(null)
  const { user, loading: authLoading } = useUser()
  const { profile, skipped, loading: profileLoading } = useProfile()
  const isAdvanced = !profileLoading && profile?.skill_tier === 'advanced'
  // Show the "set your profile" hint only to users who explicitly skipped
  // onboarding — onboarded users already have a tier, anons can't persist one.
  const showProfileHint = !profileLoading && skipped && !profile

  const swings = useMemo(() => detectSwings(allFrames), [allFrames])
  const hasMultipleSwings = swings.length > 1

  const handleComplete = (_url: string, frames: PoseFrame[]) => {
    setAllFrames(frames)
    const detected = detectSwings(frames)
    if (detected.length > 1) {
      setSelectedSwing(1)
    } else {
      setSelectedSwing(null)
    }
    setDone(true)
  }

  const handleSwingSelect = (seg: SwingSegment) => {
    setSelectedSwing(seg.index)
  }

  // The frames to send to the LLM - either the selected swing or all frames
  const analysisFrames = selectedSwing && swings[selectedSwing - 1]
    ? swings[selectedSwing - 1].frames
    : framesData

  // For baseline saving: multi-swing videos must have one swing selected.
  // Single-swing videos use all frames directly.
  const baselineFrames = analysisFrames
  const canSaveBaseline =
    done &&
    !!blobUrl &&
    !!shotType &&
    VALID_BASELINE_SHOTS.has(shotType) &&
    baselineFrames.length > 0 &&
    (!hasMultipleSwings || selectedSwing !== null)

  const saveAsBaseline = async () => {
    if (!canSaveBaseline) return
    setBaselineStatus('saving')
    setBaselineError(null)
    try {
      const keypointsJson = {
        fps_sampled: 30,
        frame_count: baselineFrames.length,
        frames: baselineFrames,
      }
      const res = await fetch('/api/baselines', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          blobUrl,
          shotType,
          keypointsJson,
          sourceSessionId: sessionId ?? undefined,
        }),
      })
      if (!res.ok) {
        const msg = (await res.text()) || `HTTP ${res.status}`
        setBaselineStatus('error')
        setBaselineError(msg)
        return
      }
      setBaselineStatus('saved')
    } catch (err) {
      setBaselineStatus('error')
      setBaselineError(err instanceof Error ? err.message : 'Failed to save baseline')
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-white mb-2">Analyze Your Swing</h1>
        <p className="text-white/50">
          Upload a video to extract pose landmarks and get AI coaching feedback.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: video + upload */}
        <div className="lg:col-span-2 space-y-6">
          {!done ? (
            <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Upload Video</h2>
              <UploadZone onComplete={handleComplete} />
            </div>
          ) : (
            blobUrl && (
              <div className="rounded-2xl border border-white/10 bg-black overflow-hidden">
                <div className="p-3 border-b border-white/5 flex items-center justify-between">
                  <span className="text-sm font-medium text-white">Your Swing</span>
                  <button
                    onClick={() => setDone(false)}
                    className="text-xs text-white/40 hover:text-white"
                  >
                    Upload different video
                  </button>
                </div>
                <VideoCanvas
                  src={localVideoUrl || blobUrl}
                  framesData={framesData}
                  visible={visible}
                  showSkeleton={showSkeleton}
                  showTrail={showTrail}
                  showControls
                  className="p-2"
                />
              </div>
            )
          )}

          {/* Swing selector for longer videos */}
          {done && hasMultipleSwings && (
            <SwingSelector
              allFrames={allFrames}
              selectedIndex={selectedSwing}
              onSelect={handleSwingSelect}
            />
          )}

          {/* Advanced-tier players care most about drift-vs-baseline; surface the
              baseline save as the lead action and keep the coaching panel below. */}
          {done && isAdvanced && (
            <div className="rounded-2xl border border-emerald-500/30 bg-gradient-to-br from-emerald-500/10 to-emerald-500/[0.02] p-5">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="text-white font-semibold text-base mb-1">
                    Your real wins are drift-focused
                  </p>
                  <p className="text-white/60 text-sm">
                    Save this and compare against future takes.
                  </p>
                  {baselineStatus === 'error' && baselineError && (
                    <p className="text-red-400 text-xs mt-2">{baselineError}</p>
                  )}
                  {baselineStatus === 'saved' && (
                    <p className="text-emerald-300 text-xs mt-2">
                      Saved. Future uploads will compare against this.
                    </p>
                  )}
                </div>
                {!authLoading && !user ? (
                  <Link
                    href="/login?next=/analyze"
                    className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-white text-sm font-semibold rounded-lg transition-colors flex-shrink-0"
                  >
                    Sign in to save
                  </Link>
                ) : baselineStatus === 'saved' ? (
                  <Link
                    href="/baseline/compare"
                    className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-white text-sm font-semibold rounded-lg transition-colors flex-shrink-0"
                  >
                    Compare new swing →
                  </Link>
                ) : (
                  <button
                    onClick={saveAsBaseline}
                    disabled={!canSaveBaseline || baselineStatus === 'saving'}
                    className={`px-4 py-2 text-sm font-semibold rounded-lg transition-colors flex-shrink-0 ${
                      !canSaveBaseline || baselineStatus === 'saving'
                        ? 'bg-white/10 text-white/40 cursor-not-allowed'
                        : 'bg-emerald-500 hover:bg-emerald-400 text-white'
                    }`}
                  >
                    {baselineStatus === 'saving' ? 'Saving...' : 'Save as baseline'}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Subtle nudge for users who skipped onboarding — one line, no CTA
              button. Only rendered when an analysis is ready so it sits near
              the coaching output where the tailoring would apply. */}
          {done && showProfileHint && (
            <p className="text-xs text-white/50">
              Not sure your level?{' '}
              <Link href="/profile" className="underline hover:text-white/80">
                Tell us for tailored advice →
              </Link>
            </p>
          )}

          {/* LLM coaching */}
          {done && (
            <LLMCoachingPanel frames={analysisFrames} />
          )}

          {/* Baseline CTA — the main pivot action. Hidden for advanced-tier
              users because the drift-focused card above already wires the
              same save-as-baseline action. */}
          {done && !isAdvanced && (
            <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4 flex items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-white font-medium text-sm">Is this your best day?</p>
                <p className="text-white/50 text-xs">
                  {!authLoading && !user
                    ? 'Sign in to save this as a baseline and track progress across future swings.'
                    : hasMultipleSwings && selectedSwing === null
                      ? 'Pick a single swing above to save it as a baseline.'
                      : baselineStatus === 'saved'
                        ? 'Saved. Upload future swings on the baseline compare page to track progress.'
                        : baselineStatus === 'error'
                          ? baselineError ?? 'Failed to save baseline.'
                          : 'Save this as a baseline and compare future swings against it.'}
                </p>
              </div>
              {!authLoading && !user ? (
                <Link
                  href="/login?next=/analyze"
                  className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-white text-sm font-semibold rounded-lg transition-colors flex-shrink-0"
                >
                  Sign in to save
                </Link>
              ) : baselineStatus === 'saved' ? (
                <Link
                  href="/baseline/compare"
                  className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-white text-sm font-semibold rounded-lg transition-colors flex-shrink-0"
                >
                  Compare new swing →
                </Link>
              ) : (
                <button
                  onClick={saveAsBaseline}
                  disabled={!canSaveBaseline || baselineStatus === 'saving'}
                  className={`px-4 py-2 text-sm font-semibold rounded-lg transition-colors flex-shrink-0 ${
                    !canSaveBaseline || baselineStatus === 'saving'
                      ? 'bg-white/10 text-white/40 cursor-not-allowed'
                      : 'bg-emerald-500 hover:bg-emerald-400 text-white'
                  }`}
                >
                  {baselineStatus === 'saving' ? 'Saving...' : 'Set as baseline'}
                </button>
              )}
            </div>
          )}

        </div>

        {/* Right: controls */}
        <div className="space-y-4">
          <JointTogglePanel />

          {framesData.length > 0 && (
            <div className="rounded-xl bg-white/5 border border-white/10 p-4">
              <h3 className="text-sm font-semibold text-white mb-2">Analysis Stats</h3>
              <div className="space-y-1.5 text-sm">
                <div className="flex justify-between text-white/60">
                  <span>Frames analyzed</span>
                  <span className="text-white font-mono">{framesData.length}</span>
                </div>
                <div className="flex justify-between text-white/60">
                  <span>Duration</span>
                  <span className="text-white font-mono">
                    {((framesData[framesData.length - 1]?.timestamp_ms ?? 0) / 1000).toFixed(1)}s
                  </span>
                </div>
                <div className="flex justify-between text-white/60">
                  <span>Joint data</span>
                  <span className="text-emerald-400 font-medium">Ready</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
