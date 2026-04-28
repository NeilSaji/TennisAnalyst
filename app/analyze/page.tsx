'use client'

import { useState, useMemo, useEffect } from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import JointTogglePanel from '@/components/JointTogglePanel'
import SwingSelector from '@/components/SwingSelector'
import SegmentPickerGrid from '@/components/SegmentPickerGrid'
import SwingBaselineGrid, { type SwingBaselineSaveOverride } from '@/components/SwingBaselineGrid'
import type { SegmentCardSaveOverride } from '@/components/SegmentCard'
import { usePoseStore, useJointStore } from '@/store'
import { detectSwings } from '@/lib/jointAngles'
import { useUser } from '@/hooks/useUser'
import { useProfile } from '@/hooks/useProfile'
import type { SwingSegment } from '@/lib/jointAngles'
import type { PoseFrame, VideoSegment } from '@/lib/supabase'

const VALID_BASELINE_SHOTS = new Set(['forehand', 'backhand', 'serve', 'volley', 'slice'])

// Dynamic import to avoid SSR issues with MediaPipe WASM
const UploadZone = dynamic(() => import('@/components/UploadZone'), { ssr: false })
const VideoCanvas = dynamic(() => import('@/components/VideoCanvas'), { ssr: false })
const LLMCoachingPanel = dynamic(() => import('@/components/LLMCoachingPanel'), { ssr: false })

export default function AnalyzePage() {
  const { framesData, blobUrl, localVideoUrl, sessionId, shotType } = usePoseStore()
  const { visible, showSkeleton, showRacket, showAngles } = useJointStore()
  const [done, setDone] = useState(false)
  const [allFrames, setAllFrames] = useState<PoseFrame[]>([])
  const [selectedSwing, setSelectedSwing] = useState<number | null>(null)
  const [baselineStatus, setBaselineStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [baselineError, setBaselineError] = useState<string | null>(null)
  const { user, loading: authLoading } = useUser()
  const { profile, skipped, loading: profileLoading } = useProfile()
  const isAdvanced = !profileLoading && profile?.skill_tier === 'advanced'
  // Always render both arms. Earlier the forehand view suppressed the
  // off-hand elbow/wrist on the theory that a one-armed shot makes the
  // off-arm visual noise — in practice it reads as "the tracker can
  // only find one arm," which is the opposite of what we want in a
  // coaching demo. Dual-arm plumbing stays in renderPose /
  // renderAngleLabels for comparison views that want it; the analyze
  // page just passes null.
  const dominantHand: 'left' | 'right' | null = null
  // Show the "set your profile" hint only to users who explicitly skipped
  // onboarding — onboarded users already have a tier, anons can't persist one.
  const showProfileHint = !profileLoading && skipped && !profile

  // Per-segment baseline save state for the multi-shot grid. Kept on the
  // analyze page (not inside the grid) so optimistic UI survives grid
  // remounts and so multiple cards can't be mid-save simultaneously.
  const [segments, setSegments] = useState<VideoSegment[]>([])
  const [savingSegmentId, setSavingSegmentId] = useState<string | null>(null)
  const [savedSegmentIds, setSavedSegmentIds] = useState<Set<string>>(new Set())
  const [errorBySegmentId, setErrorBySegmentId] = useState<Record<string, string | null>>({})

  // Prefer persisted live-session swings (from video_segments) over a fresh
  // detectSwings() pass. The two formulas can disagree by ±1–2 swings — live
  // sees them at 15fps as they're emitted, post-hoc detectSwings re-derives
  // them from smoothed frames at the persisted FPS. Showing one count in
  // live and a different count in /analyze breaks trust; the persisted list
  // is the canonical record of what the player just did.
  const swings = useMemo<SwingSegment[]>(() => {
    if (segments.length > 0 && allFrames.length > 0) {
      const fromSegments: SwingSegment[] = segments
        .slice()
        .sort((a, b) => a.segment_index - b.segment_index)
        .map((seg, i) => {
          const startFrame = Math.max(0, Math.min(allFrames.length - 1, seg.start_frame))
          const endFrame = Math.max(startFrame, Math.min(allFrames.length - 1, seg.end_frame))
          const sliced = allFrames.slice(startFrame, endFrame + 1)
          return {
            index: i + 1,
            startFrame,
            endFrame,
            startMs: seg.start_ms,
            endMs: seg.end_ms,
            // No persisted peak frame — fall back to the midpoint of the
            // segment (sufficient for VideoCanvas's contact-frame coloring;
            // it just needs *a* frame inside the swing).
            peakFrame: startFrame + Math.floor(sliced.length / 2),
            frames: sliced,
          }
        })
        .filter((s) => s.frames.length > 0)
      if (fromSegments.length > 0) return fromSegments
    }
    return detectSwings(allFrames)
  }, [segments, allFrames])
  const hasMultipleSwings = swings.length > 1

  // Contact frame for the currently-displayed swing. When multiple swings
  // exist we use the selected swing's peakFrame; otherwise the single swing's
  // peakFrame. Passed to VideoCanvas so the angle badges color-code against
  // pro benchmarks only on the contact frame — avoids a sea of red badges
  // during backswing/follow-through where the angles legitimately span a
  // wide range.
  const contactFrameIndex = useMemo<number | null>(() => {
    if (swings.length === 0) return null
    const active = selectedSwing != null ? swings[selectedSwing - 1] : swings[0]
    if (!active) return null
    // peakFrame is already an index into allFrames, and VideoCanvas uses
    // frame.frame_index for matching — they line up because extraction
    // assigns frame_index monotonically from 0.
    const peakFrame = active.frames[active.peakFrame - active.startFrame]
    return peakFrame?.frame_index ?? null
  }, [swings, selectedSwing])

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

  // Fetch video_segments rows for this session so we know whether to
  // render the multi-shot picker grid. The endpoint returns [] for
  // single-shot sessions, so segments.length > 1 is the signal.
  useEffect(() => {
    if (!done || !sessionId) {
      setSegments([])
      return
    }
    let ignore = false
    fetch(`/api/segments/${sessionId}`, { credentials: 'same-origin' })
      .then(async (res) => (res.ok ? ((await res.json()) as VideoSegment[]) : []))
      .then((rows) => {
        if (!ignore) setSegments(Array.isArray(rows) ? rows : [])
      })
      .catch(() => {
        if (!ignore) setSegments([])
      })
    return () => {
      ignore = true
    }
  }, [done, sessionId])

  // Reset per-segment save state when the user uploads a new video so
  // saved/error badges from the previous session don't bleed through.
  useEffect(() => {
    if (!done) {
      setSavingSegmentId(null)
      setSavedSegmentIds(new Set())
      setErrorBySegmentId({})
    }
  }, [done])

  const handleSaveSegmentAsBaseline = async (
    segmentId: string,
    override: SegmentCardSaveOverride,
  ) => {
    if (!sessionId || savingSegmentId) return
    setSavingSegmentId(segmentId)
    setErrorBySegmentId((prev) => ({ ...prev, [segmentId]: null }))
    try {
      const res = await fetch('/api/baselines/from-segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          sessionId,
          segmentId,
          shotTypeOverride: override.shotType,
          label: override.label,
        }),
      })
      if (!res.ok) {
        const msg = (await res.text()) || `HTTP ${res.status}`
        setErrorBySegmentId((prev) => ({ ...prev, [segmentId]: msg }))
        return
      }
      setSavedSegmentIds((prev) => {
        const next = new Set(prev)
        next.add(segmentId)
        return next
      })
    } catch (err) {
      setErrorBySegmentId((prev) => ({
        ...prev,
        [segmentId]: err instanceof Error ? err.message : 'Failed to save baseline',
      }))
    } finally {
      setSavingSegmentId((current) => (current === segmentId ? null : current))
    }
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

  // Per-swing baseline save state. The SwingBaselineGrid lets the user
  // save N independent swings from a long rally clip (e.g. 15 forehands
  // in one upload) — each becomes its own baseline row in user_baselines.
  // The grid does not fetch DB segments; it consumes the in-memory
  // detectSwings() output, so no Railway / segmentation pipeline is
  // required for this flow.
  const [savingSwingIndex, setSavingSwingIndex] = useState<number | null>(null)
  const [savedSwingIndices, setSavedSwingIndices] = useState<Set<number>>(new Set())
  const [errorBySwingIndex, setErrorBySwingIndex] = useState<Record<number, string | null>>({})

  // Reset per-swing save state when a new clip is uploaded so a previous
  // session's saved/saving flags don't leak into the new clip's grid.
  useEffect(() => {
    setSavingSwingIndex(null)
    setSavedSwingIndices(new Set())
    setErrorBySwingIndex({})
  }, [blobUrl])

  const handleSaveSwingAsBaseline = async (
    swingIndex: number,
    override: SwingBaselineSaveOverride,
  ) => {
    const swing = swings[swingIndex - 1]
    if (!swing || !blobUrl) return

    setSavingSwingIndex(swingIndex)
    setErrorBySwingIndex((prev) => ({ ...prev, [swingIndex]: null }))
    try {
      const keypointsJson = {
        fps_sampled: 30,
        frame_count: swing.frames.length,
        frames: swing.frames,
      }
      const res = await fetch('/api/baselines', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          blobUrl,
          shotType: override.shotType,
          keypointsJson,
          label: override.label,
          sourceSessionId: sessionId ?? undefined,
        }),
      })
      if (!res.ok) {
        const msg = (await res.text()) || `HTTP ${res.status}`
        setErrorBySwingIndex((prev) => ({ ...prev, [swingIndex]: msg }))
        return
      }
      setSavedSwingIndices((prev) => {
        const next = new Set(prev)
        next.add(swingIndex)
        return next
      })
    } catch (err) {
      setErrorBySwingIndex((prev) => ({
        ...prev,
        [swingIndex]: err instanceof Error ? err.message : 'Failed to save baseline',
      }))
    } finally {
      setSavingSwingIndex((current) => (current === swingIndex ? null : current))
    }
  }

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
                  showRacket={showRacket}
                  showAngles={showAngles}
                  shotType={shotType}
                  contactFrameIndex={contactFrameIndex}
                  dominantHand={dominantHand}
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

          {/* Multi-swing baseline grid. Renders one card per detected
              swing so a long rally clip (e.g. 15 forehands in a row)
              produces 15 independently-saveable baselines. Uses the
              client-side detectSwings() output directly — no Railway
              / video-segments dependency. Renders only when there's
              at least one swing AND the user has a blob URL to
              reference; the existing single-CTA path below still
              handles single-swing clips. */}
          {done && hasMultipleSwings && blobUrl && (
            <SwingBaselineGrid
              swings={swings}
              blobUrl={blobUrl}
              defaultShotType={shotType ?? 'forehand'}
              signedIn={!!user}
              onSaveSwing={handleSaveSwingAsBaseline}
              savingSwingIndex={savingSwingIndex}
              savedSwingIndices={savedSwingIndices}
              errorBySwingIndex={errorBySwingIndex}
            />
          )}

          {/* Multi-shot picker. Rendered whenever the server detected >1
              segment for this session. Users can save any single segment
              as a baseline via /api/baselines/from-segment. The existing
              "Save as baseline" CTAs below are left alone per spec. */}
          {done && sessionId && blobUrl && segments.length > 1 && (
            <SegmentPickerGrid
              sessionId={sessionId}
              segments={segments}
              blobUrl={blobUrl}
              onSaveAsBaseline={handleSaveSegmentAsBaseline}
              signedIn={!!user}
              savingSegmentId={savingSegmentId}
              savedSegmentIds={savedSegmentIds}
              errorBySegmentId={errorBySegmentId}
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

          {framesData.length > 0 && (() => {
            // Joint-centric stats. Racket detection was pulled from the
            // demo (it was firing the wrist fallback ~97% of the time
            // and reading as "jittery racket line" rather than a real
            // signal). What we DO have reliably: per-frame joint
            // visibility + computed angles — those are what the overlay
            // and the LLM coach consume.
            const confidentFrames = framesData.filter((f) => {
              const ls = f.landmarks.find((l) => l.id === 11)
              const rs = f.landmarks.find((l) => l.id === 12)
              return (
                (ls?.visibility ?? 0) > 0.5 &&
                (rs?.visibility ?? 0) > 0.5
              )
            }).length
            const anglesPct =
              framesData.length === 0
                ? 0
                : Math.round((confidentFrames / framesData.length) * 100)
            return (
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
                  <div className="flex justify-between text-white/60">
                    <span>Upper-body tracking</span>
                    <span
                      className={
                        anglesPct >= 70
                          ? 'text-emerald-400 font-mono'
                          : 'text-amber-400 font-mono'
                      }
                    >
                      {anglesPct}%
                    </span>
                  </div>
                </div>
                {anglesPct < 70 && (
                  <p className="text-[11px] text-amber-300/80 mt-2">
                    Your upper body wasn&apos;t consistently visible in the
                    frame. For best joint-angle analysis, film
                    perpendicular to your baseline with your full body in
                    view.
                  </p>
                )}
              </div>
            )
          })()}
        </div>
      </div>
    </div>
  )
}
