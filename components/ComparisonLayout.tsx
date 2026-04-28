'use client'

import { useCallback, useMemo } from 'react'
import VideoCanvas from './VideoCanvas'
import { renderPose, normalizeLandmarks } from './PoseRenderer'
import { useJointStore, useComparisonStore, useSyncStore } from '@/store'
import type { PoseFrame } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'
import { useRef, useEffect } from 'react'
import {
  detectSwingPhases,
  computeTimeMapping,
  getActivePhase,
  type SwingPhase,
} from '@/lib/syncAlignment'

const PHASE_LABELS: Record<SwingPhase, string> = {
  preparation: 'Preparation',
  backswing: 'Backswing',
  forward_swing: 'Forward Swing',
  contact: 'Contact',
  follow_through: 'Follow Through',
}

interface ComparisonLayoutProps {
  userBlobUrl: string
  userFrames: PoseFrame[]
  proFrames: PoseFrame[]
  proVideoUrl: string
  proName?: string
  // Caption for the LEFT video. Defaulted to "You" so legacy callers
  // keep the original behavior; the baseline-compare page passes
  // "Baseline" since the left video IS the saved baseline.
  userName?: string
}

export default function ComparisonLayout({
  userBlobUrl,
  userFrames,
  proFrames,
  proVideoUrl,
  proName = 'Pro',
  userName = 'You',
}: ComparisonLayoutProps) {
  const { visible, showSkeleton, showTrail, showRacket } = useJointStore()
  const { mode, setMode } = useComparisonStore()
  const { syncedTime, setSyncedTime, setTimeMapping, isPlaying, setIsPlaying } = useSyncStore()

  // Detect swing phases for both user and pro frame sequences
  const userPhases = useMemo(
    () => detectSwingPhases(userFrames),
    [userFrames]
  )
  const proPhases = useMemo(
    () => detectSwingPhases(proFrames),
    [proFrames]
  )

  // Compute time mapping from user time -> pro time based on matched phases.
  // Falls back to duration-ratio scaling when phase detection is insufficient.
  const timeMapping = useMemo(() => {
    if (userPhases.length < 2 || proPhases.length < 2) return null

    const userDurationMs =
      userFrames.length > 0
        ? userFrames[userFrames.length - 1].timestamp_ms
        : undefined
    const proDurationMs =
      proFrames.length > 0
        ? proFrames[proFrames.length - 1].timestamp_ms
        : undefined

    return computeTimeMapping(userPhases, proPhases, userDurationMs, proDurationMs)
  }, [userPhases, proPhases, userFrames, proFrames])

  // Keep the sync store's timeMapping in sync so other consumers can read it
  useEffect(() => {
    setTimeMapping(timeMapping)
    return () => setTimeMapping(null)
  }, [timeMapping, setTimeMapping])

  // Compute playback rate for the pro video so slow-motion clips play at a
  // speed that matches the user's real-time swing duration.
  const { setProPlaybackRate } = useSyncStore()
  const proPlaybackRate = useMemo(() => {
    // Use phase boundaries to calculate real swing duration for both videos.
    // Fall back to total frame duration if phase detection is insufficient.
    const userDurationMs =
      userPhases.length >= 2
        ? userPhases[userPhases.length - 1].timestampMs - userPhases[0].timestampMs
        : userFrames.length > 1
          ? userFrames[userFrames.length - 1].timestamp_ms - userFrames[0].timestamp_ms
          : 0

    const proDurationMs =
      proPhases.length >= 2
        ? proPhases[proPhases.length - 1].timestampMs - proPhases[0].timestampMs
        : proFrames.length > 1
          ? proFrames[proFrames.length - 1].timestamp_ms - proFrames[0].timestamp_ms
          : 0

    if (userDurationMs <= 0 || proDurationMs <= 0) return 1

    const ratio = proDurationMs / userDurationMs
    // Clamp to a reasonable range
    return Math.min(16, Math.max(0.25, ratio))
  }, [userPhases, proPhases, userFrames, proFrames])

  // Persist the computed playback rate so other components can read it
  useEffect(() => {
    setProPlaybackRate(proPlaybackRate)
    return () => setProPlaybackRate(1)
  }, [proPlaybackRate, setProPlaybackRate])

  // Compute the user video timestamp (in seconds) when the first detected phase
  // begins. The pro video will be held at its start frame until this point,
  // preventing stutter during the user's pre-swing dead time.
  const userSwingStartSec = useMemo(() => {
    if (userPhases.length > 0) {
      return userPhases[0].timestampMs / 1000
    }
    return 0
  }, [userPhases])

  // Determine current phase for the badge
  const activePhase = useMemo(
    () => getActivePhase(userPhases, syncedTime * 1000),
    [userPhases, syncedTime]
  )

  const handleTimeUpdate = useCallback(
    (t: number) => setSyncedTime(t),
    [setSyncedTime]
  )

  const handlePlayPause = useCallback(
    (p: boolean) => setIsPlaying(p),
    [setIsPlaying]
  )

  return (
    <div className="space-y-4">
      {/* Phase indicator badge */}
      <div className="flex items-center gap-3">
        {activePhase && (
          <span className="px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            {PHASE_LABELS[activePhase]}
          </span>
        )}

        {/* Sync status */}
        {timeMapping && (
          <span className="text-xs text-white/30" title="Videos are phase-aligned">
            Phase sync active
          </span>
        )}
      </div>

      {(
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <VideoCanvas
            src={userBlobUrl}
            framesData={userFrames}
            visible={visible}
            showSkeleton={showSkeleton}
            showTrail={showTrail}
            showRacket={showRacket}
            syncedTime={syncedTime}
            onTimeUpdate={handleTimeUpdate}
            onPlayPause={handlePlayPause}
            isPrimary
            label={userName}
          />
          <div className="relative">
            <VideoCanvas
              src={proVideoUrl}
              framesData={proFrames}
              visible={visible}
              showSkeleton={showSkeleton}
              showTrail={showTrail}
              showRacket={showRacket}
              syncedTime={syncedTime}
              onTimeUpdate={handleTimeUpdate}
              syncPlaying={isPlaying}
              isPrimary={false}
              isSecondary
              timeMapping={timeMapping}
              playbackRate={proPlaybackRate}
              proStartDelay={userSwingStartSec}
              label={proName}
            />
            {proPlaybackRate > 1.05 && (
              <span className="absolute top-2 right-2 z-10 px-2 py-0.5 rounded-full text-xs font-medium bg-white/20 text-white backdrop-blur-sm">
                {proPlaybackRate.toFixed(1)}x speed
              </span>
            )}
          </div>
        </div>
      )}

    </div>
  )
}

interface OverlayViewProps {
  userBlobUrl: string
  userFrames: PoseFrame[]
  proFrames: PoseFrame[]
  visible: Record<JointGroup, boolean>
  showSkeleton: boolean
  showTrail: boolean
  showRacket: boolean
  syncedTime: number
  onTimeUpdate: (t: number) => void
  onPlayPause: (playing: boolean) => void
  proName: string
  timeMapping: ((userTimeMs: number) => number) | null
  proPlaybackRate: number
}

function OverlayView({
  userBlobUrl,
  userFrames,
  proFrames,
  visible,
  showSkeleton,
  showTrail,
  showRacket,
  syncedTime,
  onTimeUpdate,
  onPlayPause,
  proName,
  timeMapping,
  proPlaybackRate,
}: OverlayViewProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-white/60">You</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-orange-500" />
          <span className="text-white/60">{proName}</span>
        </div>
      </div>

      <div className="relative">
        {/* User video with blue skeleton */}
        <VideoCanvas
          src={userBlobUrl}
          framesData={userFrames}
          visible={visible}
          showSkeleton={showSkeleton}
          showTrail={showTrail}
          showRacket={showRacket}
          overlayColor="rgba(59,130,246,0.9)"
          overlaySkeletonColor="rgba(59,130,246,0.5)"
          syncedTime={syncedTime}
          onTimeUpdate={onTimeUpdate}
          onPlayPause={onPlayPause}
          isPrimary
        />

        {/* Pro normalized skeleton overlay */}
        <ProSkeletonOverlay
          proFrames={proFrames}
          visible={visible}
          showSkeleton={showSkeleton}
          syncedTime={syncedTime}
          timeMapping={timeMapping}
        />

        {proPlaybackRate > 1.05 && (
          <span className="absolute top-2 right-2 z-10 px-2 py-0.5 rounded-full text-xs font-medium bg-white/20 text-white backdrop-blur-sm">
            Pro: {proPlaybackRate.toFixed(1)}x speed
          </span>
        )}
      </div>
    </div>
  )
}

interface ProSkeletonOverlayProps {
  proFrames: PoseFrame[]
  visible: Record<JointGroup, boolean>
  showSkeleton: boolean
  syncedTime: number
  timeMapping: ((userTimeMs: number) => number) | null
}

function ProSkeletonOverlay({
  proFrames,
  visible,
  showSkeleton,
  syncedTime,
  timeMapping,
}: ProSkeletonOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  // Use refs for values that change frequently to avoid recreating the rAF
  // loop on every syncedTime tick (which would cause constant teardown/restart).
  const syncedTimeRef = useRef(syncedTime)
  const visibleRef = useRef(visible)
  const showSkeletonRef = useRef(showSkeleton)
  const proFramesRef = useRef(proFrames)
  const timeMappingRef = useRef(timeMapping)

  // Sync refs outside of render via useEffect to satisfy React rules.
  useEffect(() => {
    syncedTimeRef.current = syncedTime
    visibleRef.current = visible
    showSkeletonRef.current = showSkeleton
    proFramesRef.current = proFrames
    timeMappingRef.current = timeMapping
  })

  // Keep canvas pixel dimensions in sync with the container element via
  // ResizeObserver. Canvas backing store gets DPR-scaled so retina
  // displays render the skeleton at native pixel sharpness instead of
  // the fuzzy 1x backing store the previous version produced. Mirrors
  // VideoCanvas's pattern (videoWidth × dpr + ctx.scale(dpr,dpr)).
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width && height) {
          const dpr = window.devicePixelRatio || 1
          canvas.width = Math.round(width * dpr)
          canvas.height = Math.round(height * dpr)
        }
      }
    })
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [])

  const getFrameAtTime = useCallback(
    (frames: PoseFrame[], timeMs: number): PoseFrame | null => {
      if (!frames.length) return null
      let closest = frames[0]
      let minDiff = Math.abs(frames[0].timestamp_ms - timeMs)
      for (const f of frames) {
        const diff = Math.abs(f.timestamp_ms - timeMs)
        if (diff < minDiff) {
          minDiff = diff
          closest = f
        }
      }
      return closest
    },
    []
  )

  useEffect(() => {
    const render = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // Convert backing-store dims back to logical (CSS-pixel) dims for
      // the draw call so PoseRenderer's hardcoded line widths / dot
      // radii stay constant in CSS pixels. ctx.setTransform resets any
      // accumulated scale before re-applying — without this, the scale
      // stacks on every rAF tick and the skeleton runs off the canvas
      // within a few frames.
      const dpr = window.devicePixelRatio || 1
      const logicalW = canvas.width / dpr
      const logicalH = canvas.height / dpr
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, logicalW, logicalH)

      // Apply time mapping if available: convert user time (s) -> pro time (ms)
      const userTimeMs = syncedTimeRef.current * 1000
      const proTimeMs = timeMappingRef.current
        ? timeMappingRef.current(userTimeMs)
        : userTimeMs

      const frame = getFrameAtTime(proFramesRef.current, proTimeMs)
      if (frame) {
        const normalized = normalizeLandmarks(frame)
        renderPose(ctx, normalized, logicalW, logicalH, {
          visible: visibleRef.current,
          showSkeleton: showSkeletonRef.current,
          color: 'rgba(249,115,22,0.9)',
          skeletonColor: 'rgba(249,115,22,0.5)',
        })
      }

      rafRef.current = requestAnimationFrame(render)
    }
    rafRef.current = requestAnimationFrame(render)
    return () => cancelAnimationFrame(rafRef.current)
  }, [getFrameAtTime])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
    />
  )
}
