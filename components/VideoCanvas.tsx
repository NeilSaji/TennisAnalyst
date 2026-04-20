'use client'

import { useRef, useEffect, useCallback, useState } from 'react'
import type { PoseFrame } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'
import { renderPose, normalizeLandmarks } from './PoseRenderer'
import { SwingPathTracer } from './SwingPathTracer'

interface VideoCanvasProps {
  src: string
  framesData: PoseFrame[]
  visible: Record<JointGroup, boolean>
  showSkeleton: boolean
  showTrail: boolean
  /** Render the racket-head trail (schema v2 only; no-op on legacy clips). */
  showRacket?: boolean
  overlayColor?: string
  overlaySkeletonColor?: string
  syncedTime?: number
  onTimeUpdate?: (t: number) => void
  onDurationChange?: (d: number) => void
  onPlayPause?: (playing: boolean) => void
  syncPlaying?: boolean
  isPrimary?: boolean
  isSecondary?: boolean
  timeMapping?: ((userTimeMs: number) => number) | null
  showControls?: boolean
  className?: string
  label?: string
  playbackRate?: number
  proStartDelay?: number
}

export default function VideoCanvas({
  src,
  framesData,
  visible,
  showSkeleton,
  showTrail,
  showRacket = true,
  overlayColor,
  overlaySkeletonColor,
  syncedTime,
  onTimeUpdate,
  onDurationChange,
  onPlayPause,
  syncPlaying,
  isPrimary = true,
  isSecondary = false,
  timeMapping = null,
  showControls = true,
  className = '',
  label,
  playbackRate = 1.0,
  proStartDelay = 0,
}: VideoCanvasProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  const rvfcHandleRef = useRef<number>(0)
  // Once sized for the current videoWidth/Height, avoid re-scaling the ctx
  // every frame (ctx.scale is cumulative).
  const canvasSizedRef = useRef<{ w: number; h: number; dpr: number } | null>(
    null
  )
  const tracerRef = useRef(new SwingPathTracer())
  const lastFrameIndexRef = useRef(-1)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [videoError, setVideoError] = useState(false)

  // Find the closest frame to a given video timestamp
  const getFrameAtTime = useCallback(
    (time: number): PoseFrame | null => {
      if (!framesData.length) return null
      const timeMs = time * 1000
      let closest = framesData[0]
      let minDiff = Math.abs(framesData[0].timestamp_ms - timeMs)
      for (const frame of framesData) {
        const diff = Math.abs(frame.timestamp_ms - timeMs)
        if (diff < minDiff) {
          minDiff = diff
          closest = frame
        }
      }
      return closest
    },
    [framesData]
  )

  // Main render loop. `mediaTime` is the video-clock time in seconds reported
  // by requestVideoFrameCallback; when absent (rAF fallback) we fall back to
  // the video element's currentTime.
  const renderFrame = useCallback(
    (mediaTime?: number) => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas) return
      if (videoError) return

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // DPR-aware sizing (mirrors ProSwingViewer.SkeletonPlayer). The canvas
      // backing store is videoWidth * dpr; drawing code below uses logical
      // pixels (canvas.width / dpr, canvas.height / dpr).
      const dpr = window.devicePixelRatio || 1
      const vw = video.videoWidth
      const vh = video.videoHeight
      if (vw && vh) {
        const sized = canvasSizedRef.current
        if (!sized || sized.w !== vw || sized.h !== vh || sized.dpr !== dpr) {
          canvas.width = vw * dpr
          canvas.height = vh * dpr
          // Apply once per (size, dpr) — ctx.scale is cumulative across calls.
          ctx.setTransform(1, 0, 0, 1, 0, 0)
          ctx.scale(dpr, dpr)
          canvasSizedRef.current = { w: vw, h: vh, dpr }
        }
      } else if (!canvas.width || !canvas.height) {
        canvas.width = canvas.offsetWidth
        canvas.height = canvas.offsetHeight
      }

      const logicalW = canvas.width / dpr
      const logicalH = canvas.height / dpr
      ctx.clearRect(0, 0, logicalW, logicalH)

      const t = mediaTime ?? video.currentTime
      const frame = getFrameAtTime(t)
      if (!frame) return

      const displayFrame = overlayColor ? normalizeLandmarks(frame) : frame

      // Update swing trail (advance buffer only when the underlying frame
      // changes so rapid re-renders don't stack duplicates).
      if (showTrail || showRacket) {
        if (frame.frame_index !== lastFrameIndexRef.current) {
          tracerRef.current.push(frame, logicalW, logicalH)
          lastFrameIndexRef.current = frame.frame_index
        }
        tracerRef.current.render(ctx, {
          showWristTrails: showTrail,
          showRacketTrail: showRacket,
        })
      }

      // Draw pose
      renderPose(ctx, displayFrame, logicalW, logicalH, {
        visible,
        showSkeleton,
        color: overlayColor,
        skeletonColor: overlaySkeletonColor,
      })
    },
    [
      visible,
      showSkeleton,
      showTrail,
      showRacket,
      overlayColor,
      overlaySkeletonColor,
      getFrameAtTime,
      videoError,
    ]
  )

  // Prefer requestVideoFrameCallback so we repaint in lockstep with decoded
  // video frames (frame-accurate overlay). Falls back to rAF on Safari/older
  // browsers where rVFC isn't implemented on HTMLVideoElement.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    let cancelled = false
    const hasRvfc =
      typeof HTMLVideoElement !== 'undefined' &&
      'requestVideoFrameCallback' in HTMLVideoElement.prototype

    if (hasRvfc) {
      const tick: VideoFrameRequestCallback = (_now, meta) => {
        if (cancelled) return
        renderFrame(meta?.mediaTime)
        rvfcHandleRef.current = video.requestVideoFrameCallback(tick)
      }
      // Paint once immediately so paused/first-load state is correct even
      // before a decoded frame arrives.
      renderFrame()
      rvfcHandleRef.current = video.requestVideoFrameCallback(tick)

      return () => {
        cancelled = true
        if (
          rvfcHandleRef.current &&
          typeof video.cancelVideoFrameCallback === 'function'
        ) {
          video.cancelVideoFrameCallback(rvfcHandleRef.current)
        }
        rvfcHandleRef.current = 0
      }
    }

    // rAF fallback (Safari, etc.)
    const loop = () => {
      if (cancelled) return
      renderFrame()
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => {
      cancelled = true
      cancelAnimationFrame(rafRef.current)
      rafRef.current = 0
    }
  }, [renderFrame, src])

  // Sync external time control
  useEffect(() => {
    if (syncedTime === undefined) return
    const video = videoRef.current
    if (!video) return

    // For secondary (pro) video: hold at start until the user's swing begins.
    // This prevents the pro video from stuttering during pre-swing dead time.
    if (isSecondary && proStartDelay > 0 && syncedTime < proStartDelay) {
      video.currentTime = 0
      video.pause()
      return
    }

    // For secondary (pro) video, apply phase-based time mapping if available.
    // syncedTime is in seconds (user video time); timeMapping expects/returns ms.
    let targetTime = syncedTime
    if (isSecondary && timeMapping) {
      targetTime = timeMapping(syncedTime * 1000) / 1000
    }

    // When paused, use precise seeking for scrubbing
    if (video.paused) {
      if (Math.abs(video.currentTime - targetTime) < 0.05) return
      video.currentTime = targetTime
      tracerRef.current.reset()
      lastFrameIndexRef.current = -1
      return
    }

    // When playing, adjust playbackRate to stay in sync (no seeking = no stutter)
    if (isSecondary) {
      const drift = targetTime - video.currentTime
      const baseRate = playbackRate

      if (Math.abs(drift) > 2) {
        // Way too far off - hard seek as last resort
        video.currentTime = targetTime
        video.playbackRate = baseRate
        tracerRef.current.reset()
        lastFrameIndexRef.current = -1
      } else if (drift > 0.3) {
        video.playbackRate = baseRate * 1.5
      } else if (drift > 0.1) {
        video.playbackRate = baseRate * 1.2
      } else if (drift < -0.3) {
        video.playbackRate = baseRate * 0.5
      } else if (drift < -0.1) {
        video.playbackRate = baseRate * 0.8
      } else {
        video.playbackRate = baseRate
      }
      return
    }

    // Primary video: standard sync
    if (Math.abs(video.currentTime - targetTime) < 0.05) return
    video.currentTime = targetTime
    tracerRef.current.reset()
    lastFrameIndexRef.current = -1
  }, [syncedTime, isSecondary, timeMapping, playbackRate, proStartDelay])

  // Apply playback rate to the video element (primary only; secondary is managed by sync effect)
  useEffect(() => {
    if (isSecondary) return
    const video = videoRef.current
    if (!video) return
    video.playbackRate = playbackRate
  }, [playbackRate, src, isSecondary])

  // Follow coordinated play/pause from the primary video
  useEffect(() => {
    if (syncPlaying === undefined) return
    const video = videoRef.current
    if (!video) return

    // Don't auto-play the secondary video while the user is still in pre-swing dead time
    if (isSecondary && proStartDelay > 0 && (syncedTime ?? 0) < proStartDelay) {
      video.pause()
      setPlaying(false)
      return
    }

    if (syncPlaying) {
      video.playbackRate = playbackRate
      video.play().catch(() => {})
      setPlaying(true)
    } else {
      video.pause()
      setPlaying(false)
    }
  }, [syncPlaying, playbackRate, isSecondary, proStartDelay, syncedTime])

  // Reset tracer, canvas sizing cache, and error state when src changes
  useEffect(() => {
    tracerRef.current.reset()
    lastFrameIndexRef.current = -1
    canvasSizedRef.current = null
    setVideoError(false)
  }, [src])

  const handleTimeUpdate = () => {
    const t = videoRef.current?.currentTime ?? 0
    setCurrentTime(t)
    // Only the primary video drives syncedTime to avoid a feedback loop
    // where two synced videos ping-pong time updates infinitely.
    if (isPrimary) {
      onTimeUpdate?.(t)
    }
  }

  const handleLoadedMetadata = () => {
    const d = videoRef.current?.duration ?? 0
    setDuration(d)
    onDurationChange?.(d)
  }

  const togglePlay = () => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      video.play()
      setPlaying(true)
      onPlayPause?.(true)
    } else {
      video.pause()
      setPlaying(false)
      onPlayPause?.(false)
    }
  }

  const handleScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
    const t = parseFloat(e.target.value)
    if (videoRef.current) videoRef.current.currentTime = t
    tracerRef.current.reset()
    lastFrameIndexRef.current = -1
    if (isPrimary) onTimeUpdate?.(t)
  }

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      {label && (
        <div className="text-sm font-semibold text-center text-white/80 bg-black/40 rounded px-2 py-1">
          {label}
        </div>
      )}
      <div className="relative rounded-xl overflow-hidden bg-black">
        <video
          ref={videoRef}
          src={src}
          className="w-full block"
          playsInline
          muted
          loop
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onError={() => setVideoError(true)}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
        {videoError && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80">
            <p className="text-red-400 text-sm">Failed to load video</p>
          </div>
        )}
      </div>

      {showControls && (
        <div className="flex items-center gap-3 px-1">
          <button
            onClick={togglePlay}
            className="w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white transition-colors flex-shrink-0"
            aria-label={playing ? 'Pause' : 'Play'}
          >
            {playing ? (
              <svg viewBox="0 0 24 24" className="w-4 h-4 fill-current">
                <rect x="6" y="4" width="4" height="16" />
                <rect x="14" y="4" width="4" height="16" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className="w-4 h-4 fill-current">
                <polygon points="5,3 19,12 5,21" />
              </svg>
            )}
          </button>

          <input
            type="range"
            min={0}
            max={duration || 1}
            step={0.01}
            value={currentTime}
            onChange={handleScrub}
            className="flex-1 accent-emerald-400 h-1 rounded-full"
          />

          <span className="text-xs text-white/60 font-mono tabular-nums flex-shrink-0">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>
      )}
    </div>
  )
}
