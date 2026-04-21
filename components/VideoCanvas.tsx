'use client'

import { useRef, useEffect, useCallback, useState } from 'react'
import type { PoseFrame } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'
import { renderPose, normalizeLandmarks } from './PoseRenderer'
import { SwingPathTracer } from './SwingPathTracer'

// Nearest-neighbor lookup: return the sampled pose frame whose timestamp
// is closest to the requested time. We deliberately do NOT interpolate
// between bracketing frames — interpolation synthesizes a position the
// detector never actually saw, which reads to the user as the overlay
// "predicting" joints between samples. With zero-phase (filtfilt) smoothing
// upstream, nearest-neighbor at 30fps sample rate stays visibly locked.
export function getFrameAtTime(frames: PoseFrame[], timeSec: number): PoseFrame | null {
  if (!frames.length) return null
  const timeMs = timeSec * 1000
  if (timeMs <= frames[0].timestamp_ms) return frames[0]
  const last = frames[frames.length - 1]
  if (timeMs >= last.timestamp_ms) return last
  for (let i = 1; i < frames.length; i++) {
    const cur = frames[i]
    if (cur.timestamp_ms >= timeMs) {
      const prev = frames[i - 1]
      const dPrev = timeMs - prev.timestamp_ms
      const dCur = cur.timestamp_ms - timeMs
      return dCur < dPrev ? cur : prev
    }
  }
  return last
}

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

  // Interpolates between the two bracketing sampled frames (see
  // getFrameAtTime at module scope). Eliminates the ~1 decoded-frame lag that
  // nearest-neighbor lookup produced at 60fps playback against 30fps keypoints.
  const pickFrameAtTime = useCallback(
    (time: number): PoseFrame | null => getFrameAtTime(framesData, time),
    [framesData],
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
      const frame = pickFrameAtTime(t)
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
      pickFrameAtTime,
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

  // videoEl.duration is untrustworthy — broken moov boxes and MediaRecorder
  // uploads report 0, NaN, or Infinity even when the file plays end-to-end.
  // Keypoints_json carries a reliable span since Railway walks every frame
  // during extraction. Displayed duration = max of the two trustworthy
  // signals, falling back to whichever is defined.
  const recomputeDuration = useCallback(() => {
    const video = videoRef.current
    const poseDurSec = framesData.length
      ? framesData[framesData.length - 1].timestamp_ms / 1000
      : 0
    const videoDur = video?.duration
    const videoDurOk =
      typeof videoDur === 'number' && Number.isFinite(videoDur) && videoDur > 0
    const trusted = videoDurOk ? Math.max(videoDur, poseDurSec) : poseDurSec
    setDuration(trusted)
    // Avoid notifying the parent with 0 on mount before either source
    // resolves — the original handleLoadedMetadata only fired once metadata
    // was present, and downstream consumers rely on positive values.
    if (trusted > 0) onDurationChange?.(trusted)
  }, [framesData, onDurationChange])

  const handleLoadedMetadata = () => {
    const video = videoRef.current
    // Chromium MediaRecorder bug: duration stays Infinity until the
    // playhead is seeked past the end, which forces the moov recompute.
    // durationchange fires with the real number, which re-runs recompute.
    if (video && !Number.isFinite(video.duration)) {
      video.currentTime = 1e101
      video.currentTime = 0
    }
    recomputeDuration()
  }

  const handleVideoDurationChange = () => {
    recomputeDuration()
  }

  // Pose data may arrive after loadedmetadata (props stream in). Recompute
  // duration when framesData lands so the scrubber max widens to the real
  // clip length even if videoEl.duration was wrong.
  useEffect(() => {
    recomputeDuration()
  }, [recomputeDuration])

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
          onDurationChange={handleVideoDurationChange}
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
