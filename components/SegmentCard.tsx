'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import type { VideoSegment } from '@/lib/supabase'

// Shot types the baseline table accepts. Matches the CHECK constraint in
// lib/db/004_user_baselines.sql. Classifier output includes 'unknown' and
// 'idle' too -- those aren't selectable as an override because they can't
// be saved to the baselines row.
const BASELINE_SHOTS = ['forehand', 'backhand', 'serve', 'volley', 'slice'] as const
type BaselineShot = (typeof BASELINE_SHOTS)[number]

function isBaselineShot(v: string | undefined | null): v is BaselineShot {
  return typeof v === 'string' && (BASELINE_SHOTS as readonly string[]).includes(v)
}

export interface SegmentCardSaveOverride {
  shotType: BaselineShot
  label?: string
}

interface SegmentCardProps {
  segment: VideoSegment
  blobUrl: string
  onSave: (segmentId: string, override: SegmentCardSaveOverride) => void | Promise<void>
  saving?: boolean
  saved?: boolean
  canSave?: boolean
  errorMessage?: string | null
  // When false, the save button renders as a "sign in" prompt -- the grid
  // passes auth state down so each card reflects the same gate.
  signedIn?: boolean
}

// One per-segment card on the multi-shot analyze flow. The card is
// deliberately minimal: a seek-to-midpoint video preview, a shot-type
// override dropdown, an optional label input, and a Save button wired
// up to the grid's onSaveAsBaseline handler.
export default function SegmentCard({
  segment,
  blobUrl,
  onSave,
  saving = false,
  saved = false,
  canSave = true,
  errorMessage = null,
  signedIn = true,
}: SegmentCardProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const initialShot = isBaselineShot(segment.shot_type) ? segment.shot_type : 'forehand'
  const [overrideShot, setOverrideShot] = useState<BaselineShot>(initialShot)
  const [label, setLabel] = useState<string>(
    segment.label?.trim() || `Segment #${segment.segment_index + 1} — ${overrideShot}`,
  )
  const [labelDirty, setLabelDirty] = useState(false)

  const midpointMs = useMemo(
    () => Math.round((segment.start_ms + segment.end_ms) / 2),
    [segment.start_ms, segment.end_ms],
  )
  const startSec = (segment.start_ms / 1000).toFixed(1)
  const endSec = (segment.end_ms / 1000).toFixed(1)
  const durationSec = ((segment.end_ms - segment.start_ms) / 1000).toFixed(1)

  // Seek the preview <video> to the segment midpoint so the card thumbnail
  // reflects something close to the contact frame. We only need to do this
  // once per blob load.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const handleLoaded = () => {
      try {
        video.currentTime = midpointMs / 1000
      } catch {
        // Some browsers throw if duration isn't known yet; the subsequent
        // 'durationchange' handler will retry.
      }
    }
    video.addEventListener('loadedmetadata', handleLoaded)
    return () => {
      video.removeEventListener('loadedmetadata', handleLoaded)
    }
  }, [midpointMs])

  // Keep the default label in sync with the selected shot type until the
  // user explicitly edits the input. After that we respect their input.
  useEffect(() => {
    if (labelDirty) return
    setLabel(
      segment.label?.trim() || `Segment #${segment.segment_index + 1} — ${overrideShot}`,
    )
  }, [overrideShot, labelDirty, segment.label, segment.segment_index])

  const playSegment = () => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = segment.start_ms / 1000
    void video.play()
  }

  // Pause when we pass end_ms during a "Play this segment" run.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const handleTime = () => {
      if (video.currentTime * 1000 >= segment.end_ms) {
        video.pause()
        video.currentTime = midpointMs / 1000
      }
    }
    video.addEventListener('timeupdate', handleTime)
    return () => video.removeEventListener('timeupdate', handleTime)
  }, [segment.end_ms, midpointMs])

  const classifierShot = segment.shot_type
  const overrideDiffers = classifierShot !== overrideShot
  const confidencePct = Number.isFinite(segment.confidence)
    ? Math.round((segment.confidence ?? 0) * 100)
    : null

  const handleSave = async () => {
    if (!canSave || saving) return
    await onSave(segment.id, {
      shotType: overrideShot,
      label: label.trim() || undefined,
    })
  }

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 flex flex-col gap-3">
      <div className="rounded-lg overflow-hidden bg-black aspect-video relative">
        <video
          ref={videoRef}
          src={blobUrl}
          preload="metadata"
          muted
          playsInline
          className="w-full h-full object-contain"
        />
        <button
          type="button"
          onClick={playSegment}
          className="absolute bottom-2 right-2 px-2 py-1 rounded bg-black/70 hover:bg-black text-white text-xs font-medium"
        >
          Play this segment
        </button>
      </div>

      <div className="flex items-center justify-between text-xs text-white/60">
        <span>
          #{segment.segment_index + 1} &middot; {startSec}s to {endSec}s ({durationSec}s)
        </span>
        {confidencePct !== null && (
          <span className="text-white/50">{confidencePct}% conf</span>
        )}
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-[11px] uppercase tracking-wide text-white/40">
          Shot type
        </label>
        <select
          value={overrideShot}
          onChange={(e) => setOverrideShot(e.target.value as BaselineShot)}
          className="bg-white/5 border border-white/10 text-white text-sm rounded px-2 py-1"
          aria-label="Shot type override"
        >
          {BASELINE_SHOTS.map((s) => (
            <option key={s} value={s} className="bg-neutral-900">
              {s}
            </option>
          ))}
        </select>
        {overrideDiffers && (
          <p className="text-[11px] text-amber-300/80">
            Classified as {classifierShot}, saving as {overrideShot}
          </p>
        )}
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-[11px] uppercase tracking-wide text-white/40">
          Label
        </label>
        <input
          value={label}
          onChange={(e) => {
            setLabelDirty(true)
            setLabel(e.target.value)
          }}
          className="bg-white/5 border border-white/10 text-white text-sm rounded px-2 py-1"
          placeholder="Label"
          maxLength={120}
        />
      </div>

      {errorMessage && (
        <p className="text-xs text-red-400">{errorMessage}</p>
      )}
      {saved && (
        <p className="text-xs text-emerald-300">Saved as baseline.</p>
      )}

      {signedIn ? (
        <button
          type="button"
          onClick={handleSave}
          disabled={!canSave || saving}
          className={`px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${
            !canSave || saving
              ? 'bg-white/10 text-white/40 cursor-not-allowed'
              : 'bg-emerald-500 hover:bg-emerald-400 text-white'
          }`}
        >
          {saving ? 'Saving...' : saved ? 'Saved' : 'Save as baseline'}
        </button>
      ) : (
        <p className="text-xs text-white/50">Sign in to save this as a baseline.</p>
      )}
    </div>
  )
}
