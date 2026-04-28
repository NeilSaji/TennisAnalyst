'use client'

import { useRef, useState, useEffect } from 'react'
import { upload } from '@vercel/blob/client'
import { usePoseStore } from '@/store'
import { usePoseExtractor } from '@/hooks/usePoseExtractor'
import { getPoseLandmarker } from '@/lib/mediapipe'
import { extractPoseViaRailway, RailwayExtractError } from '@/lib/poseExtractionRailway'
import type { ExtractResult } from '@/lib/poseExtraction'
import type { PoseFrame } from '@/lib/supabase'

// Gate the server-side (Railway) extraction path behind an env flag so a
// rollout can be toggled without redeploying code. When unset / false the
// upload flow keeps using in-browser MediaPipe exactly as before.
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

const SHOT_TYPES = ['forehand', 'backhand', 'serve', 'volley'] as const

interface UploadZoneProps {
  onComplete?: (blobUrl: string, frames: PoseFrame[]) => void
}

export default function UploadZone({ onComplete }: UploadZoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [shotType, setShotType] = useState<string>('forehand')
  const [statusMsg, setStatusMsg] = useState('')
  const [overallProgress, setOverallProgress] = useState(0)
  const [busy, setBusy] = useState(false)
  // Whether the MediaPipe model is currently downloading. Drives the
  // indeterminate-spinner UI so a 15-20s model load doesn't read as a
  // stuck progress bar (root cause of the "stuck at 25%" bug class).
  const [modelLoading, setModelLoading] = useState(false)
  // Last successfully-uploaded file, kept so the retry button can re-run
  // pose extraction without re-uploading to Vercel Blob (saves the user
  // ~30s on every retry).
  const [lastFile, setLastFile] = useState<File | null>(null)
  const [showRetry, setShowRetry] = useState(false)

  const {
    setFramesData,
    setBlobUrl,
    setLocalVideoUrl,
    setProcessing,
    isProcessing,
    setShotType: persistShotType,
    setExtractorBackend,
    setFallbackReason,
  } = usePoseStore()

  const { extract, progress: extractProgress, error: extractError, isProcessing: extracting, abort } = usePoseExtractor()

  // Reset only the processing flag when UploadZone mounts fresh (e.g. after back-button).
  // Don't clear framesData/blobUrl/localVideoUrl - those are needed by VideoCanvas.
  useEffect(() => {
    if (isProcessing) {
      setProcessing(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Remap 0..100 extractor progress into the 25..90 band of our overall bar
  useEffect(() => {
    if (!extracting) return
    setOverallProgress(25 + Math.round((extractProgress / 100) * 65))
  }, [extractProgress, extracting])

  // Surface extractor errors and arm the retry button so the user can
  // re-run extraction without re-uploading the file.
  useEffect(() => {
    if (extractError) {
      setStatusMsg(extractError)
      setShowRetry(true)
    }
  }, [extractError])

  const processVideo = async (file: File, opts: { skipUpload?: boolean; cachedBlobUrl?: string } = {}) => {
    setBusy(true)
    setProcessing(true)
    setShowRetry(false)
    setOverallProgress(0)
    setStatusMsg('Uploading video...')

    // 1. Upload directly to Vercel Blob via a signed client token. Going
    //    through the API route would hit Vercel's serverless body limit
    //    (4.5MB Hobby / ~100MB Pro) and fail for any real swing clip.
    //
    // Retry path: skipUpload=true reuses the cached blob URL from the
    // previous attempt so a failed extraction doesn't re-upload the
    // (potentially large) file. Saves ~30s per retry.
    let blobUrl: string
    if (opts.skipUpload && opts.cachedBlobUrl) {
      blobUrl = opts.cachedBlobUrl
      setBlobUrl(blobUrl)
    } else {
      try {
        const blobPath = `videos/${Date.now()}-${safeBlobFilename(file.name)}`
        const blob = await upload(blobPath, file, {
          access: 'public',
          handleUploadUrl: '/api/upload',
          contentType: file.type,
        })
        blobUrl = blob.url
        setBlobUrl(blobUrl)
      } catch {
        setStatusMsg('Upload failed. Please try again.')
        setShowRetry(true)
        setProcessing(false)
        setBusy(false)
        return
      }
    }

    // Cache the file + blob URL on the component so the retry button
    // can call back into processVideo without re-uploading.
    setLastFile(file)

    let result: ExtractResult | null = null
    let usedRailway = false
    let pendingSessionId: string | null = null
    let railwayFailReason: string | null = null

    // Railway path: create the session row FIRST (so Railway has something
    // to write back to), then ask Railway to extract, then poll. On any
    // failure we fall through to the browser MediaPipe path below — the
    // user should never see an upload die because Railway had a bad day.
    if (USE_RAILWAY_EXTRACT) {
      setStatusMsg('Queuing server-side extraction...')
      setOverallProgress(20)
      try {
        const pendingRes = await fetch('/api/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ blobUrl, shotType }),
        })
        if (!pendingRes.ok) {
          // Surface the underlying error detail (e.g. Supabase RLS message,
          // 'Missing service key' env failure) so the chip shows *why*,
          // not just the HTTP status. Falls back to the raw body text if
          // the response isn't JSON.
          let detail = ''
          try {
            const errBody = await pendingRes.json()
            detail = errBody?.detail || errBody?.error || ''
          } catch {
            try {
              detail = await pendingRes.text()
            } catch {
              detail = ''
            }
          }
          throw new Error(
            `create-pending failed: ${pendingRes.status}${detail ? ` (${detail})` : ''}`,
          )
        }
        const pendingBody = await pendingRes.json()
        pendingSessionId = pendingBody.sessionId

        setStatusMsg('Analyzing pose on the server...')
        result = await extractPoseViaRailway({
          sessionId: pendingSessionId!,
          blobUrl,
          onProgress: (pct) => setOverallProgress(20 + Math.round((pct / 100) * 70)),
        })
        usedRailway = true
      } catch (err) {
        // Any failure falls back to in-browser extraction silently.
        // Log the reason so we can tell 'Railway not configured' (normal
        // during staging) from 'Railway errored out' (needs attention).
        // Concatenate reason + the underlying Error.message so the
        // diagnostic chip shows BOTH the category ('error-status') and
        // the actual Railway error text ('OOM', 'CUDA out of memory',
        // 'YOLO failed: ...'). The category alone is rarely actionable.
        if (err instanceof RailwayExtractError) {
          console.info(
            '[UploadZone] Railway path unavailable, falling back to browser:',
            err.reason, err.message,
          )
          railwayFailReason =
            err.message && err.message !== err.reason
              ? `${err.reason}: ${err.message}`
              : err.reason
        } else {
          console.error('[UploadZone] Railway path errored, falling back to browser:', err)
          railwayFailReason = err instanceof Error ? err.message : 'unknown'
        }
        result = null
      }
    }

    // Browser MediaPipe fallback (or primary path when the feature flag
    // is off). Only runs when Railway didn't produce a result —
    // Railway-success users skip the 16MB MediaPipe WASM download
    // entirely. Model load only happens here (lazily) so the first-paint
    // experience for Railway users stays fast.
    if (!result) {
      setOverallProgress(15)
      setStatusMsg('Loading pose model...')
      // Explicit model-load step. First-time users wait ~5-15s for the
      // pose_landmarker_heavy weights to download from MediaPipe's CDN.
      // Indeterminate spinner is rendered while this resolves so the
      // wait reads as "actively loading", not "stuck progress bar".
      setModelLoading(true)
      try {
        await getPoseLandmarker()
      } catch {
        setStatusMsg('Failed to load pose model. Check your connection and tap Retry.')
        setShowRetry(true)
        setModelLoading(false)
        setProcessing(false)
        setBusy(false)
        return
      }
      setModelLoading(false)

      setOverallProgress(25)
      setStatusMsg('Analyzing pose from video...')
      result = await extract(file)
      if (!result) {
        // Hook handled status/error; just release the processing lock.
        // showRetry is already set by the extractError effect.
        setProcessing(false)
        setBusy(false)
        return
      }
      // If we ended up here AFTER attempting Railway, the user's actual
      // pose data is browser-mediapipe but they were *expecting* server
      // extraction — distinguish so the diagnostic chip can warn (red)
      // rather than just informing (yellow).
      if (USE_RAILWAY_EXTRACT) {
        result = { ...result, extractorBackend: 'mediapipe-browser-fallback' }
      }
    }

    setOverallProgress(95)
    setStatusMsg('Saving analysis...')

    const keypointsJson = {
      fps_sampled: result.fps,
      frame_count: result.frames.length,
      frames: result.frames,
    }

    // When Railway did the work, it already wrote the keypoints to the
    // pending session row. We only POST /api/sessions for the browser
    // path (or on Railway fallback, to upsert the browser output into
    // the pending row).
    if (!usedRailway) {
      try {
        const sessionId =
          pendingSessionId ?? usePoseStore.getState().sessionId
        const sessRes = await fetch('/api/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId, blobUrl, shotType, keypointsJson }),
        })
        if (!sessRes.ok) {
          console.error('Failed to save session:', sessRes.status, await sessRes.text())
          setStatusMsg('Warning: analysis complete but failed to save session.')
        }
      } catch (err) {
        console.error('Failed to save session:', err)
        setStatusMsg('Warning: analysis complete but failed to save session.')
      }
    }

    // Hand the object URL off to the store for playback (store revokes old on replace).
    // On the Railway path we don't have a local object URL; create one from
    // the original File so playback still works without re-downloading
    // from Vercel Blob.
    const playbackUrl = usedRailway ? URL.createObjectURL(file) : result.objectUrl
    setLocalVideoUrl(playbackUrl)
    setFramesData(result.frames)
    setExtractorBackend(result.extractorBackend)
    setFallbackReason(
      result.extractorBackend === 'mediapipe-browser-fallback' ? railwayFailReason : null,
    )
    persistShotType(shotType)
    setOverallProgress(100)
    setStatusMsg(
      `Done! Analyzed ${result.frames.length} frames${usedRailway ? ' (server)' : ''}.`,
    )
    setProcessing(false)
    setBusy(false)
    onComplete?.(blobUrl, result.frames)
  }

  const handleFile = (file: File) => {
    if (!file.type.startsWith('video/')) {
      setStatusMsg('Please upload a video file.')
      return
    }
    processVideo(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const handleRetry = () => {
    if (!lastFile) return
    // Reuse the cached blob URL so we don't re-upload a 100MB clip
    // every time the user retries an extraction failure.
    const cached = usePoseStore.getState().blobUrl ?? undefined
    processVideo(lastFile, { skipUpload: !!cached, cachedBlobUrl: cached ?? undefined })
  }

  const handleCancel = () => {
    abort()
    setStatusMsg('Cancelled.')
    setShowRetry(true)
    setProcessing(false)
    setBusy(false)
    setModelLoading(false)
  }

  const processing = busy || extracting

  return (
    <div className="space-y-4">
      {/* Shot type selector */}
      <div className="flex gap-2 flex-wrap">
        {SHOT_TYPES.map((type) => (
          <button
            key={type}
            onClick={() => setShotType(type)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium capitalize transition-all ${
              shotType === type
                ? 'bg-emerald-500 text-white'
                : 'bg-white/10 text-white/60 hover:bg-white/20 hover:text-white'
            }`}
          >
            {type}
          </button>
        ))}
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !processing && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
          dragging
            ? 'border-emerald-400 bg-emerald-500/10'
            : 'border-white/20 hover:border-white/40 hover:bg-white/5'
        } ${processing ? 'cursor-not-allowed opacity-80' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) handleFile(file)
          }}
        />

        {processing ? (
          <div className="space-y-4">
            <div className="text-4xl">🎾</div>
            <p className="text-white font-medium">{statusMsg}</p>
            {modelLoading ? (
              // Indeterminate animated bar during the MediaPipe model
              // download. Honest signal: "actively loading" rather than
              // "stuck at 25%". The pulse stripe slides infinitely.
              <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden relative">
                <div
                  className="absolute inset-y-0 w-1/3 bg-emerald-400 rounded-full animate-pulse"
                  style={{
                    animation: 'upload-zone-pulse 1.4s ease-in-out infinite',
                  }}
                />
                <style>{`
                  @keyframes upload-zone-pulse {
                    0%, 100% { left: 0%; opacity: 0.5; }
                    50% { left: 66.6%; opacity: 1; }
                  }
                `}</style>
              </div>
            ) : (
              <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-emerald-400 rounded-full transition-all duration-300"
                  style={{ width: `${overallProgress}%` }}
                />
              </div>
            )}
            <p className="text-white/50 text-sm">
              {modelLoading ? 'Loading model (one-time, ~10s)' : `${overallProgress}%`}
            </p>
            <button
              onClick={(e) => {
                e.stopPropagation()
                handleCancel()
              }}
              className="text-xs text-white/60 hover:text-white underline transition-colors"
            >
              Cancel
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-5xl">🎬</div>
            <p className="text-white text-lg font-medium">
              Drop your swing video here
            </p>
            <p className="text-white/50 text-sm">
              MP4, MOV, WebM · Max 200MB · Select shot type above first
            </p>
            {statusMsg && (
              <p className={`${/fail|error|please|cancel/i.test(statusMsg) ? 'text-red-400' : 'text-emerald-400'} text-sm font-medium`}>{statusMsg}</p>
            )}
            {showRetry && lastFile && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  handleRetry()
                }}
                className="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-white text-sm font-medium transition-colors"
              >
                Retry analysis
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
