'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { upload } from '@vercel/blob/client'
import { useLiveCapture, type LiveSessionResult } from '@/hooks/useLiveCapture'
import { useLiveCoach } from '@/hooks/useLiveCoach'
import { useLiveStore } from '@/store/live'
import { usePoseStore } from '@/store'
import LiveSwingCounter from './LiveSwingCounter'

const SHOT_TYPES = ['forehand', 'backhand', 'serve', 'volley'] as const

interface LiveCapturePanelProps {
  onSessionComplete?: (result: LiveSessionResult) => void
}

function extForMime(mimeType: string): string {
  if (mimeType.startsWith('video/mp4')) return 'mp4'
  if (mimeType.startsWith('video/webm')) return 'webm'
  if (mimeType.startsWith('video/quicktime')) return 'mov'
  return 'bin'
}

export default function LiveCapturePanel({ onSessionComplete }: LiveCapturePanelProps) {
  const router = useRouter()
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [summary, setSummary] = useState<{ swings: number; durationMs: number } | null>(null)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)

  const shotType = useLiveStore((s) => s.shotType)
  const setShotType = useLiveStore((s) => s.setShotType)
  const swingCount = useLiveStore((s) => s.swingCount)
  const setSwingCount = useLiveStore((s) => s.setSwingCount)
  const status = useLiveStore((s) => s.status)
  const setStatus = useLiveStore((s) => s.setStatus)
  const setErrorMessage = useLiveStore((s) => s.setErrorMessage)
  const errorMessage = useLiveStore((s) => s.errorMessage)
  const setSessionStartedAtMs = useLiveStore((s) => s.setSessionStartedAtMs)
  const resetSession = useLiveStore((s) => s.resetSession)
  const transcript = useLiveStore((s) => s.transcript)

  const poseSetFramesData = usePoseStore((s) => s.setFramesData)
  const poseSetBlobUrl = usePoseStore((s) => s.setBlobUrl)
  const poseSetLocalVideoUrl = usePoseStore((s) => s.setLocalVideoUrl)
  const poseSetSessionId = usePoseStore((s) => s.setSessionId)
  const poseSetShotType = usePoseStore((s) => s.setShotType)

  const swingCountRef = useRef(0)
  const coach = useLiveCoach()

  // Keep the coach's shot type in sync with the user's selection.
  useEffect(() => {
    coach.setShotType(shotType)
  }, [coach, shotType])

  const { start, stop, abort, status: captureStatus, error: captureError, isRecording, pickedMimeType } = useLiveCapture({
    onSwing: (swing) => {
      swingCountRef.current++
      setSwingCount(swingCountRef.current)
      coach.pushSwing(swing)
    },
    onStatus: (s) => {
      if (s === 'recording') setStatus('recording')
      if (s === 'error') setStatus('error')
      if (s === 'stopping') setStatus('uploading')
      if (s === 'idle') setStatus('idle')
    },
  })

  // Surface capture errors into the store
  useEffect(() => {
    if (captureError) setErrorMessage(captureError)
  }, [captureError, setErrorMessage])

  const handleStart = useCallback(async () => {
    const videoEl = videoRef.current
    if (!videoEl) return
    // Must prime TTS from the user-gesture handler so iOS Safari unlocks
    // speechSynthesis before the first coaching utterance.
    coach.primeTts()
    setErrorMessage(null)
    setSummary(null)
    swingCountRef.current = 0
    setSwingCount(0)
    setStatus('preflight')
    const startedAt = Date.now()
    setSessionStartedAtMs(startedAt)
    coach.markSessionStart(startedAt)
    await start(videoEl)
  }, [coach, setErrorMessage, setSessionStartedAtMs, setStatus, setSwingCount, start])

  const handleStop = useCallback(async () => {
    setUploadStatus('Finalizing recording…')
    const result = await stop()
    if (!result) {
      setUploadStatus(null)
      setStatus('idle')
      return
    }

    setSummary({ swings: result.swings.length, durationMs: result.durationMs })
    setStatus('uploading')
    onSessionComplete?.(result)

    // Give pending in-flight coach calls a moment to settle so their
    // analysis_events rows are written before we backfill session_id.
    await new Promise((r) => setTimeout(r, 2_000))

    let blobUrl: string
    try {
      setUploadStatus('Uploading video…')
      const ext = extForMime(result.blobMimeType)
      const blobPath = `live/${Date.now()}-session.${ext}`
      const uploaded = await upload(blobPath, result.blob, {
        access: 'public',
        handleUploadUrl: '/api/upload',
        contentType: result.blobMimeType,
      })
      blobUrl = uploaded.url
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Upload failed')
      setUploadStatus(null)
      setStatus('error')
      return
    }

    const batchEventIds = transcript
      .map((e) => e.eventId)
      .filter((id): id is string => typeof id === 'string' && id.length > 0)

    try {
      setUploadStatus('Saving session…')
      const res = await fetch('/api/sessions/live', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          blobUrl,
          shotType,
          keypointsJson: result.keypoints,
          swings: result.swings.map((s) => ({
            startFrame: s.startFrameIndex,
            endFrame: s.endFrameIndex,
            startMs: s.startMs,
            endMs: s.endMs,
          })),
          batchEventIds,
        }),
      })
      if (!res.ok) {
        const msg = await res.text().catch(() => 'Save failed')
        setErrorMessage(`Save failed: ${msg}`)
        setUploadStatus(null)
        setStatus('error')
        return
      }
      const { sessionId } = (await res.json()) as { sessionId: string }

      // Hydrate usePoseStore so /analyze picks this session up identically to
      // an uploaded clip.
      const objectUrl = URL.createObjectURL(result.blob)
      poseSetFramesData(result.keypoints.frames)
      poseSetBlobUrl(blobUrl)
      poseSetLocalVideoUrl(objectUrl)
      poseSetSessionId(sessionId)
      poseSetShotType(shotType)

      setUploadStatus(null)
      setStatus('complete')
      router.replace('/analyze')
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Save failed')
      setUploadStatus(null)
      setStatus('error')
    }
  }, [
    onSessionComplete,
    poseSetBlobUrl,
    poseSetFramesData,
    poseSetLocalVideoUrl,
    poseSetSessionId,
    poseSetShotType,
    router,
    setErrorMessage,
    setStatus,
    shotType,
    stop,
    transcript,
  ])

  const handleReset = useCallback(() => {
    abort()
    coach.reset()
    resetSession()
    swingCountRef.current = 0
    setSummary(null)
  }, [abort, coach, resetSession])

  const busy = captureStatus === 'requesting-permissions' || captureStatus === 'initializing' || captureStatus === 'stopping'
  const displaySeconds = summary ? Math.max(1, Math.round(summary.durationMs / 1000)) : 0

  return (
    <div className="space-y-4">
      {/* Shot type picker — locked during a session */}
      <div className="flex gap-2 flex-wrap">
        {SHOT_TYPES.map((type) => (
          <button
            key={type}
            onClick={() => setShotType(type)}
            disabled={isRecording || busy}
            className={`px-4 py-1.5 rounded-full text-sm font-medium capitalize transition-all ${
              shotType === type
                ? 'bg-emerald-500 text-white'
                : 'bg-white/10 text-white/60 hover:bg-white/20 hover:text-white'
            } ${isRecording || busy ? 'opacity-60 cursor-not-allowed' : ''}`}
          >
            {type}
          </button>
        ))}
      </div>

      {/* Preview / idle card */}
      <div className="relative rounded-2xl border border-white/10 bg-black overflow-hidden aspect-video">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
          playsInline
          muted
        />
        {!isRecording && !busy ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-gradient-to-br from-emerald-500/10 to-transparent">
            <div className="text-5xl">🎾</div>
            <p className="text-white font-medium">Lean your phone on anything side-on. Tap Start and play.</p>
            <p className="text-white/50 text-sm text-center max-w-sm px-4">
              Your bag, a bench, the fence — whatever you've got. We coach you every few swings and save the session for later review.
            </p>
          </div>
        ) : null}

        {isRecording ? (
          <div className="absolute top-3 left-3">
            <LiveSwingCounter swingCount={swingCount} />
          </div>
        ) : null}
      </div>

      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-3">
        {!isRecording ? (
          <button
            onClick={handleStart}
            disabled={busy}
            className="flex-1 bg-emerald-500 hover:bg-emerald-400 disabled:bg-white/10 disabled:text-white/40 text-white font-bold rounded-2xl px-6 py-4 text-lg transition-all"
          >
            {busy ? 'Starting…' : 'Start'}
          </button>
        ) : (
          <button
            onClick={handleStop}
            className="flex-1 bg-red-500 hover:bg-red-400 text-white font-bold rounded-2xl px-6 py-4 text-lg transition-all"
          >
            Stop
          </button>
        )}
        {summary ? (
          <button
            onClick={handleReset}
            className="bg-white/10 hover:bg-white/20 text-white font-medium rounded-2xl px-6 py-4 transition-all"
          >
            New session
          </button>
        ) : null}
      </div>

      {errorMessage ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 text-red-200 text-sm px-4 py-3">
          {errorMessage}
        </div>
      ) : null}

      {pickedMimeType && isRecording ? (
        <p className="text-white/40 text-xs">Recording {pickedMimeType}</p>
      ) : null}

      {uploadStatus ? (
        <div className="rounded-xl border border-white/10 bg-white/[0.04] text-white text-sm px-4 py-3">
          {uploadStatus}
        </div>
      ) : null}

      {summary && !uploadStatus ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] px-5 py-4 space-y-1">
          <p className="text-white font-medium">
            {summary.swings} {summary.swings === 1 ? 'swing' : 'swings'} detected in {displaySeconds}s
          </p>
          <p className="text-white/50 text-sm">
            Opening the session in Analyze…
          </p>
        </div>
      ) : null}

      <p className="text-white/40 text-xs leading-relaxed">
        Status: <span className="text-white/60">{status}</span>
        {' · '}
        Any side-on setup works — prop the phone on your bag, a bench, the fence. Just keep your full body in frame. Front and back-of-court angles are too noisy to coach from.
      </p>
    </div>
  )
}
