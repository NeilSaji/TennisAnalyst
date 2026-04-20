'use client'

import { useRef, useState, useEffect } from 'react'
import { upload } from '@vercel/blob/client'
import { usePoseStore } from '@/store'
import { usePoseExtractor } from '@/hooks/usePoseExtractor'
import type { PoseFrame } from '@/lib/supabase'

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

  const { setFramesData, setBlobUrl, setLocalVideoUrl, setProcessing, isProcessing, setShotType: persistShotType } =
    usePoseStore()

  const { extract, progress: extractProgress, error: extractError, isProcessing: extracting } = usePoseExtractor()

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

  // Surface extractor errors
  useEffect(() => {
    if (extractError) setStatusMsg(extractError)
  }, [extractError])

  const processVideo = async (file: File) => {
    setBusy(true)
    setProcessing(true)
    setOverallProgress(0)
    setStatusMsg('Uploading video...')

    // 1. Upload directly to Vercel Blob via a signed client token. Going
    //    through the API route would hit Vercel's serverless body limit
    //    (4.5MB Hobby / ~100MB Pro) and fail for any real swing clip.
    let blobUrl: string
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
      setProcessing(false)
      setBusy(false)
      return
    }

    setOverallProgress(15)
    setStatusMsg('Loading pose model...')

    setOverallProgress(25)
    setStatusMsg('Analyzing pose from video...')

    const result = await extract(file)
    if (!result) {
      // Hook handled status/error; just release the processing lock
      setProcessing(false)
      setBusy(false)
      return
    }

    setOverallProgress(95)
    setStatusMsg('Saving analysis...')

    const keypointsJson = {
      fps_sampled: result.fps,
      frame_count: result.frames.length,
      frames: result.frames,
    }

    try {
      const sessionId = usePoseStore.getState().sessionId
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

    // Hand the object URL off to the store for playback (store revokes old on replace)
    setLocalVideoUrl(result.objectUrl)
    setFramesData(result.frames)
    persistShotType(shotType)
    setOverallProgress(100)
    setStatusMsg(`Done! Analyzed ${result.frames.length} frames.`)
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
            <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-emerald-400 rounded-full transition-all duration-300"
                style={{ width: `${overallProgress}%` }}
              />
            </div>
            <p className="text-white/50 text-sm">{overallProgress}%</p>
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
              <p className={`${/fail|error|please/i.test(statusMsg) ? 'text-red-400' : 'text-emerald-400'} text-sm font-medium`}>{statusMsg}</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
