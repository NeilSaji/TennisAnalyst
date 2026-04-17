'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { upload as uploadToBlob } from '@vercel/blob/client'
import {
  adminAuthHeaders,
  clearAdminToken,
  getAdminToken,
  setAdminToken,
  verifyAdminToken,
} from '@/lib/adminAuthClient'
import { trimVideoInBrowser } from '@/lib/ffmpegClient'

/* ------------------------------------------------------------------ */
/*  YouTube IFrame API types                                          */
/* ------------------------------------------------------------------ */
declare global {
  interface Window {
    YT: {
      Player: new (
        elementId: string,
        config: {
          videoId: string
          events?: {
            onReady?: () => void
          }
        },
      ) => YTPlayer
    }
    onYouTubeIframeAPIReady: (() => void) | undefined
  }
}

interface YTPlayer {
  getCurrentTime: () => number
  destroy: () => void
}

/* ------------------------------------------------------------------ */
/*  Types                                                             */
/* ------------------------------------------------------------------ */
type ShotType = 'forehand' | 'backhand' | 'serve' | 'volley'
type CameraAngle = 'side' | 'behind' | 'front' | 'court_level'

interface SavedClip {
  proName: string
  shotType: ShotType
  cameraAngle: CameraAngle
  duration: number
  speedFactor: number
  status: 'saved' | 'error'
}

interface ProEntry {
  name: string
  nationality?: string
}

type SpeedOption = {
  label: string
  factor: number
}

const SPEED_OPTIONS: SpeedOption[] = [
  { label: '1/4×', factor: 1 / 4 },
  { label: '1/3×', factor: 1 / 3 },
  { label: '1/2×', factor: 1 / 2 },
  { label: 'Normal', factor: 1 },
  { label: '2×', factor: 2 },
  { label: '3×', factor: 3 },
  { label: '4×', factor: 4 },
]

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */
function extractVideoId(url: string): string | null {
  // youtube.com/watch?v=ID
  const longMatch = url.match(/[?&]v=([a-zA-Z0-9_-]{11})/)
  if (longMatch) return longMatch[1]
  // youtu.be/ID
  const shortMatch = url.match(/youtu\.be\/([a-zA-Z0-9_-]{11})/)
  if (shortMatch) return shortMatch[1]
  return null
}

function formatTime(seconds: number | null): string {
  if (seconds === null) return '--:--.-'
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs < 10 ? '0' : ''}${secs.toFixed(1)}`
}

// Must match the server-side convention in /api/admin/tag-clip so downloaded
// files can later be matched to DB rows by filename.
function sanitizeFilename(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')
}

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */
export default function TagClipsPage() {
  const [authState, setAuthState] = useState<'checking' | 'locked' | 'unlocked'>(
    'checking',
  )
  const [pwInput, setPwInput] = useState('')
  const [pwError, setPwError] = useState<string | null>(null)
  const [pwSubmitting, setPwSubmitting] = useState(false)

  // Try the cached token on mount. If it works, unlock; otherwise show prompt.
  // Wrapped in a void IIFE so every setAuthState resolves asynchronously,
  // sidestepping react-hooks/immutability's "cascading renders" warning.
  useEffect(() => {
    void (async () => {
      const cached = getAdminToken()
      if (!cached) {
        setAuthState('locked')
        return
      }
      const ok = await verifyAdminToken(cached)
      if (ok) {
        setAuthState('unlocked')
      } else {
        clearAdminToken()
        setAuthState('locked')
      }
    })()
  }, [])

  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!pwInput || pwSubmitting) return
    setPwSubmitting(true)
    setPwError(null)
    const ok = await verifyAdminToken(pwInput)
    setPwSubmitting(false)
    if (ok) {
      setAdminToken(pwInput)
      setPwInput('')
      setAuthState('unlocked')
    } else {
      setPwError('Wrong password (or admin is disabled on this deploy).')
    }
  }

  if (authState === 'checking') {
    return (
      <div className="max-w-md mx-auto px-4 py-16 text-center text-white/50 text-sm">
        Checking access…
      </div>
    )
  }

  if (authState === 'locked') {
    return (
      <div className="max-w-md mx-auto px-4 py-16">
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6">
          <h1 className="text-xl font-bold text-white mb-2">Clip Studio</h1>
          <p className="text-white/50 text-sm mb-6">
            Admin access required. Enter the password to continue.
          </p>
          <form onSubmit={handlePasswordSubmit} className="space-y-3">
            <input
              type="password"
              value={pwInput}
              onChange={(e) => setPwInput(e.target.value)}
              autoFocus
              placeholder="Admin password"
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white placeholder:text-white/30 focus:outline-none focus:border-emerald-500/50"
            />
            {pwError && <p className="text-red-400 text-sm">{pwError}</p>}
            <button
              type="submit"
              disabled={!pwInput || pwSubmitting}
              className="w-full px-6 py-2.5 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition-colors"
            >
              {pwSubmitting ? 'Checking…' : 'Unlock'}
            </button>
          </form>
        </div>
      </div>
    )
  }

  return <TagClipsStudio onLockOut={() => setAuthState('locked')} />
}

function TagClipsStudio({ onLockOut }: { onLockOut: () => void }) {
  /* --- Source selection --- */
  const [sourceType, setSourceType] = useState<'youtube' | 'upload'>('upload')

  /* --- YouTube state --- */
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [videoId, setVideoId] = useState<string | null>(null)
  const [urlError, setUrlError] = useState<string | null>(null)
  const [ytReady, setYtReady] = useState(false)
  const playerRef = useRef<YTPlayer | null>(null)
  const apiLoadedRef = useRef(false)

  /* --- Local-upload state --- */
  const [localFile, setLocalFile] = useState<File | null>(null)
  const [localFileUrl, setLocalFileUrl] = useState<string | null>(null)
  const [localFileError, setLocalFileError] = useState<string | null>(null)
  const [localDragging, setLocalDragging] = useState(false)
  const [ffmpegProgress, setFfmpegProgress] = useState<number | null>(null)
  const localVideoRef = useRef<HTMLVideoElement | null>(null)
  const localFileInputRef = useRef<HTMLInputElement | null>(null)

  // Revoke the object URL when the local file changes or component unmounts
  // so we don't leak blob: URLs into the document's memory.
  useEffect(() => {
    if (!localFileUrl) return
    return () => {
      URL.revokeObjectURL(localFileUrl)
    }
  }, [localFileUrl])

  /* --- Timestamps --- */
  const [startTime, setStartTime] = useState<number | null>(null)
  const [endTime, setEndTime] = useState<number | null>(null)

  /* --- Metadata --- */
  const [proName, setProName] = useState('')
  const [nationality, setNationality] = useState('')
  const [shotType, setShotType] = useState<ShotType | null>(null)
  const [cameraAngle, setCameraAngle] = useState<CameraAngle | null>(null)

  /* --- Pros list --- */
  const [pros, setPros] = useState<ProEntry[]>([])

  /* --- Save state --- */
  const [saving, setSaving] = useState(false)
  const [saveResult, setSaveResult] = useState<{ ok: boolean; msg: string } | null>(null)
  const [savedClips, setSavedClips] = useState<SavedClip[]>([])

  /* --- Preview state --- */
  const [speedFactor, setSpeedFactor] = useState<number>(1)
  const [previewing, setPreviewing] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  // Set only for the upload flow: the trimmed blob, reused on Save so we
  // don't have to re-encode a second time.
  const [previewBlob, setPreviewBlob] = useState<Blob | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)
  // The speedFactor that the current previewUrl was generated with.
  // Save sends this value, so the user cannot save at a speed they haven't verified.
  const [previewedSpeedFactor, setPreviewedSpeedFactor] = useState<number | null>(null)

  /* --- Fuzzy-match guard (server rejected the save — user needs to confirm) --- */
  const [fuzzySuggestions, setFuzzySuggestions] = useState<string[] | null>(null)

  /* ---------------------------------------------------------------- */
  /*  Load YouTube IFrame API                                         */
  /* ---------------------------------------------------------------- */
  useEffect(() => {
    if (apiLoadedRef.current) return
    if (window.YT) {
      apiLoadedRef.current = true
      return
    }
    const tag = document.createElement('script')
    tag.src = 'https://www.youtube.com/iframe_api'
    document.head.appendChild(tag)
    apiLoadedRef.current = true
  }, [])

  /* ---------------------------------------------------------------- */
  /*  Create / recreate player when videoId changes                   */
  /* ---------------------------------------------------------------- */
  useEffect(() => {
    if (!videoId) return

    let cancelled = false

    const initPlayer = () => {
      if (cancelled) return
      // Destroy previous player if any
      if (playerRef.current) {
        playerRef.current.destroy()
        playerRef.current = null
      }
      setYtReady(false)
      playerRef.current = new window.YT.Player('yt-player', {
        videoId,
        events: {
          onReady: () => {
            if (!cancelled) setYtReady(true)
          },
        },
      })
    }

    if (window.YT && window.YT.Player) {
      initPlayer()
    } else {
      window.onYouTubeIframeAPIReady = initPlayer
    }

    return () => {
      cancelled = true
    }
  }, [videoId])

  /* ---------------------------------------------------------------- */
  /*  Fetch pros                                                      */
  /* ---------------------------------------------------------------- */
  useEffect(() => {
    fetch('/api/pros')
      .then((r) => (r.ok ? r.json() : []))
      .then((data: ProEntry[]) => setPros(Array.isArray(data) ? data : []))
      .catch(() => {
        /* non-critical — datalist simply stays empty */
      })
  }, [])

  /* ---------------------------------------------------------------- */
  /*  Handlers                                                        */
  /* ---------------------------------------------------------------- */
  const clearPreview = useCallback(() => {
    setPreviewUrl((prev) => {
      if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev)
      return null
    })
    setPreviewBlob(null)
    setPreviewedSpeedFactor(null)
    setPreviewError(null)
  }, [])

  const handleLoad = useCallback(() => {
    setUrlError(null)
    const id = extractVideoId(youtubeUrl)
    if (!id) {
      setUrlError('Invalid YouTube URL. Use youtube.com/watch?v=... or youtu.be/...')
      return
    }
    setVideoId(id)
    setStartTime(null)
    setEndTime(null)
    setSaveResult(null)
    clearPreview()
  }, [youtubeUrl, clearPreview])

  const getActiveTime = useCallback((): number | null => {
    if (sourceType === 'youtube') {
      return playerRef.current ? playerRef.current.getCurrentTime() : null
    }
    const v = localVideoRef.current
    return v ? v.currentTime : null
  }, [sourceType])

  const handleMarkStart = useCallback(() => {
    const t = getActiveTime()
    if (t === null) return
    setStartTime(t)
    setSaveResult(null)
    clearPreview()
  }, [clearPreview, getActiveTime])

  const handleMarkEnd = useCallback(() => {
    const t = getActiveTime()
    if (t === null) return
    setEndTime(t)
    setSaveResult(null)
    clearPreview()
  }, [clearPreview, getActiveTime])

  const handleReset = useCallback(() => {
    setStartTime(null)
    setEndTime(null)
    setProName('')
    setNationality('')
    setShotType(null)
    setCameraAngle(null)
    setSaveResult(null)
    setSpeedFactor(1)
    clearPreview()
  }, [clearPreview])

  const acceptLocalFile = useCallback((file: File) => {
    setLocalFileError(null)
    if (!file.type.startsWith('video/')) {
      setLocalFileError('File must be a video')
      return
    }
    // Cap at 500MB so ffmpeg.wasm doesn't blow the wasm heap in the browser.
    if (file.size > 500 * 1024 * 1024) {
      setLocalFileError('File too large (max 500MB)')
      return
    }
    // Revoke previous local URL, reset clip timings & preview
    if (localFileUrl) URL.revokeObjectURL(localFileUrl)
    const url = URL.createObjectURL(file)
    setLocalFile(file)
    setLocalFileUrl(url)
    setStartTime(null)
    setEndTime(null)
    clearPreview()
  }, [localFileUrl, clearPreview])

  const hasSource = sourceType === 'youtube' ? !!videoId : !!localFileUrl

  const handlePreview = useCallback(async () => {
    if (startTime === null || endTime === null || endTime <= startTime) return
    setPreviewing(true)
    setPreviewError(null)
    setPreviewUrl(null)
    setFfmpegProgress(null)

    try {
      if (sourceType === 'upload') {
        if (!localFile) throw new Error('No video selected')
        // Cap source duration to match the YouTube path (prevents OOM on the
        // WASM heap from a 20-minute source).
        if (endTime - startTime > 60) {
          throw new Error('Clip must be under 60 seconds')
        }
        const blob = await trimVideoInBrowser({
          file: localFile,
          startSec: startTime,
          endSec: endTime,
          speedFactor,
          onProgress: (r) => setFfmpegProgress(r),
        })
        const url = URL.createObjectURL(blob)
        // Revoke any previous preview URL before overwriting to avoid leaks.
        setPreviewUrl((prev) => {
          if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev)
          return url
        })
        setPreviewBlob(blob)
        setPreviewedSpeedFactor(speedFactor)
        return
      }

      // YouTube path (server-side yt-dlp + ffmpeg, local-only in practice).
      // Server returns a previewId we then pull from an auth-gated endpoint
      // and wrap in a blob: URL for <video>, so the preview file never sits
      // under public/ where anyone who guessed the UUID could grab it.
      const res = await fetch('/api/admin/preview-clip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...adminAuthHeaders() },
        body: JSON.stringify({ youtubeUrl, startTime, endTime, speedFactor }),
      })
      if (res.status === 401) {
        clearAdminToken()
        onLockOut()
        return
      }
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `Server error ${res.status}`)
      }
      const data = (await res.json()) as { videoUrl: string }
      const fileRes = await fetch(data.videoUrl, { headers: adminAuthHeaders() })
      if (fileRes.status === 401) {
        clearAdminToken()
        onLockOut()
        return
      }
      if (!fileRes.ok) throw new Error(`Failed to fetch preview (${fileRes.status})`)
      const previewFileBlob = await fileRes.blob()
      const url = URL.createObjectURL(previewFileBlob)
      setPreviewUrl((prev) => {
        if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev)
        return url
      })
      setPreviewedSpeedFactor(speedFactor)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      setPreviewError(message)
    } finally {
      setPreviewing(false)
      setFfmpegProgress(null)
    }
  }, [sourceType, localFile, youtubeUrl, startTime, endTime, speedFactor, onLockOut])

  const handleSpeedChange = useCallback(
    (f: number) => {
      setSpeedFactor(f)
      if (previewedSpeedFactor !== null && previewedSpeedFactor !== f) {
        // Previewed clip no longer matches the selected speed — require a re-preview.
        setPreviewUrl(null)
      }
    },
    [previewedSpeedFactor],
  )

  const proNameIsKnown = pros.some(
    (p) => p.name.toLowerCase() === proName.trim().toLowerCase(),
  )

  const canDownload =
    previewUrl !== null &&
    previewedSpeedFactor === speedFactor &&
    proName.trim() !== '' &&
    shotType !== null &&
    cameraAngle !== null

  const handleDownload = useCallback(() => {
    if (!canDownload || !previewUrl || !shotType || !cameraAngle) return
    const filename = `${sanitizeFilename(proName)}_${shotType}_${cameraAngle}_${Date.now()}.mp4`
    const a = document.createElement('a')
    a.href = previewUrl
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }, [canDownload, previewUrl, proName, shotType, cameraAngle])

  const canSave =
    startTime !== null &&
    endTime !== null &&
    endTime > startTime &&
    proName.trim() !== '' &&
    shotType !== null &&
    cameraAngle !== null &&
    !saving &&
    (sourceType === 'youtube' ? !!videoId : !!previewBlob && previewedSpeedFactor === speedFactor)

  const handleSave = useCallback(async (opts?: { confirmNewPro?: boolean }) => {
    if (!canSave) return
    setSaving(true)
    setSaveResult(null)
    setFuzzySuggestions(null)

    if (sourceType === 'upload') {
      // Upload flow: push the preview Blob to Vercel Blob via the signed
      // client-token endpoint, then POST metadata to /api/clips/save. We
      // skip the server-side yt-dlp/ffmpeg entirely.
      try {
        if (!previewBlob) throw new Error('Preview is missing')
        const sanitized = sanitizeFilename(proName)
        const blobPath = `pro-videos/${sanitized}_${shotType}_${cameraAngle}_${Date.now()}.mp4`
        // Admin token rides through the documented clientPayload channel so
        // /api/clips/upload can authenticate the token-generation request.
        const uploadedBlob = await uploadToBlob(blobPath, previewBlob, {
          access: 'public',
          handleUploadUrl: '/api/clips/upload',
          contentType: 'video/mp4',
          clientPayload: JSON.stringify({ adminToken: getAdminToken() ?? '' }),
        })
        const res = await fetch('/api/clips/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...adminAuthHeaders() },
          body: JSON.stringify({
            blobUrl: uploadedBlob.url,
            proName: proName.trim(),
            nationality: proNameIsKnown ? undefined : nationality.trim() || undefined,
            shotType,
            cameraAngle,
            speedFactor,
            durationSec: (endTime! - startTime!) / speedFactor,
            sourceLabel: localFile?.name,
            confirmNewPro: opts?.confirmNewPro ?? false,
          }),
        })
        if (res.status === 401) {
          clearAdminToken()
          onLockOut()
          setSaving(false)
          return
        }
        if (res.status === 409) {
          const data = (await res.json().catch(() => null)) as
            | { error?: string; suggestions?: string[]; code?: string }
            | null
          if (data?.code === 'fuzzy_match' && Array.isArray(data.suggestions)) {
            setFuzzySuggestions(data.suggestions)
            setSaveResult({ ok: false, msg: data.error ?? 'Possible typo — confirm name.' })
            setSaving(false)
            return
          }
        }
        if (!res.ok) {
          const text = await res.text()
          throw new Error(text || `Server error ${res.status}`)
        }
        setSaveResult({ ok: true, msg: 'Clip saved successfully!' })
        setSavedClips((prev) => [
          ...prev,
          {
            proName: proName.trim(),
            shotType: shotType!,
            cameraAngle: cameraAngle!,
            duration: (endTime! - startTime!) / speedFactor,
            speedFactor,
            status: 'saved',
          },
        ])
        handleReset()
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Unknown error'
        setSaveResult({ ok: false, msg: message })
        setSavedClips((prev) => [
          ...prev,
          {
            proName: proName.trim(),
            shotType: shotType ?? 'forehand',
            cameraAngle: cameraAngle ?? 'side',
            duration:
              endTime !== null && startTime !== null
                ? (endTime - startTime) / speedFactor
                : 0,
            speedFactor,
            status: 'error',
          },
        ])
      } finally {
        setSaving(false)
      }
      return
    }

    // YouTube flow: server does everything.
    const body = {
      youtubeUrl,
      startTime,
      endTime,
      proName: proName.trim(),
      nationality: proNameIsKnown ? undefined : nationality.trim() || undefined,
      shotType,
      cameraAngle,
      speedFactor,
      confirmNewPro: opts?.confirmNewPro ?? false,
    }

    try {
      const res = await fetch('/api/admin/tag-clip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...adminAuthHeaders() },
        body: JSON.stringify(body),
      })
      if (res.status === 401) {
        clearAdminToken()
        onLockOut()
        setSaving(false)
        return
      }
      if (res.status === 409) {
        const data = (await res.json().catch(() => null)) as
          | { error?: string; suggestions?: string[]; code?: string }
          | null
        if (data?.code === 'fuzzy_match' && Array.isArray(data.suggestions)) {
          setFuzzySuggestions(data.suggestions)
          setSaveResult({ ok: false, msg: data.error ?? 'Possible typo — confirm name.' })
          setSaving(false)
          return
        }
      }
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `Server error ${res.status}`)
      }
      setSaveResult({ ok: true, msg: 'Clip saved successfully!' })
      setSavedClips((prev) => [
        ...prev,
        {
          proName: proName.trim(),
          shotType: shotType!,
          cameraAngle: cameraAngle!,
          duration: (endTime! - startTime!) / speedFactor,
          speedFactor,
          status: 'saved',
        },
      ])
      handleReset()
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      setSaveResult({ ok: false, msg: message })
      setSavedClips((prev) => [
        ...prev,
        {
          proName: proName.trim(),
          shotType: shotType ?? 'forehand',
          cameraAngle: cameraAngle ?? 'side',
          duration:
            endTime !== null && startTime !== null
              ? (endTime - startTime) / speedFactor
              : 0,
          speedFactor,
          status: 'error',
        },
      ])
    } finally {
      setSaving(false)
    }
  }, [
    canSave,
    sourceType,
    previewBlob,
    localFile,
    youtubeUrl,
    startTime,
    endTime,
    proName,
    proNameIsKnown,
    nationality,
    shotType,
    cameraAngle,
    speedFactor,
    handleReset,
    onLockOut,
  ])

  /* ---------------------------------------------------------------- */
  /*  Derived                                                         */
  /* ---------------------------------------------------------------- */
  const duration =
    startTime !== null && endTime !== null && endTime > startTime
      ? endTime - startTime
      : null

  /* ---------------------------------------------------------------- */
  /*  Render                                                          */
  /* ---------------------------------------------------------------- */
  const SHOT_TYPES: ShotType[] = ['forehand', 'backhand', 'serve', 'volley']
  const CAMERA_ANGLES: CameraAngle[] = ['side', 'behind', 'front', 'court_level']

  // YouTube source ships with yt-dlp + ffmpeg binaries via
  // youtube-dl-exec and @ffmpeg-installer/ffmpeg, so it works on Vercel too.
  // Keeping the variable as `true` to avoid touching the conditional below.
  const youtubeSourceEnabled = true

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-black text-white mb-2">Clip Studio</h1>
        <p className="text-white/50">
          Trim tennis clips and add them to the pro database. Upload a local video, or paste a YouTube URL.
        </p>
      </div>

      {/* Source toggle. The YouTube tab is only shown when the server has
          yt-dlp + ffmpeg installed (opted in via NEXT_PUBLIC_YT_SOURCE_ENABLED);
          on Vercel it stays hidden so users can't hit the dead path. */}
      {youtubeSourceEnabled && (
        <div className="flex gap-2 p-1 bg-white/5 rounded-xl w-fit mb-6">
          <button
            onClick={() => {
              setSourceType('upload')
              clearPreview()
            }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              sourceType === 'upload' ? 'bg-white text-black' : 'text-white/50 hover:text-white'
            }`}
          >
            Upload Video
          </button>
          <button
            onClick={() => {
              setSourceType('youtube')
              clearPreview()
            }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              sourceType === 'youtube' ? 'bg-white text-black' : 'text-white/50 hover:text-white'
            }`}
          >
            YouTube URL
          </button>
        </div>
      )}

      {/* Upload Video source */}
      {sourceType === 'upload' && (
        <>
          <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6 mb-6">
            <label className="block text-sm font-semibold text-white mb-3">Upload Video</label>
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setLocalDragging(true)
              }}
              onDragLeave={() => setLocalDragging(false)}
              onDrop={(e) => {
                e.preventDefault()
                setLocalDragging(false)
                const f = e.dataTransfer.files[0]
                if (f) acceptLocalFile(f)
              }}
              onClick={() => localFileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
                localDragging
                  ? 'border-emerald-400 bg-emerald-500/10'
                  : 'border-white/20 hover:border-white/40 hover:bg-white/5'
              }`}
            >
              <input
                ref={localFileInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) acceptLocalFile(f)
                }}
              />
              <div className="text-4xl mb-2">🎬</div>
              <p className="text-white font-medium">
                {localFile ? localFile.name : 'Drop a video here or click to choose'}
              </p>
              <p className="text-white/40 text-xs mt-1">MP4, MOV, WebM · Max 500MB · Trimmed in-browser</p>
              {localFileError && <p className="text-red-400 text-sm mt-2">{localFileError}</p>}
            </div>
          </div>

          {localFileUrl && (
            <div className="rounded-2xl border border-white/10 bg-black overflow-hidden mb-6">
              <video
                ref={localVideoRef}
                src={localFileUrl}
                controls
                className="w-full aspect-video"
                playsInline
              />
            </div>
          )}
        </>
      )}

      {/* YouTube URL source */}
      {sourceType === 'youtube' && (
        <>
          <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6 mb-6">
            <label className="block text-sm font-semibold text-white mb-2">YouTube URL</label>
            <div className="flex gap-3">
              <input
                type="text"
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleLoad()}
                placeholder="https://www.youtube.com/watch?v=..."
                className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white placeholder:text-white/30 focus:outline-none focus:border-emerald-500/50 transition-colors"
              />
              <button
                onClick={handleLoad}
                className="px-6 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl transition-colors flex-shrink-0"
              >
                Load
              </button>
            </div>
            {urlError && <p className="text-red-400 text-sm mt-2">{urlError}</p>}
          </div>

          {videoId && (
            <div className="rounded-2xl border border-white/10 bg-black overflow-hidden mb-6">
              <div className="aspect-video">
                <div id="yt-player" className="w-full h-full" />
              </div>
            </div>
          )}
        </>
      )}

      {/* Mark Start / Mark End */}
      {hasSource && (
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6 mb-6">
          <div className="flex flex-wrap items-center gap-4">
            <button
              onClick={handleMarkStart}
              disabled={sourceType === 'youtube' ? !ytReady : !localFileUrl}
              className="px-5 py-2.5 bg-white/10 hover:bg-white/15 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold rounded-xl border border-white/10 transition-colors"
            >
              Mark Start
            </button>
            <button
              onClick={handleMarkEnd}
              disabled={sourceType === 'youtube' ? !ytReady : !localFileUrl}
              className="px-5 py-2.5 bg-white/10 hover:bg-white/15 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold rounded-xl border border-white/10 transition-colors"
            >
              Mark End
            </button>

            <div className="flex items-center gap-3 text-sm ml-auto">
              <span className="text-white/50">
                Start:{' '}
                <span className="text-white font-mono">{formatTime(startTime)}</span>
              </span>
              <span className="text-white/30">|</span>
              <span className="text-white/50">
                End:{' '}
                <span className="text-white font-mono">{formatTime(endTime)}</span>
              </span>
              {duration !== null && (
                <>
                  <span className="text-white/30">|</span>
                  <span className="text-emerald-400 font-mono font-medium">
                    {duration.toFixed(1)}s
                  </span>
                </>
              )}
            </div>
          </div>
          {startTime !== null && endTime !== null && endTime <= startTime && (
            <p className="text-red-400 text-sm mt-2">
              End time must be after start time.
            </p>
          )}
        </div>
      )}

      {/* Preview + Speed */}
      {hasSource && startTime !== null && endTime !== null && endTime > startTime && (
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6 mb-6 space-y-5">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <h2 className="text-lg font-semibold text-white">Preview & Speed</h2>
            <button
              onClick={handlePreview}
              disabled={previewing}
              className="px-5 py-2 bg-white/10 hover:bg-white/15 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold rounded-xl border border-white/10 transition-colors flex items-center gap-2 text-sm"
            >
              {previewing && (
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              )}
              {previewing ? 'Generating...' : previewUrl ? 'Re-generate Preview' : 'Preview Clip'}
            </button>
          </div>

          {/* Speed selector */}
          <div>
            <label className="block text-sm font-medium text-white/70 mb-2">
              Playback Speed{' '}
              <span className="text-white/40">(applied when saved — audio dropped if not Normal)</span>
            </label>
            <div className="flex flex-wrap gap-2">
              {SPEED_OPTIONS.map((opt) => (
                <button
                  key={opt.label}
                  onClick={() => handleSpeedChange(opt.factor)}
                  className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors border ${
                    Math.abs(speedFactor - opt.factor) < 1e-6
                      ? 'bg-emerald-500 border-emerald-500 text-white'
                      : 'bg-white/5 border-white/10 text-white/60 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {previewError && (
            <p className="text-red-400 text-sm whitespace-pre-wrap break-words">
              Preview failed: {previewError}
            </p>
          )}

          {previewing && sourceType === 'upload' && ffmpegProgress !== null && (
            <div className="space-y-1">
              <p className="text-white/50 text-xs">
                Trimming in browser… {Math.round(ffmpegProgress * 100)}%
              </p>
              <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-400 transition-all"
                  style={{ width: `${Math.round(ffmpegProgress * 100)}%` }}
                />
              </div>
            </div>
          )}

          {previewUrl ? (
            <div className="rounded-xl border border-white/10 bg-black overflow-hidden">
              <video
                key={previewUrl}
                src={previewUrl}
                controls
                className="w-full aspect-video"
              />
              <div className="px-4 py-2 text-xs text-white/50 border-t border-white/10">
                Previewed at{' '}
                <span className="text-white/80 font-mono">
                  {SPEED_OPTIONS.find((o) => Math.abs(o.factor - (previewedSpeedFactor ?? 1)) < 1e-6)?.label ?? '1×'}
                </span>
                . If this looks right, fill out the metadata below and Save.
              </div>
            </div>
          ) : (
            <p className="text-white/40 text-sm">
              {previewedSpeedFactor !== null && previewedSpeedFactor !== speedFactor
                ? 'Speed changed — click “Re-generate Preview” to see the new result.'
                : 'No preview yet. Click “Preview Clip” to download the selected segment and verify quality before saving.'}
            </p>
          )}
        </div>
      )}

      {/* Metadata form */}
      {hasSource && (
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6 mb-6 space-y-5">
          <h2 className="text-lg font-semibold text-white">Clip Metadata</h2>

          {/* Pro name */}
          <div>
            <label className="block text-sm font-medium text-white/70 mb-1.5">
              Pro Name
            </label>
            <input
              type="text"
              list="pro-names"
              value={proName}
              onChange={(e) => {
                setProName(e.target.value)
                setFuzzySuggestions(null)
              }}
              placeholder="e.g. Roger Federer"
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white placeholder:text-white/30 focus:outline-none focus:border-emerald-500/50 transition-colors"
            />
            <datalist id="pro-names">
              {pros.map((p) => (
                <option key={p.name} value={p.name} />
              ))}
            </datalist>

            {fuzzySuggestions && fuzzySuggestions.length > 0 && (
              <div className="mt-3 rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 space-y-3">
                <p className="text-sm text-amber-200">
                  <strong>Did you mean:</strong> “{fuzzySuggestions[0]}”?
                  {fuzzySuggestions.length > 1 && (
                    <span className="text-amber-200/60">
                      {' '}
                      (also: {fuzzySuggestions.slice(1).map((s) => `“${s}”`).join(', ')})
                    </span>
                  )}
                </p>
                <div className="flex flex-wrap gap-2">
                  {fuzzySuggestions.map((s) => (
                    <button
                      key={s}
                      onClick={() => {
                        setProName(s)
                        setFuzzySuggestions(null)
                      }}
                      className="px-3 py-1.5 text-xs font-medium rounded-lg bg-emerald-500 hover:bg-emerald-400 text-white transition-colors"
                    >
                      Use “{s}”
                    </button>
                  ))}
                  <button
                    onClick={() => {
                      setFuzzySuggestions(null)
                      handleSave({ confirmNewPro: true })
                    }}
                    disabled={saving}
                    className="px-3 py-1.5 text-xs font-medium rounded-lg bg-white/10 hover:bg-white/15 text-white/80 border border-white/10 transition-colors"
                  >
                    Create new pro anyway
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Nationality — only shown for new pros */}
          {proName.trim() !== '' && !proNameIsKnown && (
            <div>
              <label className="block text-sm font-medium text-white/70 mb-1.5">
                Nationality{' '}
                <span className="text-white/40">(new pro)</span>
              </label>
              <input
                type="text"
                value={nationality}
                onChange={(e) => setNationality(e.target.value)}
                placeholder="e.g. Switzerland"
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white placeholder:text-white/30 focus:outline-none focus:border-emerald-500/50 transition-colors"
              />
            </div>
          )}

          {/* Shot type */}
          <div>
            <label className="block text-sm font-medium text-white/70 mb-2">
              Shot Type
            </label>
            <div className="flex flex-wrap gap-2">
              {SHOT_TYPES.map((st) => (
                <button
                  key={st}
                  onClick={() => setShotType(st)}
                  className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors border ${
                    shotType === st
                      ? 'bg-emerald-500 border-emerald-500 text-white'
                      : 'bg-white/5 border-white/10 text-white/60 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  {st.charAt(0).toUpperCase() + st.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Camera angle */}
          <div>
            <label className="block text-sm font-medium text-white/70 mb-2">
              Camera Angle
            </label>
            <div className="flex flex-wrap gap-2">
              {CAMERA_ANGLES.map((ca) => (
                <button
                  key={ca}
                  onClick={() => setCameraAngle(ca)}
                  className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors border ${
                    cameraAngle === ca
                      ? 'bg-emerald-500 border-emerald-500 text-white'
                      : 'bg-white/5 border-white/10 text-white/60 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  {ca
                    .split('_')
                    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
                    .join(' ')}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Action buttons */}
      {hasSource && (
        <div className="flex items-center gap-4 mb-6">
          <button
            onClick={() => handleSave()}
            disabled={!canSave}
            className="px-6 py-3 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-colors flex items-center gap-2"
          >
            {saving && (
              <svg
                className="animate-spin h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
            )}
            {saving ? 'Saving...' : 'Save Clip'}
          </button>

          <button
            onClick={handleDownload}
            disabled={!canDownload}
            title={
              !previewUrl
                ? 'Generate a preview first'
                : previewedSpeedFactor !== speedFactor
                  ? 'Re-generate preview at the selected speed first'
                  : !proName.trim() || !shotType || !cameraAngle
                    ? 'Fill in pro, shot, and angle first'
                    : 'Download the previewed clip to your computer'
            }
            className="px-6 py-3 bg-sky-500 hover:bg-sky-400 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-colors"
          >
            Download Clip
          </button>

          <button
            onClick={handleReset}
            className="px-6 py-3 bg-white/10 hover:bg-white/15 text-white font-semibold rounded-xl border border-white/10 transition-colors"
          >
            Reset
          </button>

          {saveResult && (
            <p
              className={`text-sm font-medium ${
                saveResult.ok ? 'text-emerald-400' : 'text-red-400'
              }`}
            >
              {saveResult.msg}
            </p>
          )}
        </div>
      )}

      {/* Saved clips table */}
      {savedClips.length > 0 && (
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden">
          <div className="p-4 border-b border-white/5">
            <h2 className="text-lg font-semibold text-white">
              Saved Clips{' '}
              <span className="text-white/40 font-normal text-sm">
                ({savedClips.length})
              </span>
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/5 text-white/40 text-left">
                  <th className="px-4 py-3 font-medium">Pro</th>
                  <th className="px-4 py-3 font-medium">Shot</th>
                  <th className="px-4 py-3 font-medium">Angle</th>
                  <th className="px-4 py-3 font-medium">Duration</th>
                  <th className="px-4 py-3 font-medium">Speed</th>
                  <th className="px-4 py-3 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {savedClips.map((clip, i) => (
                  <tr key={i} className="border-b border-white/5 last:border-0">
                    <td className="px-4 py-3 text-white">{clip.proName}</td>
                    <td className="px-4 py-3 text-white/70 capitalize">{clip.shotType}</td>
                    <td className="px-4 py-3 text-white/70 capitalize">
                      {clip.cameraAngle.replace('_', ' ')}
                    </td>
                    <td className="px-4 py-3 text-white/70 font-mono">
                      {clip.duration.toFixed(1)}s
                    </td>
                    <td className="px-4 py-3 text-white/70 font-mono">
                      {SPEED_OPTIONS.find((o) => Math.abs(o.factor - clip.speedFactor) < 1e-6)?.label ?? `${clip.speedFactor}×`}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full ${
                          clip.status === 'saved'
                            ? 'bg-emerald-500/10 text-emerald-400'
                            : 'bg-red-500/10 text-red-400'
                        }`}
                      >
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            clip.status === 'saved' ? 'bg-emerald-400' : 'bg-red-400'
                          }`}
                        />
                        {clip.status === 'saved' ? 'Saved' : 'Error'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
