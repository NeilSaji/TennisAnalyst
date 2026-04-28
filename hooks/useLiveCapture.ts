'use client'

import { useCallback, useRef, useState } from 'react'
import { createPoseDetector, type PoseDetector, type PoseDetectStats } from '@/lib/browserPose'
import { computeJointAngles } from '@/lib/jointAngles'
import { isBodyVisible, isFrameConfident, smoothFrames } from '@/lib/poseSmoothing'
import { classifyCameraAngle } from '@/lib/cameraAngle'
import { LiveSwingDetector, type StreamedSwing } from '@/lib/liveSwingDetector'
import type { PoseFrame, Landmark, KeypointsJson } from '@/lib/supabase'

export type LiveCaptureStatus =
  | 'idle'
  | 'requesting-permissions'
  | 'initializing'
  | 'recording'
  | 'stopping'
  | 'error'

export type LiveSessionResult = {
  blob: Blob
  blobMimeType: string
  keypoints: KeypointsJson
  swings: StreamedSwing[]
  durationMs: number
}

export type PoseQuality = 'good' | 'weak' | 'no-body' | 'wrong-angle'

/**
 * Phase 5D — model-load progress callback. Fires while the YOLO + RTMPose
 * weights download on the very first session. After the first run the
 * loader's IndexedDB/cache layer makes subsequent sessions ready in
 * <500ms; in that case the callback may fire once at (total, total) or
 * not at all. The UX layer is expected to debounce so a sub-300ms cache
 * hit doesn't flash a useless loading state.
 */
export type ModelLoadProgress = (
  loaded: number,
  total: number,
  label: 'yolo' | 'rtmpose',
) => void

export interface UseLiveCaptureOptions {
  onSwing?: (swing: StreamedSwing) => void
  onStatus?: (status: LiveCaptureStatus) => void
  // Target pose-detection rate. Real frames from getUserMedia are typically
  // 30fps; we run detection every Nth callback to cap inference load.
  targetDetectionFps?: number
  /**
   * Fired only on transitions between pose-quality states (not per-frame).
   * 'good' = frame passed the strict body-presence gate (`isBodyVisible`)
   *          AND the camera-angle classifier returned 'side-on' or 'oblique'.
   * 'weak' = frame passed `isFrameConfident` but not `isBodyVisible` —
   *          the player is only partially in frame (face-only, legs cut).
   * 'wrong-angle' = body is visible, but the camera angle is 'front-on' or
   *          'unknown' — the swing-mechanics signal we grade off doesn't
   *          project usefully into 2D, so we suppress detector input.
   * 'no-body' = no usable detection in the last ~1s.
   *
   * Phase 1/1.5 only: callers are expected to set up the wiring; the UI
   * pill that consumes this lives in Phase 3. Detector behavior already
   * does the right thing without a UI consumer (face-only and wrong-angle
   * frames are excluded from the detector).
   */
  onPoseQuality?: (q: PoseQuality) => void
  /**
   * Fires once per detection-tick that survives the body-presence gate
   * (the same frames that drive the swing detector). Phase 3 wires this
   * into the on-screen skeleton overlay — the consumer is expected to
   * stash the latest frame in a ref and draw it from a single rAF loop
   * rather than re-rendering React state at 15fps.
   */
  onPoseFrame?: (frame: PoseFrame) => void
  /**
   * Phase 5D — model-load progress callback. The browser pose detector
   * fetches YOLO11n + RTMPose-m on first instantiation; this fires while
   * those bytes arrive so the panel can show a "Loading pose model…"
   * progress bar before recording begins. Subsequent sessions hit the
   * loader's cache and resolve in <500ms.
   */
  onModelLoadProgress?: ModelLoadProgress
}

export interface UseLiveCaptureReturn {
  start: (
    videoEl: HTMLVideoElement,
    opts?: {
      facingMode?: 'user' | 'environment'
      // Camera aspect orientation. 'portrait' requests 720x1280 to match
      // a phone held vertically; 'landscape' requests 1280x720. Pick this
      // at the panel layer based on viewport orientation — getting a
      // portrait stream into a landscape container (or vice versa) makes
      // object-cover crop half the player away.
      aspect?: 'portrait' | 'landscape'
    },
  ) => Promise<void>
  stop: () => Promise<LiveSessionResult | null>
  abort: () => void
  status: LiveCaptureStatus
  error: string | null
  isRecording: boolean
  pickedMimeType: string | null
  /**
   * Camera facing actually used by the active session. 'user' = selfie /
   * front camera (mirror the on-screen view). 'environment' = back
   * camera (do NOT mirror — the user is filming away from themselves).
   * Null while no session is running.
   */
  facingMode: 'user' | 'environment' | null
  /**
   * Snapshot of the most recent ONNX pipeline call's diagnostics —
   * used by the live UI to surface a debug pill without DevTools.
   * Returns null when no detector exists yet (pre-start).
   */
  getLastDetectStats: () => PoseDetectStats | null
}

// Prefer mp4 (iOS native), fall back to webm variants (Chrome/Android).
const CANDIDATE_MIME_TYPES = [
  'video/mp4;codecs=avc1',
  'video/mp4',
  'video/webm;codecs=vp9,opus',
  'video/webm;codecs=vp8,opus',
  'video/webm',
]

function pickMimeType(): string | null {
  if (typeof MediaRecorder === 'undefined') return null
  for (const m of CANDIDATE_MIME_TYPES) {
    try {
      if (MediaRecorder.isTypeSupported(m)) return m
    } catch {
      // Some browsers throw on malformed codec strings — just try the next one
    }
  }
  return null
}

// How long after the last good/weak detection before we declare the
// player gone. Matches the Phase 3 spec "no-body if no detection for ~1s".
const NO_BODY_TIMEOUT_MS = 1000

export function useLiveCapture(
  options: UseLiveCaptureOptions = {},
): UseLiveCaptureReturn {
  const {
    onSwing,
    onStatus,
    onPoseQuality,
    onPoseFrame,
    onModelLoadProgress,
    targetDetectionFps = 15,
  } = options

  const [status, setStatusState] = useState<LiveCaptureStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [pickedMimeType, setPickedMimeType] = useState<string | null>(null)
  const [facingMode, setFacingMode] = useState<'user' | 'environment' | null>(null)

  const generationRef = useRef(0)
  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const videoElRef = useRef<HTMLVideoElement | null>(null)
  const rvfcHandleRef = useRef<number | null>(null)
  const fallbackIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const framesRef = useRef<PoseFrame[]>([])
  const swingsRef = useRef<StreamedSwing[]>([])
  const detectorRef = useRef<LiveSwingDetector | null>(null)
  // Browser pose detector (YOLO + RTMPose via onnxruntime-web). Created in
  // start(), released in cleanup(). Phase 5D replaced MediaPipe here; the
  // 33-entry BlazePose landmark shape is identical so downstream gates and
  // smoothing remain unchanged.
  const poseDetectorRef = useRef<PoseDetector | null>(null)
  const startedAtRef = useRef<number>(0)
  const frameCounterRef = useRef<number>(0)
  const lastDetectAtRef = useRef<number>(0)
  // Track only the last *emitted* quality so we fire onPoseQuality on
  // transitions, not on every frame.
  const lastQualityRef = useRef<PoseQuality | null>(null)
  // Wall-clock of the last frame where ANY pose was detected (good or
  // weak). Used to flip to 'no-body' when nothing has been detected for
  // NO_BODY_TIMEOUT_MS. We also poll this inside the detection loop on
  // frames where the detector returns null.
  const lastDetectionAtRef = useRef<number>(0)
  // Guard re-entry of runOneDetection: detect() is now async, and we
  // don't want to fire a second inference while the first is still in
  // flight (back-pressure from a slow ONNX provider would otherwise pile
  // up frames).
  const inflightRef = useRef<boolean>(false)

  const setStatus = useCallback(
    (next: LiveCaptureStatus) => {
      setStatusState(next)
      onStatus?.(next)
    },
    [onStatus],
  )

  const teardownLoop = useCallback(() => {
    const videoEl = videoElRef.current
    if (videoEl && rvfcHandleRef.current != null && 'cancelVideoFrameCallback' in videoEl) {
      try {
        ;(videoEl as unknown as { cancelVideoFrameCallback: (h: number) => void }).cancelVideoFrameCallback(
          rvfcHandleRef.current,
        )
      } catch {
        // ignore — best effort
      }
    }
    rvfcHandleRef.current = null
    if (fallbackIntervalRef.current != null) {
      clearInterval(fallbackIntervalRef.current)
      fallbackIntervalRef.current = null
    }
  }, [])

  const cleanup = useCallback(() => {
    teardownLoop()
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      try { recorderRef.current.stop() } catch { /* ignore */ }
    }
    recorderRef.current = null
    if (streamRef.current) {
      for (const t of streamRef.current.getTracks()) {
        try { t.stop() } catch { /* ignore */ }
      }
    }
    streamRef.current = null
    if (videoElRef.current) {
      try {
        videoElRef.current.srcObject = null
      } catch { /* ignore */ }
    }
    videoElRef.current = null
    canvasRef.current = null
    detectorRef.current = null
    if (poseDetectorRef.current) {
      try { poseDetectorRef.current.dispose() } catch { /* ignore */ }
    }
    poseDetectorRef.current = null
    inflightRef.current = false
    setFacingMode(null)
  }, [teardownLoop])

  const abort = useCallback(() => {
    generationRef.current++
    cleanup()
    setStatus('idle')
  }, [cleanup, setStatus])

  const start = useCallback(async (
    videoEl: HTMLVideoElement,
    opts: {
      facingMode?: 'user' | 'environment'
      aspect?: 'portrait' | 'landscape'
    } = {},
  ) => {
    const requestedFacingMode = opts.facingMode ?? 'environment'
    const requestedAspect = opts.aspect ?? 'landscape'
    // Phone in portrait wants 720w x 1280h; laptop / phone-in-landscape
    // wants 1280w x 720h. Browsers honor this as a hint, not a hard
    // constraint — they pick the closest supported camera mode.
    const requestedWidth = requestedAspect === 'portrait' ? 720 : 1280
    const requestedHeight = requestedAspect === 'portrait' ? 1280 : 720
    const generation = ++generationRef.current
    setError(null)
    framesRef.current = []
    swingsRef.current = []
    chunksRef.current = []
    frameCounterRef.current = 0
    lastDetectAtRef.current = 0
    lastQualityRef.current = null
    lastDetectionAtRef.current = 0
    inflightRef.current = false
    detectorRef.current = new LiveSwingDetector()

    setStatus('requesting-permissions')
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: requestedFacingMode },
          width: { ideal: requestedWidth },
          height: { ideal: requestedHeight },
        },
        audio: true,
      })
    } catch (err) {
      if (generationRef.current !== generation) return
      setError(err instanceof Error ? err.message : 'Camera or microphone permission denied')
      setStatus('error')
      return
    }
    if (generationRef.current !== generation) {
      for (const t of stream.getTracks()) t.stop()
      return
    }
    streamRef.current = stream
    // Read back the facing mode actually granted — on devices with only
    // one camera (laptops) the browser ignores `ideal: 'environment'`
    // and gives us the front cam, so the UI needs to know the truth
    // before deciding whether to mirror. Defensive: not every test
    // double / older browser exposes getVideoTracks/getSettings, so
    // fall back to the requested mode if introspection fails.
    let grantedFacingMode: 'user' | 'environment' = requestedFacingMode
    try {
      const videoTrack = stream.getVideoTracks?.()?.[0]
      const settings = videoTrack?.getSettings?.()
      if (settings?.facingMode === 'user' || settings?.facingMode === 'environment') {
        grantedFacingMode = settings.facingMode
      }
    } catch {
      // getSettings() throws on a few mobile browsers — keep requestedFacingMode
    }
    setFacingMode(grantedFacingMode)

    setStatus('initializing')

    // Create the ONNX pose detector. First-session cold-start downloads
    // ~50MB of model weights and surfaces progress through
    // onModelLoadProgress; subsequent sessions hit the loader's cache and
    // resolve in <500ms.
    let poseDetector: PoseDetector | null = null
    try {
      poseDetector = await createPoseDetector({
        onProgress: onModelLoadProgress,
      })
    } catch (err) {
      if (generationRef.current !== generation) return
      setError(err instanceof Error ? err.message : 'Pose model failed to load')
      cleanup()
      setStatus('error')
      return
    }

    if (generationRef.current !== generation) {
      try { poseDetector?.dispose() } catch { /* ignore */ }
      return
    }
    poseDetectorRef.current = poseDetector

    videoElRef.current = videoEl
    videoEl.srcObject = stream
    videoEl.muted = true
    videoEl.playsInline = true
    try {
      await videoEl.play()
    } catch (err) {
      if (generationRef.current !== generation) return
      setError(err instanceof Error ? err.message : 'Preview playback failed')
      cleanup()
      setStatus('error')
      return
    }

    // Build MediaRecorder after the stream is live to give the encoder a
    // defined source resolution.
    const mimeType = pickMimeType()
    if (!mimeType) {
      setError('This browser does not support MediaRecorder')
      cleanup()
      setStatus('error')
      return
    }
    setPickedMimeType(mimeType)

    const recorder = new MediaRecorder(stream, { mimeType })
    recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) chunksRef.current.push(ev.data)
    }
    recorder.start(1000) // request a chunk every second so a mid-session crash keeps partial data
    recorderRef.current = recorder
    startedAtRef.current = performance.now()

    const canvas = document.createElement('canvas')
    canvas.width = videoEl.videoWidth || 640
    canvas.height = videoEl.videoHeight || 360
    canvasRef.current = canvas
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) {
      setError('2D canvas context unavailable')
      cleanup()
      setStatus('error')
      return
    }

    const minFrameGapMs = 1000 / targetDetectionFps

    const emitQuality = (q: PoseQuality) => {
      if (lastQualityRef.current === q) return
      lastQualityRef.current = q
      onPoseQuality?.(q)
    }

    const runOneDetection = async (nowMs: number) => {
      if (generationRef.current !== generation) return
      if (nowMs - lastDetectAtRef.current < minFrameGapMs) return
      // Drop frames while a previous inference is still running. ONNX
      // detect() is async (and on WASM/CPU can be slow); we'd rather
      // skip a tick than queue up backlog.
      if (inflightRef.current) return
      lastDetectAtRef.current = nowMs

      const detector = poseDetectorRef.current
      if (!detector) return

      inflightRef.current = true
      try {
        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height)
        const landmarks: Landmark[] | null = await detector.detect(canvas)
        if (generationRef.current !== generation) return
        if (!landmarks || landmarks.length === 0) {
          // No pose at all this frame. If it's been long enough since the
          // last hit, declare the player gone.
          if (
            lastDetectionAtRef.current > 0 &&
            nowMs - lastDetectionAtRef.current >= NO_BODY_TIMEOUT_MS
          ) {
            emitQuality('no-body')
          } else if (lastDetectionAtRef.current === 0) {
            // Never seen anyone yet — start in 'no-body' so the UI can
            // prompt "move into frame" before the first detection lands.
            emitQuality('no-body')
          }
          return
        }
        if (!isFrameConfident(landmarks)) {
          if (
            lastDetectionAtRef.current > 0 &&
            nowMs - lastDetectionAtRef.current >= NO_BODY_TIMEOUT_MS
          ) {
            emitQuality('no-body')
          } else if (lastDetectionAtRef.current === 0) {
            // First seconds of the session and the detector is returning
            // landmarks but at low confidence. Without this branch the
            // pill stayed null forever — user saw no signal at all.
            emitQuality('no-body')
          }
          return
        }
        // We have a usable pose — refresh the "last detected at" clock so
        // the no-body timer resets even if the body-visible gate fails.
        lastDetectionAtRef.current = nowMs

        const frame: PoseFrame = {
          frame_index: frameCounterRef.current++,
          timestamp_ms: Math.round(nowMs - startedAtRef.current),
          landmarks,
          joint_angles: computeJointAngles(landmarks),
        }
        // Always push to framesRef so /analyze still has the full sequence
        // even on weak frames. Only the strict gate below decides whether
        // this frame drives the live swing detector.
        framesRef.current.push(frame)

        if (!isBodyVisible(landmarks)) {
          // Frame was loosely confident but full body wasn't visible
          // (face-only, legs cut off, subject too small). Don't feed the
          // detector — that would let face-only fidget become a swing.
          emitQuality('weak')
          return
        }

        // Phase 1.5: camera-angle gate. Body is in frame, but if the
        // camera is square-on (or we can't tell), the swing mechanics we
        // grade off collapse into the depth axis where the model's z is
        // noise. Treat this exactly like a weak frame: keep the frame in
        // framesRef so /analyze still has it, but don't feed the live
        // detector and don't count it toward warmup.
        const cameraAngle = classifyCameraAngle(landmarks)
        if (cameraAngle === 'front-on' || cameraAngle === 'unknown') {
          emitQuality('wrong-angle')
          return
        }

        emitQuality('good')

        // Hand the latest body-visible frame to the overlay layer (rAF
        // loop in LiveCapturePanel reads it from a ref). Fired only on
        // good frames so the skeleton never draws over a partial body.
        onPoseFrame?.(frame)

        const emitted = detectorRef.current?.feed(frame) ?? null
        if (emitted) {
          swingsRef.current.push(emitted)
          onSwing?.(emitted)
        }
      } catch {
        // A single bad frame should never kill the loop
      } finally {
        inflightRef.current = false
      }
    }

    const rvfcSupported =
      typeof videoEl.requestVideoFrameCallback === 'function' &&
      typeof videoEl.cancelVideoFrameCallback === 'function'

    if (rvfcSupported) {
      const schedule = () => {
        if (generationRef.current !== generation) return
        rvfcHandleRef.current = videoEl.requestVideoFrameCallback((now) => {
          // Fire-and-forget: schedule the next callback immediately so
          // the rVFC cadence isn't gated on inference latency. The
          // inflight guard inside runOneDetection prevents pile-up.
          void runOneDetection(now)
          schedule()
        })
      }
      schedule()
    } else {
      fallbackIntervalRef.current = setInterval(
        () => { void runOneDetection(performance.now()) },
        Math.round(minFrameGapMs),
      )
    }

    setStatus('recording')
  }, [cleanup, onSwing, onPoseQuality, onPoseFrame, onModelLoadProgress, setStatus, targetDetectionFps])

  const stop = useCallback(async (): Promise<LiveSessionResult | null> => {
    const generation = generationRef.current
    const recorder = recorderRef.current
    const stream = streamRef.current

    if (!recorder || !stream) {
      cleanup()
      setStatus('idle')
      return null
    }

    setStatus('stopping')
    teardownLoop()

    const finalBlob = await new Promise<Blob>((resolve) => {
      const finalize = () => {
        const mimeType = recorder.mimeType || 'video/webm'
        const blob = new Blob(chunksRef.current, { type: mimeType })
        resolve(blob)
      }
      if (recorder.state === 'inactive') {
        finalize()
        return
      }
      recorder.onstop = () => finalize()
      try {
        recorder.stop()
      } catch {
        finalize()
      }
    })

    if (generationRef.current !== generation) {
      cleanup()
      return null
    }

    for (const t of stream.getTracks()) {
      try { t.stop() } catch { /* ignore */ }
    }

    const durationMs = Math.round(performance.now() - startedAtRef.current)

    // Final smoothing pass for persistence quality — matches what /analyze
    // ingests for uploaded clips.
    const smoothed = smoothFrames(framesRef.current)
    const keypoints: KeypointsJson = {
      fps_sampled: targetDetectionFps,
      frame_count: smoothed.length,
      frames: smoothed,
      schema_version: 2,
    }

    const result: LiveSessionResult = {
      blob: finalBlob,
      blobMimeType: recorder.mimeType || 'video/webm',
      keypoints,
      swings: swingsRef.current.slice(),
      durationMs,
    }

    cleanup()
    setStatus('idle')
    return result
  }, [cleanup, setStatus, targetDetectionFps, teardownLoop])

  return {
    start,
    stop,
    abort,
    status,
    error,
    isRecording: status === 'recording',
    pickedMimeType,
    facingMode,
    getLastDetectStats: () => poseDetectorRef.current?.getLastStats() ?? null,
  }
}
