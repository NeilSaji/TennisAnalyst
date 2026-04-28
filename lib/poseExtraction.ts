'use client'

import { getPoseLandmarker } from '@/lib/mediapipe'
import { computeJointAngles } from '@/lib/jointAngles'
import {
  isFrameConfident,
  smoothFrames,
  filterImplausibleArmJoints,
} from '@/lib/poseSmoothing'
import type { PoseFrame, Landmark } from '@/lib/supabase'

// Side of the square canvas used for the pass-2 crop. MediaPipe's Heavy
// model internally runs at 256x256; 512 gives headroom for the upscaled
// crop without wasting too much GPU time.
const CROP_CANVAS_SIZE = 512
// Expansion factor around the pass-1 bounding box. 1.5x leaves breathing
// room for the arm/racket to extend outside the initial detection without
// clipping when we crop for pass 2.
const CROP_MARGIN = 1.5
// Only landmarks above this visibility count toward the pass-1 bbox.
// On a far/noisy subject, MediaPipe sprays a few spurious "confident"
// landmarks across the court — a higher gate keeps the bbox centered
// on the actual player rather than the most confident outlier.
const CROP_BBOX_VIS_GATE = 0.5
// Pass-2 is kept disabled for now. User report on cea292d-d6ebff7
// showed the skeleton rendering as a correctly-shaped body displaced
// horizontally from the player — a coordinate-transform bug introduced
// by drawing a non-square (16:9-proportioned) "square-in-normalized-
// coords" region into a 1:1 canvas. Math looks linearly correct, but
// MediaPipe's detection on the aspect-stretched image plus the
// transform-back produced a shifted result in practice. Turn this off
// while we ship a known-good single-pass path; revisit only after we
// have a reproducible minimal case.
const CROP_ENABLED = false

type RawLandmark = { x: number; y: number; z?: number; visibility?: number }

// Compute a square, frame-clamped crop rectangle (in [0,1] normalized
// coords) around the pass-1 landmarks. Returns null when the pass-1
// detection is too sparse to trust.
function computeCropRect(landmarks: RawLandmark[]): {
  x: number
  y: number
  size: number
} | null {
  let minX = 1
  let minY = 1
  let maxX = 0
  let maxY = 0
  let any = false
  for (const lm of landmarks) {
    if ((lm.visibility ?? 0) < CROP_BBOX_VIS_GATE) continue
    // Also exclude extrapolated landmarks outside the frame — MediaPipe
    // happily projects a wrist below the ground or past the net edge,
    // and including those pulls the bbox off the player.
    if (lm.x < 0 || lm.x > 1 || lm.y < 0 || lm.y > 1) continue
    any = true
    if (lm.x < minX) minX = lm.x
    if (lm.y < minY) minY = lm.y
    if (lm.x > maxX) maxX = lm.x
    if (lm.y > maxY) maxY = lm.y
  }
  if (!any) return null
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  const w = (maxX - minX) * CROP_MARGIN
  const h = (maxY - minY) * CROP_MARGIN
  let size = Math.max(w, h)
  if (size <= 0) return null
  let x = cx - size / 2
  let y = cy - size / 2
  // Clamp into the frame, shrinking size if the expanded box would overhang.
  if (x < 0) x = 0
  if (y < 0) y = 0
  if (x + size > 1) size = 1 - x
  if (y + size > 1) size = 1 - y
  if (size <= 0) return null
  return { x, y, size }
}

// Which extraction backend produced these frames. Surfaced in the UI as
// a small diagnostic chip — when tracing looks bad, the chip tells us in
// 10 seconds whether the user is on the rtmpose-Railway path I've been
// testing locally or some other path (server mediapipe / browser
// fallback) that has different keypoint quality. Without this label the
// "tracing is off" complaint is unobservable.
export type ExtractorBackend =
  | 'rtmpose-railway'
  | 'mediapipe-railway'
  | 'mediapipe-browser'
  | 'mediapipe-browser-fallback'

export type ExtractResult = {
  frames: PoseFrame[]
  fps: number
  // The object URL created for the hidden <video>. Caller is responsible for
  // revoking it (or handing it off for playback). Null on abort.
  objectUrl: string | null
  extractorBackend: ExtractorBackend
}

export type ExtractOptions = {
  fps?: number
  // Called with 0..100 over the course of extraction. Does NOT include upload
  // or model-load time — caller remaps to their own progress bar if needed.
  onProgress?: (pct: number) => void
  abortSignal?: AbortSignal
}

class AbortError extends Error {
  constructor() {
    super('aborted')
    this.name = 'AbortError'
  }
}

// Race a promise against a timeout. The original failure mode (no
// timeout, no user-visible error) was the thing that made iOS Safari
// bugs look like "stuck at 25%" rather than "extraction failed because
// X" — fail fast with a specific message so the caller can surface it.
function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  message: string,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(message)), ms)
    promise.then(
      (v) => {
        clearTimeout(t)
        resolve(v)
      },
      (e) => {
        clearTimeout(t)
        reject(e)
      },
    )
  })
}

/**
 * Extract pose keypoints from every ~1/fps second of a video File.
 *
 * Lifted verbatim from the seek-loop that was duplicated in UploadZone and
 * the /compare page. One deliberate unification: this uses getMonotonicTimestamp
 * for every detectForVideo call, not raw currentTime*1000. The shared
 * PoseLandmarker singleton requires strictly increasing timestamps across the
 * life of the instance, and using raw ts silently breaks on the second page
 * that reuses the singleton.
 */
export async function extractPoseFromVideo(
  source: File,
  opts: ExtractOptions = {}
): Promise<ExtractResult> {
  const fps = opts.fps ?? 30
  const signal = opts.abortSignal
  const check = () => {
    if (signal?.aborted) throw new AbortError()
  }

  const poseLandmarker = await getPoseLandmarker()
  check()

  // Build our own hidden video element. Callers used to manage this ref
  // themselves; now it's fully encapsulated.
  const videoEl = document.createElement('video')
  videoEl.muted = true
  videoEl.playsInline = true
  videoEl.preload = 'auto'

  const objectUrl = URL.createObjectURL(source)
  videoEl.src = objectUrl

  try {
    // Wait for metadata (dimensions/duration). iOS Safari can stall
    // here silently on unsupported codecs — 15s cap + concrete error
    // beats a forever-25% bar.
    await withTimeout(
      new Promise<void>((resolve) => {
        if (videoEl.readyState >= 1) {
          resolve()
          return
        }
        videoEl.onloadedmetadata = () => resolve()
      }),
      15_000,
      'Video metadata never loaded (codec unsupported or file corrupt?). Try re-recording with "Most Compatible" format in iPhone Settings → Camera → Formats.',
    )
    check()

    // iOS Safari needs a play()/pause() to prime the video decoder
    // BEFORE it will advance readyState past 2 or fire canplay. The
    // previous iteration of this fix waited for canplay first and
    // THEN called play() — order was backwards; iOS would sit on
    // readyState 2 indefinitely and the canplay timeout tripped with
    // "Video failed to become playable in time." Correct order: prime
    // first, then check readyState.
    //
    // Muted + playsInline are set above, which satisfies every modern
    // browser's autoplay policy. A raw play() on a just-loaded element
    // can still be rejected on some iOS versions; that's caught and
    // ignored — the seek loop below has a per-frame timeout so even a
    // decoder that never truly primes won't hang the whole extraction.
    try {
      await videoEl.play()
      videoEl.pause()
    } catch {
      // Priming failed. The seek loop's 3s-per-frame timeout still
      // bounds the damage.
    }
    check()

    // readyState 2 = HAVE_CURRENT_DATA, which is enough for drawImage
    // to produce a valid canvas. We used to wait for readyState 3
    // (HAVE_FUTURE_DATA) via canplay, but that's overkill — if
    // individual frames fail to draw the existing per-frame error
    // handling skips them.
    if (videoEl.readyState < 2) {
      await withTimeout(
        new Promise<void>((resolve) => {
          const onReady = () => {
            videoEl.removeEventListener('loadeddata', onReady)
            resolve()
          }
          videoEl.addEventListener('loadeddata', onReady)
        }),
        15_000,
        'Video failed to become readable in time (often HEIF/HEVC on iPhone — try Settings → Camera → Formats → "Most Compatible").',
      )
    }
    check()

    const duration = videoEl.duration

    // Adaptive sampling FPS for long clips. Browser MediaPipe detection
    // runs at ~50-100ms per frame on a phone, so a 2-minute clip at the
    // default 30fps would be 3,600 detections = 3-6 minutes of compute.
    // For analysis purposes (skeleton + joint angles + swing detection)
    // we don't need 30fps — peak swing motion only spans ~10-15 frames
    // even at 15fps, and the visualization stays smooth because the
    // canvas redraws on every video frame regardless of sample density.
    //
    // Tiers tuned so total extraction time stays roughly bounded under
    // ~90 seconds on a typical mobile, regardless of clip length:
    //   <= 30s  → 30 fps (current default; short clips get full fidelity)
    //   30-60s  → 20 fps
    //   60-120s → 15 fps
    //   > 120s  → 10 fps
    const effectiveFps =
      !Number.isFinite(duration) || duration <= 30
        ? fps
        : duration <= 60
          ? Math.min(fps, 20)
          : duration <= 120
            ? Math.min(fps, 15)
            : Math.min(fps, 10)
    if (effectiveFps !== fps) {
      console.info(
        `[poseExtraction] long clip (${duration.toFixed(1)}s) — sampling at ${effectiveFps}fps instead of ${fps}fps to keep extraction under ~90s`,
      )
    }
    const frameInterval = 1 / effectiveFps
    const frames: PoseFrame[] = []
    let frameIndex = 0

    const canvas = document.createElement('canvas')
    canvas.width = videoEl.videoWidth || 640
    canvas.height = videoEl.videoHeight || 360
    const ctx = canvas.getContext('2d')!

    // Pass-2 crop canvas. Square so MediaPipe's model input (also square)
    // isn't letterboxed. Lets us stretch a small crop of the source frame
    // into the detector at effective high resolution — the whole reason
    // small-in-frame subjects (~15% tall) were getting garbage landmarks.
    const cropCanvas = document.createElement('canvas')
    cropCanvas.width = CROP_CANVAS_SIZE
    cropCanvas.height = CROP_CANVAS_SIZE
    const cropCtx = cropCanvas.getContext('2d')!

    // Seek video to a timestamp and wait for onseeked, with 3s timeout per frame
    const seekToFrame = (time: number): Promise<boolean> =>
      new Promise((resolve) => {
        let settled = false
        const settle = (ok: boolean) => {
          if (settled) return
          settled = true
          videoEl.onseeked = null
          resolve(ok)
        }
        const timeout = setTimeout(() => settle(false), 3000)
        videoEl.onseeked = () => {
          clearTimeout(timeout)
          settle(true)
        }
        videoEl.currentTime = time
      })

    while (frameIndex * frameInterval <= duration) {
      check()

      const currentTime = frameIndex * frameInterval
      const seeked = await seekToFrame(currentTime)
      if (!seeked) {
        frameIndex++
        continue
      }

      // Wait one animation frame so the decoded frame is composited and ready
      // for drawImage.
      await new Promise<void>((r) => requestAnimationFrame(() => r()))
      check()

      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height)

      try {
        // Pass 1: locate the player on the full frame. MediaPipe's tiny
        // internal input size means landmarks on a 15%-of-frame subject
        // are noisy, but a bounding box is still reliable enough to crop
        // for the second pass.
        const pass1 = poseLandmarker.detect(canvas)
        if (!pass1.landmarks?.[0]?.length) {
          frameIndex++
          continue
        }
        let finalLandmarks: RawLandmark[] = pass1.landmarks[0]

        // Pass 2: if the subject occupies a smallish portion of the frame,
        // crop around the pass-1 box and re-detect. The crop stretches to
        // fill a 512x512 canvas, so the detector sees the player at much
        // higher effective resolution. Landmarks come back in crop-space
        // and must be transformed back to full-frame normalized coords.
        const crop = CROP_ENABLED ? computeCropRect(pass1.landmarks[0]) : null
        if (crop && crop.size < 0.85) {
          const srcX = crop.x * canvas.width
          const srcY = crop.y * canvas.height
          const srcW = crop.size * canvas.width
          const srcH = crop.size * canvas.height
          cropCtx.clearRect(0, 0, cropCanvas.width, cropCanvas.height)
          cropCtx.drawImage(
            videoEl,
            srcX,
            srcY,
            srcW,
            srcH,
            0,
            0,
            cropCanvas.width,
            cropCanvas.height,
          )
          const pass2 = poseLandmarker.detect(cropCanvas)
          if (pass2.landmarks?.[0]?.length) {
            finalLandmarks = pass2.landmarks[0].map((lm: RawLandmark) => ({
              ...lm,
              x: crop.x + lm.x * crop.size,
              y: crop.y + lm.y * crop.size,
            }))
          }
        }

        const landmarks: Landmark[] = finalLandmarks.map(
          (lm: RawLandmark, id: number) => {
            // Force visibility to 0 for landmarks the model predicted
            // OUTSIDE the frame. MediaPipe will report high "visibility"
            // for extrapolated joints (e.g., a foot below the bottom of
            // a cropped frame), but the user doesn't want predictions —
            // if the joint isn't in view, it should disappear and
            // reappear when it comes back. The render-side 0.6 cutoff
            // then drops these entirely.
            const inFrame =
              lm.x >= 0 && lm.x <= 1 && lm.y >= 0 && lm.y <= 1
            const rawVis = lm.visibility ?? 1
            return {
              id,
              name: `landmark_${id}`,
              x: lm.x,
              y: lm.y,
              z: lm.z ?? 0,
              visibility: inFrame ? rawVis : 0,
            }
          },
        )

        // Skip low-confidence detections (warm-up artifacts, off-camera, etc).
        if (isFrameConfident(landmarks)) {
          const joint_angles = computeJointAngles(landmarks)
          frames.push({
            frame_index: frameIndex,
            timestamp_ms: currentTime * 1000,
            landmarks,
            joint_angles,
          })
        }
      } catch {
        // Skip frames where detection fails
      }

      frameIndex++
      if (opts.onProgress && duration > 0) {
        opts.onProgress(Math.min(100, Math.round((currentTime / duration) * 100)))
      }
    }

    // Smooth first (filtfilt zero-phase), then apply bone-length
    // plausibility so the filter works on the smoothed trajectories.
    // Implausibly-placed elbow/wrist landmarks have their visibility
    // zeroed so the render-side cutoff hides them — partial skeletons
    // through contact instead of wrong-but-visible ones.
    const smoothed = smoothFrames(frames)
    const cleaned = filterImplausibleArmJoints(smoothed)
    // Return the EFFECTIVE fps the loop actually used so downstream
    // smoothing / swing-detection windows compute against real timing.
    // Returning the requested fps would lie about the data density.
    return {
      frames: cleaned,
      fps: effectiveFps,
      objectUrl,
      extractorBackend: 'mediapipe-browser',
    }
  } catch (err) {
    // On abort or error, release the object URL — nobody's going to use it
    URL.revokeObjectURL(objectUrl)
    if (err instanceof AbortError) {
      return {
        frames: [],
        fps,
        objectUrl: null,
        extractorBackend: 'mediapipe-browser',
      }
    }
    throw err
  } finally {
    videoEl.src = ''
    videoEl.removeAttribute('src')
    videoEl.load()
  }
}
