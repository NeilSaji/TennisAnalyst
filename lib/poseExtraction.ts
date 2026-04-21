'use client'

import { getPoseLandmarker } from '@/lib/mediapipe'
import { computeJointAngles } from '@/lib/jointAngles'
import { isFrameConfident, smoothFrames } from '@/lib/poseSmoothing'
import type { PoseFrame, Landmark } from '@/lib/supabase'

// Side of the square canvas used for the pass-2 crop. MediaPipe's Heavy
// model internally runs at 256x256; 512 gives headroom for the upscaled
// crop without wasting too much GPU time.
const CROP_CANVAS_SIZE = 512
// Expansion factor around the pass-1 bounding box. 1.5x leaves breathing
// room for the arm/racket to extend outside the initial detection without
// clipping when we crop for pass 2.
const CROP_MARGIN = 1.5
// Don't bother with the second pass if the subject already fills most of
// the frame — the upscale gains are marginal and the extra detect call
// doubles extraction time for no benefit.
const CROP_SKIP_THRESHOLD = 0.85

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
    if ((lm.visibility ?? 0) < 0.3) continue
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

export type ExtractResult = {
  frames: PoseFrame[]
  fps: number
  // The object URL created for the hidden <video>. Caller is responsible for
  // revoking it (or handing it off for playback). Null on abort.
  objectUrl: string | null
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
    await new Promise<void>((resolve) => {
      videoEl.onloadedmetadata = () => resolve()
    })
    check()

    // loadedmetadata only guarantees dimensions/duration, not a decodable frame.
    if (videoEl.readyState < 3) {
      await new Promise<void>((resolve) => {
        videoEl.oncanplay = () => {
          videoEl.oncanplay = null
          resolve()
        }
      })
    }
    check()

    const duration = videoEl.duration
    const frameInterval = 1 / fps
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
        const crop = computeCropRect(pass1.landmarks[0])
        if (crop && crop.size < CROP_SKIP_THRESHOLD) {
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
          (lm: RawLandmark, id: number) => ({
            id,
            name: `landmark_${id}`,
            x: lm.x,
            y: lm.y,
            z: lm.z ?? 0,
            visibility: lm.visibility ?? 1,
          }),
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

    const smoothed = smoothFrames(frames)
    return { frames: smoothed, fps, objectUrl }
  } catch (err) {
    // On abort or error, release the object URL — nobody's going to use it
    URL.revokeObjectURL(objectUrl)
    if (err instanceof AbortError) {
      return { frames: [], fps, objectUrl: null }
    }
    throw err
  } finally {
    videoEl.src = ''
    videoEl.removeAttribute('src')
    videoEl.load()
  }
}
