'use client'

import { getPoseLandmarker, getMonotonicTimestamp } from '@/lib/mediapipe'
import { computeJointAngles } from '@/lib/jointAngles'
import { isFrameConfident, smoothFrames } from '@/lib/poseSmoothing'
import type { PoseFrame, Landmark } from '@/lib/supabase'

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
        const ts = getMonotonicTimestamp(currentTime * 1000)
        const result = poseLandmarker.detectForVideo(canvas, ts)
        if (result.landmarks?.[0]?.length) {
          const rawLandmarks = result.landmarks[0]
          const landmarks: Landmark[] = rawLandmarks.map(
            (lm: { x: number; y: number; z?: number; visibility?: number }, id: number) => ({
              id,
              name: `landmark_${id}`,
              x: lm.x,
              y: lm.y,
              z: lm.z ?? 0,
              visibility: lm.visibility ?? 1,
            })
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
