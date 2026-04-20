'use client'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PoseLandmarkerInstance = any

// Store the in-flight Promise so concurrent callers share it.
// On failure the Promise is cleared so the next call retries.
let initPromise: Promise<PoseLandmarkerInstance> | null = null

async function doInit(): Promise<PoseLandmarkerInstance> {
  const vision = await import('@mediapipe/tasks-vision')
  const { PoseLandmarker, FilesetResolver } = vision

  const filesetResolver = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm'
  )

  return PoseLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  })
}

export function getPoseLandmarker(): Promise<PoseLandmarkerInstance> {
  if (!initPromise) {
    initPromise = doInit().catch((err) => {
      // Clear so the next caller triggers a fresh attempt
      initPromise = null
      throw err
    })
  }
  return initPromise
}

// Monotonic timestamp tracker for detectForVideo.
// MediaPipe requires strictly increasing timestamps across the lifetime of a
// PoseLandmarker instance. When the singleton is reused across pages (e.g.
// analyze -> compare), new processing sessions must continue from the last
// timestamp rather than restarting at 0.
let _highWaterMark = 0

/**
 * Record a timestamp that was sent to detectForVideo so later callers can
 * continue above it. UploadZone (analyze page) doesn't need this -it is
 * always the first consumer -but it's harmless if called.
 */
export function recordTimestamp(ts: number): void {
  if (ts > _highWaterMark) {
    _highWaterMark = ts
  }
}

/**
 * Return a monotonically-increasing timestamp suitable for detectForVideo.
 * `relativeMs` is the frame's position within the current video (e.g.
 * currentTime * 1000). The returned value is offset so it always exceeds
 * any timestamp previously sent to the landmarker.
 */
export function getMonotonicTimestamp(relativeMs: number): number {
  // Add 1ms gap so the very first frame (relativeMs === 0) is still strictly
  // greater than the previous high-water mark.
  const ts = _highWaterMark + 1 + relativeMs
  if (ts > _highWaterMark) {
    _highWaterMark = ts
  }
  return ts
}

