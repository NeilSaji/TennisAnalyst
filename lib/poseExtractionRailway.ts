'use client'

import type { PoseFrame } from '@/lib/supabase'
import type { ExtractResult, ExtractorBackend } from '@/lib/poseExtraction'

// Poll interval (ms) for the session-status check. Extraction on Railway
// runs server-side and typically finishes in 60-90s on a 1-2 minute clip.
// 3s polls mean ~20-30 requests per extraction — negligible load.
const POLL_INTERVAL_MS = 3000

// Hard client-side timeout (ms). If Railway doesn't report 'complete'
// within this window, we give up and the caller falls back to browser
// MediaPipe. Tuned against IMG_1097.mov: 114s clip took ~75s end-to-end
// on Railway's default CPU tier, so 180s leaves headroom for longer
// clips or warm-up.
const EXTRACTION_TIMEOUT_MS = 180_000

class RailwayExtractError extends Error {
  constructor(message: string, public reason: 'not-configured' | 'queue-failed' | 'timeout' | 'error-status' | 'aborted') {
    super(message)
    this.name = 'RailwayExtractError'
  }
}

export type RailwayExtractOptions = {
  // sessionId of a ROW ALREADY CREATED by /api/sessions with
  // status='pending'. Railway writes keypoints_json back to this row
  // when it finishes.
  sessionId: string
  blobUrl: string
  // Called with 0..100. Since Railway's /extract is fire-and-poll,
  // we can't emit real progress — this callback just advances through
  // best-guess checkpoints (25, 50, 75) over time. The UI shouldn't
  // make promises based on intermediate values.
  onProgress?: (pct: number) => void
  abortSignal?: AbortSignal
}

function buildExtractResultFromKeypoints(
  keypointsJson: {
    frames: PoseFrame[]
    fps_sampled?: number
    pose_backend?: string
  },
  blobUrl: string,
): ExtractResult {
  // Map the Railway-side backend name to the UI label. Anything we
  // don't recognize falls back to mediapipe-railway (the historical
  // default) so old keypoints_json rows written before this field
  // existed don't break.
  let extractorBackend: ExtractorBackend = 'mediapipe-railway'
  if (keypointsJson.pose_backend === 'rtmpose') {
    extractorBackend = 'rtmpose-railway'
  }
  return {
    frames: keypointsJson.frames,
    fps: keypointsJson.fps_sampled ?? 30,
    // The remote extraction doesn't create a local object URL. The
    // caller already has the File locally and will create its own
    // object URL for playback — we return the blobUrl as a fallback
    // so downstream code that expects a non-null URL keeps working.
    objectUrl: blobUrl,
    extractorBackend,
  }
}

/**
 * Ask Railway to run pose extraction on an already-uploaded blob, then
 * poll Supabase (via /api/sessions/[id]) until keypoints land. Throws
 * RailwayExtractError on any failure — caller is expected to catch and
 * fall back to in-browser extraction.
 *
 * Why this design (not a fetch + synchronous body):
 *   • Railway's /extract endpoint is intentionally async — it queues a
 *     BackgroundTask and returns {status:'queued'} immediately so the
 *     HTTP connection doesn't block for 60-90s (and hit any intermediate
 *     timeouts). The authoritative result is written to Supabase, which
 *     we poll.
 */
export async function extractPoseViaRailway(
  opts: RailwayExtractOptions,
): Promise<ExtractResult> {
  const { sessionId, blobUrl, onProgress, abortSignal } = opts

  if (abortSignal?.aborted) {
    throw new RailwayExtractError('aborted before start', 'aborted')
  }

  onProgress?.(10)

  // Kick off the job.
  const queueRes = await fetch('/api/extract', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId, blobUrl }),
    signal: abortSignal,
  })

  if (queueRes.status === 503) {
    throw new RailwayExtractError(
      'Railway not configured',
      'not-configured',
    )
  }

  if (!queueRes.ok) {
    throw new RailwayExtractError(
      `Railway queue rejected: HTTP ${queueRes.status}`,
      'queue-failed',
    )
  }

  onProgress?.(25)

  // Poll until done or timeout.
  const deadline = Date.now() + EXTRACTION_TIMEOUT_MS
  // Advance progress from 25 → 90 based on elapsed / total budget so the
  // UI has SOMETHING to move. Real completion jumps to 100 at the end.
  const pollStart = Date.now()

  while (Date.now() < deadline) {
    if (abortSignal?.aborted) {
      throw new RailwayExtractError('aborted during poll', 'aborted')
    }

    await new Promise<void>((resolve, reject) => {
      const t = setTimeout(resolve, POLL_INTERVAL_MS)
      abortSignal?.addEventListener(
        'abort',
        () => {
          clearTimeout(t)
          reject(new RailwayExtractError('aborted during poll sleep', 'aborted'))
        },
        { once: true },
      )
    })

    const pollRes = await fetch(
      `/api/sessions/${encodeURIComponent(sessionId)}?include=keypoints`,
      { signal: abortSignal },
    )

    // A transient 4xx/5xx here shouldn't kill the whole extract — keep
    // polling until deadline. Only treat an explicit 'error' status or
    // unrecoverable 404 as terminal.
    if (pollRes.status === 404) {
      throw new RailwayExtractError('session vanished mid-extract', 'error-status')
    }
    if (!pollRes.ok) continue

    const body = (await pollRes.json()) as {
      status?: string
      keypoints_json?: {
        frames: PoseFrame[]
        fps_sampled?: number
        pose_backend?: string
      }
      error_message?: string
    }

    if (body.status === 'complete' && body.keypoints_json) {
      onProgress?.(100)
      return buildExtractResultFromKeypoints(body.keypoints_json, blobUrl)
    }

    if (body.status === 'error') {
      throw new RailwayExtractError(
        body.error_message ?? 'Railway reported error',
        'error-status',
      )
    }

    const pct = 25 + Math.min(65, Math.floor(((Date.now() - pollStart) / EXTRACTION_TIMEOUT_MS) * 65))
    onProgress?.(pct)
  }

  throw new RailwayExtractError('timeout waiting for Railway', 'timeout')
}

export { RailwayExtractError }
