// Telemetry helper: calls Railway's /classify-angle to tag the camera angle
// of an analyzed clip, then UPDATEs the matching analysis_events row with the
// returned capture_quality_flag. Strictly fire-and-forget -- any failure
// (Railway down, classify timeout, DB update error) is logged and swallowed
// so it never blocks the coaching stream.
//
// Requires:
//   RAILWAY_SERVICE_URL=https://...railway.app
//   EXTRACT_API_KEY=<same key used for /extract>
// Missing either -> we skip the call entirely and the row keeps capture_quality_flag=null.
// This is intentional: telemetry must never block coaching.

import { supabaseAdmin } from '@/lib/supabase'

const RAILWAY_SERVICE_URL = process.env.RAILWAY_SERVICE_URL
const EXTRACT_API_KEY = process.env.EXTRACT_API_KEY
export const RAILWAY_CONFIGURED = !!RAILWAY_SERVICE_URL && !!EXTRACT_API_KEY

// Module-level guard so we warn once per process, not per request. Next.js
// reuses the module across requests in production, so this stays truthy
// between invocations and the log doesn't spam.
let _warnedMissingEnv = false

// The DB enum accepts these five string values. 'yellow_oblique' is reserved
// for a future classifier that can actually distinguish oblique angles --
// the current camera_classifier.py heuristics can't, so we never emit it.
const VALID_FLAGS = new Set([
  'green_side',
  'yellow_oblique',
  'red_front_or_back',
  'unknown',
])

interface ClassifyAngleResponse {
  capture_quality_flag?: unknown
  raw_label?: unknown
  samples_considered?: unknown
  error?: unknown
}

/**
 * Fire a background promise to classify the clip and UPDATE the analysis_events
 * row. Never awaited by the caller. Returns immediately so the analyze route
 * can close its stream without waiting on Railway.
 *
 * No-op (with a single process-lifetime warning) when env vars are missing, or
 * when eventId / blobUrl is missing. Any thrown error inside the async chain
 * is caught and logged.
 */
export function classifyAndTagCaptureQuality(
  eventId: string | null,
  blobUrl: string | null,
): void {
  if (!eventId || !blobUrl) return

  if (!RAILWAY_CONFIGURED) {
    if (!_warnedMissingEnv) {
      _warnedMissingEnv = true
      console.warn(
        '[capture-quality] RAILWAY_SERVICE_URL or EXTRACT_API_KEY unset — capture_quality_flag will stay null on all analysis_events rows.',
      )
    }
    return
  }

  // Kick off the promise and intentionally do not await. The outer try/catch
  // catches synchronous throws from fetch init; the .catch on the chain
  // catches async rejections. Either way, we swallow-and-log.
  try {
    void _classifyAndTag(eventId, blobUrl).catch((err) => {
      console.error('[capture-quality] background task failed:', err)
    })
  } catch (err) {
    console.error('[capture-quality] failed to schedule:', err)
  }
}

async function _classifyAndTag(eventId: string, blobUrl: string): Promise<void> {
  const controller = new AbortController()
  // Hard cap: if Railway is slow, we'd rather lose the telemetry than leak
  // a background fetch that outlives the serverless invocation.
  const timer = setTimeout(() => controller.abort(), 60_000)

  let flag: string = 'unknown'
  try {
    const resp = await fetch(`${RAILWAY_SERVICE_URL}/classify-angle`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${EXTRACT_API_KEY}`,
      },
      body: JSON.stringify({ video_url: blobUrl }),
      signal: controller.signal,
    })

    if (!resp.ok) {
      console.error(
        `[capture-quality] Railway returned ${resp.status} for event ${eventId}`,
      )
      return
    }

    const data = (await resp.json().catch(() => null)) as ClassifyAngleResponse | null
    const maybeFlag = typeof data?.capture_quality_flag === 'string' ? data.capture_quality_flag : null
    if (maybeFlag && VALID_FLAGS.has(maybeFlag)) {
      flag = maybeFlag
    } else {
      console.error(
        `[capture-quality] invalid flag from Railway for event ${eventId}:`,
        maybeFlag,
      )
      return
    }
  } catch (err) {
    console.error(`[capture-quality] fetch failed for event ${eventId}:`, err)
    return
  } finally {
    clearTimeout(timer)
  }

  const { error } = await supabaseAdmin
    .from('analysis_events')
    .update({ capture_quality_flag: flag })
    .eq('id', eventId)
  if (error) {
    console.error(
      `[capture-quality] DB update failed for event ${eventId}:`,
      error,
    )
  }
}
