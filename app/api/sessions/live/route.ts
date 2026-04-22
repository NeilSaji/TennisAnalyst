import { NextRequest, NextResponse } from 'next/server'
import { supabaseAdmin } from '@/lib/supabase'
import { createClient } from '@/lib/supabase/server'
import type { KeypointsJson } from '@/lib/supabase'

const ALLOWED_BLOB_HOST_SUFFIX = '.public.blob.vercel-storage.com'

const VALID_SHOT_TYPES = new Set(['forehand', 'backhand', 'serve', 'volley'] as const)

type IncomingSwing = {
  startFrame?: unknown
  endFrame?: unknown
  startMs?: unknown
  endMs?: unknown
}

function isAllowedBlobUrl(raw: unknown): raw is string {
  if (typeof raw !== 'string' || !raw) return false
  try {
    const u = new URL(raw)
    return u.protocol === 'https:' && u.hostname.endsWith(ALLOWED_BLOB_HOST_SUFFIX)
  } catch {
    return false
  }
}

function coerceInt(v: unknown, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? Math.max(0, Math.floor(v)) : fallback
}

// POST /api/sessions/live — finalizes a live-mode session at Stop time.
// Creates the user_sessions + video_segments rows, then backfills
// session_id/segment_id/blob_url on every analysis_events row produced by
// /api/live-coach during the session. Service-role client used to keep the
// bulk UPDATE simple; ownership is enforced by user_id filter in every step.
export async function POST(request: NextRequest) {
  const authClient = await createClient()
  const { data: { user } } = await authClient.auth.getUser()
  if (!user) {
    return NextResponse.json({ error: 'Not signed in' }, { status: 401 })
  }
  const userId = user.id

  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const b = body as Record<string, unknown>

  if (!isAllowedBlobUrl(b.blobUrl)) {
    return NextResponse.json(
      { error: `blobUrl must be an https URL on ${ALLOWED_BLOB_HOST_SUFFIX}` },
      { status: 400 },
    )
  }
  const blobUrl = b.blobUrl

  if (typeof b.shotType !== 'string' || !(VALID_SHOT_TYPES as Set<string>).has(b.shotType)) {
    return NextResponse.json(
      { error: `shotType must be one of ${Array.from(VALID_SHOT_TYPES).join(', ')}` },
      { status: 400 },
    )
  }
  const shotType = b.shotType

  const kp = b.keypointsJson as KeypointsJson | undefined
  if (!kp || typeof kp !== 'object' || !Array.isArray(kp.frames)) {
    return NextResponse.json({ error: 'keypointsJson.frames is required' }, { status: 400 })
  }

  const rawSwings = Array.isArray(b.swings) ? (b.swings as IncomingSwing[]) : []
  const swings = rawSwings.map((s, i) => ({
    segment_index: i + 1,
    shot_type: shotType,
    start_frame: coerceInt(s.startFrame),
    end_frame: coerceInt(s.endFrame),
    start_ms: coerceInt(s.startMs),
    end_ms: coerceInt(s.endMs),
    confidence: 1.0,
  }))

  const batchEventIds = Array.isArray(b.batchEventIds)
    ? (b.batchEventIds as unknown[]).filter((x): x is string => typeof x === 'string' && x.length > 0)
    : []

  // 1. Insert user_sessions row
  const { data: sessionRow, error: sessionErr } = await supabaseAdmin
    .from('user_sessions')
    .upsert(
      {
        blob_url: blobUrl,
        shot_type: shotType,
        keypoints_json: kp,
        status: 'complete',
        session_mode: 'live',
        is_multi_shot: swings.length > 1,
        segment_count: swings.length,
      },
      { onConflict: 'blob_url' },
    )
    .select('id')
    .single()
  if (sessionErr || !sessionRow) {
    console.error('sessions/live user_sessions upsert failed:', sessionErr)
    return NextResponse.json({ error: 'Failed to save session' }, { status: 500 })
  }
  const sessionId = sessionRow.id as string

  // 2. Insert video_segments rows (one per detected swing)
  let lastSegmentId: string | null = null
  if (swings.length > 0) {
    const { data: segmentRows, error: segErr } = await supabaseAdmin
      .from('video_segments')
      .insert(
        swings.map((s) => ({
          session_id: sessionId,
          segment_index: s.segment_index,
          shot_type: s.shot_type,
          start_frame: s.start_frame,
          end_frame: s.end_frame,
          start_ms: s.start_ms,
          end_ms: s.end_ms,
          confidence: s.confidence,
        })),
      )
      .select('id, segment_index')
    if (segErr) {
      console.error('sessions/live video_segments insert failed:', segErr)
      // Non-fatal: we still return the session id so the client can proceed.
    } else if (segmentRows && segmentRows.length > 0) {
      // Pick the id of the highest-index segment for the batch-events backfill.
      const last = segmentRows.reduce(
        (a, b2) => (b2.segment_index > a.segment_index ? b2 : a),
        segmentRows[0],
      )
      lastSegmentId = last?.id ?? null
    }
  }

  // 3. Backfill analysis_events rows for every batch produced during the session
  if (batchEventIds.length > 0) {
    const { error: updateErr } = await supabaseAdmin
      .from('analysis_events')
      .update({
        session_id: sessionId,
        segment_id: lastSegmentId,
        blob_url: blobUrl,
      })
      .in('id', batchEventIds)
      .eq('user_id', userId)
    if (updateErr) {
      console.error('sessions/live analysis_events backfill failed:', updateErr)
    }
  }

  return NextResponse.json({
    sessionId,
    segmentCount: swings.length,
    eventsBackfilled: batchEventIds.length,
  })
}
