// Telemetry uses Railway's /classify-angle endpoint. Requires:
//   RAILWAY_SERVICE_URL=https://...railway.app
//   EXTRACT_API_KEY=<same key used for /extract>
// Missing either -> capture_quality_flag stays null on every row.
// This is intentional -- telemetry must never block coaching.
//
// Note: segment-analyze has no per-segment blob_url (segments are keypoint
// slices of the parent session). classifyAndTagCaptureQuality is still
// called for symmetry, but its blob_url=null no-op keeps the flag null here
// until we wire segment video URLs through.

import { NextRequest, NextResponse, after } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { supabase, supabaseAdmin } from '@/lib/supabase'
import { createClient } from '@/lib/supabase/server'
import { buildAngleSummary } from '@/lib/jointAngles'
import { getBiomechanicsReference } from '@/lib/biomechanics-reference'
import { classifyAndTagCaptureQuality } from '@/lib/captureQuality'
import {
  buildInferredTierCoachingBlock,
  buildTierCoachingBlock,
  getCoachingContext,
  isTierDowngrade,
  parseTierAssessmentTrailer,
} from '@/lib/profile'
import type { KeypointsJson } from '@/lib/supabase'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string; segmentId: string }> }
) {
  const { sessionId, segmentId } = await params

  // Body is optional now that pro comparisons are gone. Parse-and-discard so
  // old clients that post JSON don't get a 400.
  try { await request.json() } catch { /* ignore */ }

  // Tier-calibrate the segment prompt the same way as the whole-clip analyzer.
  // Three-way branch: profile -> tier block; skipped -> inferred-tier block;
  // neither -> generic calibration fallback.
  const authClient = await createClient()
  const { profile, skipped } = await getCoachingContext(authClient)
  const { data: userData } = await authClient.auth.getUser().catch(() => ({ data: { user: null } }))
  const userId = userData?.user?.id ?? null
  const tierBlock = profile
    ? buildTierCoachingBlock(profile)
    : skipped
      ? buildInferredTierCoachingBlock()
      : buildTierCoachingBlock(null)

  const { data: segment, error: segError } = await supabase
    .from('video_segments')
    .select('id, shot_type, keypoints_json, segment_index')
    .eq('id', segmentId)
    .eq('session_id', sessionId)
    .single()

  if (segError || !segment) {
    return NextResponse.json({ error: 'Segment not found' }, { status: 404 })
  }

  const segKeypoints = segment.keypoints_json as KeypointsJson | null
  if (!segKeypoints?.frames?.length) {
    return NextResponse.json({ error: 'Segment has no keypoints data' }, { status: 400 })
  }

  const userSummary = buildAngleSummary(segKeypoints.frames)
  const shotType = segment.shot_type ?? 'forehand'
  const soloRef = getBiomechanicsReference('all')

  const prompt = `You are a tennis coach. This is segment #${segment.segment_index} from a practice video, classified as a ${shotType}.

${tierBlock}

STRICT RULES:
- NEVER mention degrees, angles, or numbers. Describe in feel and body language.
- NEVER rate or score. Just give advice.
- NEVER use em dashes. Use commas or periods.

REFERENCE: ${soloRef}
USER SWING (segment #${segment.segment_index}): ${userSummary}

## What You're Doing Well
Two or three sentences.

## 3 Things to Work On
Three numbered coaching cues with drills.

## Your Practice Plan
Three one-sentence focus points.

Keep it under 350 words.`

  const coachedTier = profile?.skill_tier ?? null

  // Write the telemetry row BEFORE streaming starts so we can hand the frontend
  // the event id via X-Analysis-Event-Id for the thumbs-feedback wire-up.
  // Wrapped in try/catch: telemetry must NEVER fail the coaching stream.
  let eventId: string | null = null
  try {
    const { data: inserted, error: insertError } = await supabaseAdmin
      .from('analysis_events')
      .insert({
        user_id: userId,
        session_id: sessionId,
        segment_id: segmentId,
        self_reported_tier: profile?.skill_tier ?? null,
        was_skipped: skipped,
        handedness: profile?.dominant_hand ?? null,
        backhand_style: profile?.backhand_style ?? null,
        primary_goal: profile?.primary_goal ?? null,
        shot_type: shotType,
        blob_url: null,
        composite_metrics: { user_summary: userSummary, segment_index: segment.segment_index },
        llm_coached_tier: coachedTier,
        llm_assessed_tier: null,
        llm_tier_downgrade: false,
        capture_quality_flag: null,
      })
      .select('id')
      .single()
    if (insertError) {
      console.error('analysis_events insert failed:', insertError)
    } else {
      eventId = inserted?.id ?? null
    }
  } catch (err) {
    console.error('analysis_events insert threw:', err)
  }

  // Telemetry-only camera-angle classification. Segment rows currently have
  // no blob_url -> this is a no-op. Kept wired up so that when segment video
  // URLs land, the call already exists. Wrapped in after() for Vercel.
  after(() => classifyAndTagCaptureQuality(eventId, null))

  const messageStream = anthropic.messages.stream({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    system: `You are a veteran tennis coach. Talk like a real person. Short sentences. Casual tone. Use "you" and "your" constantly. No em dashes. No raw numbers.`,
    messages: [{ role: 'user', content: prompt }],
  })

  const encoder = new TextEncoder()
  const ERROR_PREFIX = '\n\n[ERROR] '

  // See analyze/route.ts for the tradeoff rationale — we buffer the stream so
  // we can strip the [TIER_ASSESSMENT: ...] trailer before the client sees it.
  const stream = new ReadableStream({
    async start(controller) {
      let buffered = ''
      let streamFailed = false
      let streamError: string | null = null
      try {
        for await (const chunk of messageStream) {
          if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
            buffered += chunk.delta.text
          }
        }
      } catch (err) {
        streamFailed = true
        streamError = err instanceof Error ? err.message : 'Analysis stream failed'
      }

      const { assessedTier, stripped } = parseTierAssessmentTrailer(buffered)
      controller.enqueue(encoder.encode(stripped))
      if (streamFailed && streamError) {
        controller.enqueue(encoder.encode(`${ERROR_PREFIX}${streamError}`))
      }
      controller.close()

      // Wrapped in after() so Vercel keeps the invocation live past
      // controller.close() — see analyze/route.ts for the full rationale.
      if (eventId) {
        const backfillCoached = coachedTier ?? (assessedTier && assessedTier !== 'unknown' ? assessedTier : null)
        const downgrade = isTierDowngrade(backfillCoached, assessedTier)
        after(async () => {
          const { error } = await supabaseAdmin
            .from('analysis_events')
            .update({
              llm_assessed_tier: assessedTier,
              llm_coached_tier: backfillCoached,
              llm_tier_downgrade: downgrade,
            })
            .eq('id', eventId)
          if (error) console.error('analysis_events update failed:', error)
        })
      }
    },
  })

  const headers: Record<string, string> = {
    'Content-Type': 'text/plain; charset=utf-8',
    'Cache-Control': 'no-cache',
    'X-Content-Type-Options': 'nosniff',
  }
  if (eventId) headers['X-Analysis-Event-Id'] = eventId

  return new NextResponse(stream, { headers })
}
