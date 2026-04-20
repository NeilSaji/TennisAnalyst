import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { supabase } from '@/lib/supabase'
import { createClient } from '@/lib/supabase/server'
import { buildAngleSummary } from '@/lib/jointAngles'
import { getBiomechanicsReference } from '@/lib/biomechanics-reference'
import {
  buildInferredTierCoachingBlock,
  buildTierCoachingBlock,
  getCoachingContext,
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

  const messageStream = anthropic.messages.stream({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    system: `You are a veteran tennis coach. Talk like a real person. Short sentences. Casual tone. Use "you" and "your" constantly. No em dashes. No raw numbers.`,
    messages: [{ role: 'user', content: prompt }],
  })

  const encoder = new TextEncoder()
  const ERROR_PREFIX = '\n\n[ERROR] '
  const stream = new ReadableStream({
    async start(controller) {
      try {
        for await (const chunk of messageStream) {
          if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
            controller.enqueue(encoder.encode(chunk.delta.text))
          }
        }
        controller.close()
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Analysis stream failed'
        controller.enqueue(encoder.encode(`${ERROR_PREFIX}${msg}`))
        controller.close()
      }
    },
  })

  return new NextResponse(stream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-cache',
      'X-Content-Type-Options': 'nosniff',
    },
  })
}
