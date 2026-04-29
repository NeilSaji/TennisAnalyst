import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { supabase } from '@/lib/supabase'
import { buildAngleSummary } from '@/lib/jointAngles'
import type { PoseFrame } from '@/lib/supabase'

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })

const SYSTEM_PROMPT = `You are a tennis coach reviewing several extracted shots from one player's session.
Pick the SINGLE best shot from the list using criteria like:
 - cleanest contact-extension elbow angle for the shot type
 - stable spine and trunk rotation through contact
 - solid knee bend without collapse
 - highest pose-detection confidence (low confidence = unreliable evidence)

Reply ONLY with valid JSON, no code fences, no prose around it:
{"bestIndex": <0-based index, integer>, "reasoning": "<2-3 short sentences explaining why this shot stood out>"}

Keep reasoning tight and concrete — reference specific angles or phases when you can. No headers, no bullets.`

export async function POST(
  _req: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> },
) {
  const { sessionId } = await params

  const { data, error } = await supabase
    .from('video_segments')
    .select('segment_index, shot_type, confidence, keypoints_json')
    .eq('session_id', sessionId)
    .order('segment_index', { ascending: true })

  if (error || !data || data.length === 0) {
    return NextResponse.json({ error: 'No segments found for this session' }, { status: 404 })
  }

  if (data.length < 2) {
    return NextResponse.json({
      bestIndex: 0,
      reasoning: 'Only one shot detected in this session.',
    })
  }

  const blocks = data
    .map((seg) => {
      const frames: PoseFrame[] = seg.keypoints_json?.frames ?? []
      const summary = frames.length > 0 ? buildAngleSummary(frames) : 'no pose frames captured'
      const confPct = (seg.confidence * 100).toFixed(0)
      return `Shot ${seg.segment_index + 1} (${seg.shot_type}, detection confidence ${confPct}%):\n${summary}`
    })
    .join('\n\n')

  let resp
  try {
    resp = await anthropic.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 300,
      system: [{ type: 'text', text: SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } }],
      messages: [{ role: 'user', content: blocks }],
    })
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : 'Anthropic call failed' },
      { status: 500 },
    )
  }

  const text = resp.content
    .filter((b): b is Anthropic.TextBlock => b.type === 'text')
    .map((b) => b.text)
    .join('\n')
    .trim()

  const cleaned = text
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim()

  let parsed: { bestIndex: number; reasoning: string }
  try {
    parsed = JSON.parse(cleaned)
  } catch {
    return NextResponse.json({
      bestIndex: 0,
      reasoning: text || 'Could not parse the model response.',
    })
  }

  const idx = Number(parsed.bestIndex)
  const safeIdx = Number.isInteger(idx) && idx >= 0 && idx < data.length ? idx : 0

  return NextResponse.json({
    bestIndex: safeIdx,
    reasoning: typeof parsed.reasoning === 'string' ? parsed.reasoning : '',
  })
}
