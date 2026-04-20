import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { supabase } from '@/lib/supabase'
import { buildAngleSummary, sampleKeyFrames } from '@/lib/jointAngles'
import { getBiomechanicsReference } from '@/lib/biomechanics-reference'
import type { KeypointsJson } from '@/lib/supabase'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

export async function POST(request: NextRequest) {
  let body
  try { body = await request.json() }
  catch { return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 }) }
  const {
    sessionId,
    proSwingId,
    keypointsJson: inlineKeypoints,
    compareKeypointsJson,
    userFocus,
    compareMode,
    baselineLabel,
  } = body
  const focus =
    typeof userFocus === 'string' && userFocus.trim() ? userFocus.trim() : null
  const isBaselineCompare = compareMode === 'baseline'
  const baselineTag =
    typeof baselineLabel === 'string' && baselineLabel.trim()
      ? baselineLabel.trim().slice(0, 80)
      : 'your best day'

  // Get pro swing data if proSwingId is provided
  let proSwing: { keypoints_json?: KeypointsJson; pros?: { name: string }; shot_type?: string } | null = null
  if (proSwingId) {
    const { data, error: proError } = await supabase
      .from('pro_swings')
      .select('*, pros(name)')
      .eq('id', proSwingId)
      .single()

    if (proError || !data) {
      return NextResponse.json({ error: 'Pro swing not found' }, { status: 404 })
    }
    proSwing = data
  }

  // Get user keypoints - from inline payload or session
  let userKeypoints: KeypointsJson | null = inlineKeypoints ?? null

  if (!userKeypoints && sessionId) {
    const { data: session, error: sessionError } = await supabase
      .from('user_sessions')
      .select('keypoints_json')
      .eq('id', sessionId)
      .single()

    if (sessionError) {
      return NextResponse.json({ error: 'Session not found' }, { status: 404 })
    }
    userKeypoints = session?.keypoints_json ?? null
  }

  if (!userKeypoints || !userKeypoints.frames?.length) {
    return NextResponse.json(
      { error: 'No keypoints data available' },
      { status: 400 }
    )
  }

  // Validate frame elements are non-null objects with joint_angles to prevent
  // a TypeError crash inside buildAngleSummary from malformed input
  const framesValid = userKeypoints.frames.every(
    (f) => f !== null && typeof f === 'object' && typeof f.joint_angles === 'object'
  )
  if (!framesValid) {
    return NextResponse.json({ error: 'Invalid keypoints format' }, { status: 400 })
  }

  // Build compact angle summary for the user
  const userSummary = buildAngleSummary(userKeypoints.frames)

  // Optional second-take keypoints for self-compare mode. Validated the same
  // way as user keypoints to avoid crashes inside buildAngleSummary.
  let compareSummary: string | null = null
  if (compareKeypointsJson && typeof compareKeypointsJson === 'object') {
    const cmp = compareKeypointsJson as KeypointsJson
    if (Array.isArray(cmp.frames) && cmp.frames.length > 0) {
      const cmpValid = cmp.frames.every(
        (f) => f !== null && typeof f === 'object' && typeof f.joint_angles === 'object',
      )
      if (!cmpValid) {
        return NextResponse.json(
          { error: 'Invalid compareKeypointsJson format' },
          { status: 400 },
        )
      }
      compareSummary = buildAngleSummary(cmp.frames)
    }
  }

  // Shared coaching rubric: read the user's actual data and calibrate the
  // severity of advice to how polished the technique already is. Without this,
  // the model tends to give the same "bend your knees more" notes to a pro and
  // a first-timer.
  const SKILL_CALIBRATION = `READ THE SKILL LEVEL FIRST:
Before giving any advice, look at the joint angles and phase timing. Judge how refined this swing already is.
- Polished / near-pro mechanics: give SUBTLE refinements, not rebuilds. Telling an advanced player to "bend your knees more" when their legs are already loaded is useless. Find the 5% that would make them even better.
- Solid intermediate: point to the 2 or 3 biggest gaps and give practical drills.
- Still developing: focus on foundations, and pick the ONE thing that unlocks everything else.
Match the advice to what the swing actually needs. Never default to generic cues.`

  const focusBlock = focus
    ? `\nTHE PLAYER SPECIFICALLY WANTS FEEDBACK ON: "${focus}"\nWeave a direct answer to this into your response. You can still cover the essentials, but this is their priority.\n`
    : ''

  let prompt: string

  if (proSwing) {
    // Pro comparison mode
    const proKeypoints: KeypointsJson | null = proSwing.keypoints_json ?? null
    const proName = proSwing.pros?.name ?? 'Pro player'
    const shotType = proSwing.shot_type ?? 'tennis'

    if (!proKeypoints?.frames?.length) {
      return NextResponse.json(
        { error: 'Pro swing has no keypoints data. It may not have been processed yet.' },
        { status: 400 }
      )
    }

    const proSummary = buildAngleSummary(proKeypoints.frames)

    const strokeRef = getBiomechanicsReference(
      ['forehand', 'backhand', 'serve'].includes(shotType) ? shotType as 'forehand' | 'backhand' | 'serve' : 'all'
    )

    prompt = `You are a tennis coach helping a player model their ${shotType} after ${proName}'s. You just watched their swing side by side with ${proName}'s video. Your whole job is to help them move more like ${proName}.

${SKILL_CALIBRATION}
${focusBlock}
STRICT RULES:
- NEVER mention degrees, angles, or numbers of any kind. Describe everything in feel and body language.
- NEVER rate or score the player. No X/100, no percentages, no grades. Just give advice.
- NEVER use em dashes. Use commas or periods.
- Talk like a real person. Short sentences. "You" and "your" constantly.
- Reference ${proName} by name frequently. Compare what the player does to what ${proName} does. "Watch how ${proName} sits into his legs here" or "See how ${proName}'s hips fire before the arm even starts moving."
- If the player's swing is already close to ${proName}'s, SAY THAT and give small tweaks. Don't invent problems.

Use the data below to understand what's happening, but ONLY talk in coaching language. The player never sees these numbers.

REFERENCE: ${strokeRef}
USER SWING DATA: ${userSummary}
${proName.toUpperCase()} SWING DATA: ${proSummary}

Respond in this format:

## What You're Doing Well
Two or three sentences. Be specific. If something in their swing already looks like ${proName}'s, say so.

## How to Get Closer to ${proName}'s ${shotType}

**1. [Short coaching cue]**
Describe what ${proName} does differently and why it matters. Then give one drill or feel-based tip to close the gap. Use phrases like "load into your legs the way ${proName} does", "let the racket lag behind your hand", "turn your hips before your shoulders".

**2. [Short coaching cue]**
Same approach. What ${proName} does, why it works, one thing to try on the next ball.

**3. [Short coaching cue]**
Same approach. What ${proName} does, why it works, one thing to try.

## Your Practice Plan
Three things to focus on next time you hit, all aimed at making your swing look and feel more like ${proName}'s. One sentence each.

Keep it under 350 words. Sound like a coach who's watched a lot of ${proName} and knows exactly what makes that swing tick.`
  } else if (compareSummary) {
    // Self-compare mode: same player, two takes. Coach for CONSISTENCY —
    // spot what's drifting between the two swings rather than rebuilding either.
    //
    // Baseline variant (compareMode === 'baseline'): same data plumbing, but
    // the framing shifts from "two takes side by side" to "best day vs today".
    // Everything else (rules, voice, biomechanics reference) is identical.
    const soloRef = getBiomechanicsReference('all')

    const framingParagraph = isBaselineCompare
      ? `You are a tennis coach helping a player compare today's swing against their best-day baseline ("${baselineTag}"). Your job is progress tracking — show them what's held up since that peak, what's drifted, and how to lock the good stuff back in.`
      : `You are a tennis coach watching the same player hit two different swings back to back. Your job is consistency — help them spot what's staying the same and what's drifting between takes.`

    const anchorRule = isBaselineCompare
      ? `- Do NOT suggest rebuilds. The baseline IS the anchor — anywhere today's swing drifts from "${baselineTag}", coach them back toward how they moved on their best day.`
      : `- Do NOT suggest rebuilds. Both swings come from the same player, so pick the cleaner take as the anchor and talk about matching to it.`

    const specificExample = isBaselineCompare
      ? `- Be SPECIFIC about differences: "your hips turned further on your best day but today your arm got ahead of them", not "your swing looks different".`
      : `- Be SPECIFIC about differences: "your hips turned further in take 1 but your arm lagged behind in take 2", not "your swing was inconsistent".`

    const leftLabel = isBaselineCompare ? `BEST-DAY BASELINE ("${baselineTag}") DATA` : 'TAKE 1 DATA'
    const rightLabel = isBaselineCompare ? 'TODAY DATA' : 'TAKE 2 DATA'

    const heldUpHeading = isBaselineCompare ? "What's Held Up From Your Best Day" : "What's Consistent"
    const driftingHeading = isBaselineCompare ? "What's Drifted" : "What's Drifting"
    const lockItHeading = isBaselineCompare ? 'Lock It Back In' : 'Lock It In'

    const heldUpBody = isBaselineCompare
      ? `Two or three sentences on what today's swing kept from the best-day baseline. Reinforce what's still working.`
      : `Two or three sentences on what they're doing the same in both takes. Reinforce what's working.`

    const driftingItemBody = isBaselineCompare
      ? `What changed from the best day to today, which version looked cleaner, and one feel-based cue to get back to best-day quality.`
      : `What changed between the two takes, which take was cleaner, and one feel-based cue to anchor the next swing.`

    const lockItBody = isBaselineCompare
      ? `Two short sentences on what to groove next session so today's swing matches your best day again.`
      : `Two short sentences on what to groove next session so these swings match.`

    prompt = `${framingParagraph}
${focusBlock}
STRICT RULES:
- NEVER mention degrees, angles, or numbers of any kind. Describe everything in feel and body language.
- NEVER rate or score the player. No X/100, no percentages, no grades.
- NEVER use em dashes. Use commas or periods.
${anchorRule}
${specificExample}

Use the data below to understand what's happening, but ONLY talk in coaching language.

REFERENCE: ${soloRef}
${leftLabel}: ${userSummary}
${rightLabel}: ${compareSummary}

Respond in this format:

## ${heldUpHeading}
${heldUpBody}

## ${driftingHeading}

**1. [specific element]**
${driftingItemBody}

**2. [specific element]**
Same structure.

**3. [specific element]**
Same structure.

## ${lockItHeading}
${lockItBody}

Keep it under 350 words. Sound like a coach helping them tighten up.`
  } else {
    // Solo mode: single clip, general coaching without any reference.
    const soloRef = getBiomechanicsReference('all')

    prompt = `You are a tennis coach talking to a player right after watching their swing on video. Be encouraging and practical.

${SKILL_CALIBRATION}
${focusBlock}
STRICT RULES:
- NEVER mention degrees, angles, or numbers of any kind. Not even once. Describe everything in feel and body language.
- NEVER rate or score the player (no X/100, no percentages, no grades). Just give advice.
- NEVER use em dashes. Use commas or periods.
- Talk like a real person. Short sentences. "You" and "your" constantly.
- If the swing is already clean, SAY THAT and give fine-tuning cues. Don't fabricate problems.

Use the data below to understand what's happening, but ONLY talk in coaching language. The player never sees these numbers.

REFERENCE: ${soloRef}
USER SWING DATA: ${userSummary}

Respond in this format:

## What You're Doing Well
Two or three sentences about what's genuinely working in their swing. Be specific, not generic.

## 3 Things to Work On

**1. [Short coaching cue]**
What you see, why it matters for their game, how it compares to solid technique. Give one drill or feel-based tip they can try on the next ball. Use phrases like "load into your legs", "let the racket drop behind you", "turn your hips before your shoulders".

**2. [Short coaching cue]**
Same approach. Observation, why it matters, one actionable tip.

**3. [Short coaching cue]**
Same approach. Observation, why it matters, one actionable tip.

## Your Practice Plan
Three specific things to focus on in their next hitting session. Make each one a single sentence they can remember on court.

Keep it under 350 words. Sound like a coach who believes in this player.`
  }

  const systemPrompt = `You are a veteran tennis coach who has been on court for 30 years. You talk like a real person having a conversation with a player between points.

VOICE RULES (follow these strictly):
- Write like you TALK. Short sentences. Casual tone. "Your hips are way ahead of your arm here" not "The hip rotation precedes the arm extension."
- NEVER use em dashes. Use periods or commas instead.
- NEVER list raw degree numbers on their own. Wrong: "elbow: 170°, ideal: 110°". Right: "your arm is almost locked straight when you want a nice relaxed bend."
- You CAN mention a number to back up a point, but always in a natural sentence: "you're only getting about 20 degrees of hip turn when the pros are closer to 45."
- Use coaching cues a player can FEEL: "load your weight into your back leg", "let the racket drop behind you like a pendulum", "snap your hips like you're throwing a punch."
- Keep it practical. Every piece of advice should be something they can try on the very next ball.
- Sound encouraging, not critical. You're helping, not grading.
- No bullet points with just numbers. No tables. No clinical language.
- Write "you" and "your" constantly. Talk TO the player.`

  const messageStream = anthropic.messages.stream({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    system: systemPrompt,
    messages: [{ role: 'user', content: prompt }],
  })

  const encoder = new TextEncoder()
  const ERROR_PREFIX = '\n\n[ERROR] '
  const stream = new ReadableStream({
    async start(controller) {
      try {
        for await (const chunk of messageStream) {
          if (
            chunk.type === 'content_block_delta' &&
            chunk.delta.type === 'text_delta'
          ) {
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
