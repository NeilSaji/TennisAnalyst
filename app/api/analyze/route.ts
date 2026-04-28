// Telemetry uses Railway's /classify-angle endpoint. Requires:
//   RAILWAY_SERVICE_URL=https://...railway.app
//   EXTRACT_API_KEY=<same key used for /extract>
// Missing either -> capture_quality_flag stays null on every row.
// This is intentional -- telemetry must never block coaching.

import { NextRequest, NextResponse } from 'next/server'
import { after } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { supabase, supabaseAdmin } from '@/lib/supabase'
import { createClient } from '@/lib/supabase/server'
import { buildAngleSummary, detectSwings } from '@/lib/jointAngles'
import { getBiomechanicsReference } from '@/lib/biomechanics-reference'
import { SHOT_TYPE_CONFIGS } from '@/lib/shotTypeConfig'
import { classifyAndTagCaptureQuality } from '@/lib/captureQuality'
import {
  buildCoachingToolInput,
  buildInferredTierCoachingBlock,
  buildTierCoachingBlock,
  COACHING_TOOL_NAME,
  COACHING_TOOL_SCHEMAS,
  DEFAULT_MAX_TOKENS,
  extractResponseMetrics,
  getCoachingContext,
  isTierDowngrade,
  parseTierAssessmentTrailer,
  renderCoachingToolInputToMarkdown,
  TIER_MAX_TOKENS,
  type CoachingToolInput,
  type SkillTier,
} from '@/lib/profile'
import { sanitizePromptInput } from '@/lib/sanitize'
import type { KeypointsJson } from '@/lib/supabase'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

export async function POST(request: NextRequest) {
  // Resolve profile + skipped state in a single getUser() round trip so every
  // prompt branch can tier-calibrate. Anonymous / legacy users return null
  // profile and skipped=false, falling through to the generic calibration
  // block. Users who explicitly skipped onboarding get the inferred-tier block
  // instead, so we don't ignore their "I don't want to self-report" signal.
  const authClient = await createClient()
  const { profile, skipped } = await getCoachingContext(authClient)
  const { data: userData } = await authClient.auth.getUser().catch(() => ({ data: { user: null } }))
  const userId = userData?.user?.id ?? null

  let body
  try { body = await request.json() }
  catch { return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 }) }
  const {
    sessionId,
    keypointsJson: inlineKeypoints,
    compareKeypointsJson,
    userFocus,
    compareMode,
    baselineLabel,
    shotType: bodyShotType,
    blobUrl: bodyBlobUrl,
  } = body

  const focus = sanitizePromptInput(userFocus, 240)
  const isBaselineCompare = compareMode === 'baseline'
  const baselineTag = sanitizePromptInput(baselineLabel, 80) ?? 'your best day'

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

  // Resolve shot type once, early. Used by both the prompt (shape the coaching
  // for forehand vs. backhand vs. serve) and the telemetry row (kept as-is
  // downstream). Priority: request body > session metadata > null.
  let resolvedShotType: string | null = typeof bodyShotType === 'string' ? bodyShotType : null
  let resolvedBlobUrl: string | null = typeof bodyBlobUrl === 'string' ? bodyBlobUrl : null
  if ((!resolvedShotType || !resolvedBlobUrl) && sessionId) {
    try {
      const { data: sessionMeta } = await supabase
        .from('user_sessions')
        .select('shot_type, blob_url')
        .eq('id', sessionId)
        .single()
      if (sessionMeta) {
        resolvedShotType = resolvedShotType ?? sessionMeta.shot_type ?? null
        resolvedBlobUrl = resolvedBlobUrl ?? sessionMeta.blob_url ?? null
      }
    } catch (err) {
      console.error('analyze: shot_type/blob_url lookup failed:', err)
    }
  }

  // Run the shot-type-specific mistakeChecks from SHOT_TYPE_CONFIGS against
  // the peak-activity frame of the primary swing. Gives the LLM a curated
  // list of issues to prioritize instead of fabricating from raw angles.
  // Only runs when we actually know the shot type — SHOT_TYPE_CONFIGS
  // normalizes unknown to forehand, and running forehand checks on a serve
  // would produce bogus detected-issue cites. Capped at 2 so the model can
  // still speak freely.
  let detectedMistakes: string[] = []
  if (
    resolvedShotType &&
    SHOT_TYPE_CONFIGS[resolvedShotType] &&
    userKeypoints.frames.length > 0
  ) {
    const swings = detectSwings(userKeypoints.frames)
    const primary = swings[0]
    if (primary) {
      // peakFrame is an index into allFrames; primary.frames is a slice of
      // those (from startFrame to endFrame). Translate so we index the slice.
      const peakIdxInSlice = primary.peakFrame - primary.startFrame
      const peakFrame = primary.frames[Math.max(0, Math.min(primary.frames.length - 1, peakIdxInSlice))]
      if (peakFrame && peakFrame.joint_angles) {
        const checks = SHOT_TYPE_CONFIGS[resolvedShotType].mistakeChecks
        for (const check of checks) {
          try {
            if (check.detect(peakFrame.joint_angles)) {
              detectedMistakes.push(`${check.label}: ${check.tip}`)
              if (detectedMistakes.length >= 2) break
            }
          } catch {
            // Defensive: a mistakeCheck that throws on unexpected input
            // shouldn't block coaching. Silent skip is fine.
          }
        }
      }
    }
  }

  // Map to the reference library's vocabulary. Only forehand/backhand/serve
  // have dedicated references; volley/slice/unknown fall back to 'all'.
  const refShotType: 'forehand' | 'backhand' | 'serve' | 'all' =
    resolvedShotType === 'forehand' || resolvedShotType === 'backhand' || resolvedShotType === 'serve'
      ? resolvedShotType
      : 'all'

  // Shared prompt blocks injected into BOTH the solo and compare branches.
  // When shot type IS specified we lock the prompt to that shot. When it
  // is NOT specified we just stay silent about it. The previous "name
  // which shot you think this is" instruction made the model open every
  // response with "Looks like a forehand to me", which read as low-confidence
  // and was a user-flagged regression. Better to skip the meta-statement
  // entirely and let the coaching focus on what to actually fix.
  const shotIntroBlock = resolvedShotType
    ? `\nSHOT TYPE: You are analyzing a ${resolvedShotType} swing. Tailor every cue and reference angle to this specific shot. Do NOT generalize across shot types. Do NOT open by naming the shot type.\n`
    : ''
  const detectedIssuesBlock = detectedMistakes.length > 0
    ? `\nAUTO-DETECTED ISSUES (system-flagged for this swing, weave coaching around these, do not just list them):\n${detectedMistakes.map((m, i) => `${i + 1}. ${m}`).join('\n')}\n`
    : ''

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

  // Tier-aware coaching rubric. Three-way branch:
  //   profile            -> tier rules + handedness + goal weighting
  //   skipped && !profile -> infer-tier block (LLM names its guess inline)
  //   neither            -> generic fallback calibration
  const tierBlock = profile
    ? buildTierCoachingBlock(profile)
    : skipped
      ? buildInferredTierCoachingBlock()
      : buildTierCoachingBlock(null)

  const focusBlock = focus
    ? `\nTHE PLAYER ASKED: "${focus}"\nThis is the QUESTION you must answer. Open your response by directly addressing this exact question in 1 to 2 sentences before anything else. Do not generalize. Do not pivot to unrelated cues. After answering their question, you can add the standard coaching sections, but the question comes first and gets a real answer grounded in their swing data.\n`
    : ''

  let prompt: string

  if (compareSummary) {
    // Self-compare mode: same player, two takes. Coach for CONSISTENCY —
    // spot what's drifting between the two swings rather than rebuilding either.
    //
    // Baseline variant (compareMode === 'baseline'): same data plumbing, but
    // the framing shifts from "two takes side by side" to "best day vs today".
    // Everything else (rules, voice, biomechanics reference) is identical.
    const soloRef = getBiomechanicsReference(refShotType)

    const framingParagraph = isBaselineCompare
      ? `You are a tennis coach helping a player compare today's swing against their best-day baseline ("${baselineTag}"). Your job is progress tracking. Show them what's held up since that peak, what's drifted, and how to lock the good stuff back in.`
      : `You are a tennis coach watching the same player hit two different swings back to back. Your job is consistency. Help them spot what's staying the same and what's drifting between takes.`

    const anchorRule = isBaselineCompare
      ? `- Do NOT suggest rebuilds. The baseline IS the anchor. Anywhere today's swing drifts from "${baselineTag}", coach them back toward how they moved on their best day.`
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
${shotIntroBlock}
${tierBlock}
${focusBlock}${detectedIssuesBlock}
STRICT RULES:
- NEVER mention degrees, angles, or numbers of any kind. Describe everything in feel and body language.
- NEVER rate or score the player. No X/100, no percentages, no grades.
- WRITE LIKE A HUMAN COACH TALKING. Never use em dashes (—), en dashes (–), or hyphens (-) as a way to chain ideas or replace punctuation. Do not use them for emphasis, parenthetical asides, or to glue two clauses together. If you would have used a dash, write a full sentence with a period, OR connect the clauses with a comma, "and", "but", or a colon. Compound-word hyphens inside a single word like "feel-based" or "external-focus" are fine.
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
    const soloRef = getBiomechanicsReference(refShotType)

    // Advanced players get a trimmed prompt because listing three "tips"
    // when the swing is already clean forces the model to fabricate
    // problems. Baseline-compare is the right place for drift detection,
    // so we point them there instead.
    const isAdvanced = profile?.skill_tier === 'advanced'
    const advancedTrim = isAdvanced
      ? `\nADVANCED TRIM: Output at most 2 sentences unless you see a genuine mechanical issue. If the swing is solid, say so and redirect them to baseline-compare for drift detection. Do not force a "3 things to work on" section when there's nothing to fix.\n`
      : ''

    prompt = `You are a tennis coach talking to a player right after watching their swing on video. Be encouraging and practical.
${shotIntroBlock}
${tierBlock}
${advancedTrim}${focusBlock}${detectedIssuesBlock}
STRICT RULES:
- NEVER mention degrees, angles, or numbers of any kind. Not even once. Describe everything in feel and body language.
- NEVER rate or score the player (no X/100, no percentages, no grades). Just give advice.
- WRITE LIKE A HUMAN COACH TALKING. Never use em dashes (—), en dashes (–), or hyphens (-) as a way to chain ideas or replace punctuation. Do not use them for emphasis, parenthetical asides, or to glue two clauses together. If you would have used a dash, write a full sentence with a period, OR connect the clauses with a comma, "and", "but", or a colon. Compound-word hyphens inside a single word like "feel-based" or "external-focus" are fine.
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

  // Fallback (streaming) system prompt. Kept verbatim from the pre-tool-use
  // behavior so the fallback path produces the same output shape it did
  // before. Any mention of tool_use here would confuse the model on the
  // streaming branch.
  const fallbackSystemPrompt = `You are a veteran tennis coach who has been on court for 30 years. You talk like a real person having a conversation with a player between points.

VOICE RULES (follow these strictly):
- Write like you TALK. Short sentences. Casual tone. "Your hips are way ahead of your arm here" not "The hip rotation precedes the arm extension."
- WRITE LIKE A HUMAN COACH TALKING. Never use em dashes (—), en dashes (–), or hyphens (-) as a way to chain ideas or replace punctuation. Do not use them for emphasis, parenthetical asides, or to glue two clauses together. If you would have used a dash, write a full sentence with a period, OR connect the clauses with a comma, "and", "but", or a colon. Compound-word hyphens inside a single word like "feel-based" or "external-focus" are fine.
- NEVER list raw degree numbers on their own. Wrong: "elbow: 170°, ideal: 110°". Right: "your arm is almost locked straight when you want a nice relaxed bend."
- You CAN mention a number to back up a point, but always in a natural sentence: "you're only getting about 20 degrees of hip turn when ideal range is around 45."
- Use coaching cues a player can FEEL: "load your weight into your back leg", "let the racket drop behind you like a pendulum", "snap your hips like you're throwing a punch."
- Keep it practical. Every piece of advice should be something they can try on the very next ball.
- Sound encouraging, not critical. You're helping, not grading.
- No bullet points with just numbers. No tables. No clinical language.
- Write "you" and "your" constantly. Talk TO the player.

For every coaching cue you give, also recommend ONE OR TWO short exercises the player can do to fix that specific issue. Real drills, not generic "practice more" filler. Examples: "shadow forehands focusing on a relaxed elbow at contact, 20 reps each side", "slow-mo unit-turn drill against a fence, 10 reps", "wall rallies emphasizing early shoulder turn, 50 contacts." Specific, court-runnable, tied to the cue you just gave.`

  // Primary (tool_use) system prompt. Relaxes the em-dash rule just for the
  // literal advanced-baseline template, and instructs the model to emit its
  // response through the emit_coaching tool rather than freeform prose.
  const primarySystemPrompt = `You are a veteran tennis coach who has been on court for 30 years. You talk like a real person having a conversation with a player between points.

VOICE RULES (follow these strictly):
- Write like you TALK. Short sentences. Casual tone. "Your hips are way ahead of your arm here" not "The hip rotation precedes the arm extension."
- WRITE LIKE A HUMAN COACH TALKING. Never use em dashes (—), en dashes (–), or hyphens (-) as a way to chain ideas or replace punctuation. Do not use them for emphasis, parenthetical asides, or to glue two clauses together. If you would have used a dash, write a full sentence with a period, OR connect the clauses with a comma, "and", "but", or a colon. Compound-word hyphens inside a single word like "feel-based" or "external-focus" are fine.
- NEVER list raw degree numbers on their own. Wrong: "elbow: 170°, ideal: 110°". Right: "your arm is almost locked straight when you want a nice relaxed bend."
- You CAN mention a number to back up a point, but always in a natural sentence: "you're only getting about 20 degrees of hip turn when ideal range is around 45."
- Use coaching cues a player can FEEL: "load your weight into your back leg", "let the racket drop behind you like a pendulum", "snap your hips like you're throwing a punch."
- Keep it practical. Every piece of advice should be something they can try on the very next ball.
- Sound encouraging, not critical. You're helping, not grading.
- No bullet points with just numbers. No tables. No clinical language.
- Write "you" and "your" constantly. Talk TO the player.

For every cue you emit, the schema requires 1-2 entries in the cue's "exercises" array. These are short, court-runnable drills that fix the specific issue described in that cue. Real drills, not generic "practice more" filler. Examples: "shadow forehands focusing on a relaxed elbow at contact, 20 reps each side", "slow-mo unit-turn drill against a fence, 10 reps", "wall rallies emphasizing early shoulder turn, 50 contacts." Each exercise must tie directly to the cue's title.

You will emit your response by calling the emit_coaching tool. Do not write prose outside the tool call.`

  // llm_coached_tier is the tier we're actually telling the LLM to coach to.
  // For self-reported users, that's the profile tier. For skipped users, the
  // LLM picks one from the swing data — null at insert, backfilled from the
  // parsed assessment trailer after the stream completes.
  const coachedTier = profile?.skill_tier ?? null
  const tierKey: SkillTier | null = profile?.skill_tier ?? null
  const maxTokens = tierKey ? TIER_MAX_TOKENS[tierKey] : DEFAULT_MAX_TOKENS
  // Schema selection falls back to intermediate when the user has no profile;
  // metadata (tierKey) stays null so metrics and downgrade logic don't lie.
  const toolSchema = COACHING_TOOL_SCHEMAS[tierKey ?? 'intermediate']

  // resolvedShotType + resolvedBlobUrl are set above (before the prompt is
  // built) so the coaching prompt can be shot-type-aware. The telemetry row
  // below just consumes those same resolved values.

  // Insert the telemetry row BEFORE streaming starts so the X-Analysis-Event-Id
  // header can be set on the response the frontend binds its thumbs button to.
  // Wrapped in try/catch: telemetry must NEVER fail the coaching stream.
  let eventId: string | null = null
  try {
    const { data: inserted, error: insertError } = await supabaseAdmin
      .from('analysis_events')
      .insert({
        user_id: userId,
        session_id: sessionId ?? null,
        segment_id: null,
        self_reported_tier: profile?.skill_tier ?? null,
        was_skipped: skipped,
        handedness: profile?.dominant_hand ?? null,
        backhand_style: profile?.backhand_style ?? null,
        primary_goal: profile?.primary_goal ?? null,
        shot_type: resolvedShotType,
        blob_url: resolvedBlobUrl,
        composite_metrics: { user_summary: userSummary },
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

  // Telemetry-only camera-angle classification. Wrapped in after() so Vercel
  // keeps the invocation warm long enough for the Railway call + DB UPDATE
  // to complete. Without after(), serverless freeze after stream close was
  // dropping the UPDATE and leaving capture_quality_flag null in prod.
  after(() => classifyAndTagCaptureQuality(eventId, resolvedBlobUrl))

  const encoder = new TextEncoder()
  const ERROR_PREFIX = '\n\n[ERROR] '

  // Try the structured (tool_use) path first. Returns null on any failure so
  // the caller can fall back to the streaming path. Kill-switch short-circuits
  // before we ever touch the SDK.
  async function tryStructured(): Promise<
    | { markdown: string; toolInput: CoachingToolInput; outputTokens: number | null }
    | null
  > {
    if (process.env.STRUCTURED_OUTPUT_DISABLE === '1') return null
    // Skipped users (no profile) need the streaming path so the model can emit
    // a [TIER_ASSESSMENT: ...] trailer for tier inference. The tool_use schema
    // has no trailer slot, so structured-path output would discard that signal.
    if (!profile && skipped) return null
    try {
      const response = await anthropic.messages.create({
        model: 'claude-sonnet-4-6',
        max_tokens: maxTokens,
        system: primarySystemPrompt,
        messages: [{ role: 'user', content: prompt }],
        // TODO: drop this cast once COACHING_TOOL_SCHEMAS types `input_schema`
        // as `Anthropic.Tool.InputSchema` instead of `Record<string, unknown>`.
        tools: [toolSchema as unknown as Anthropic.Tool],
        tool_choice: { type: 'tool', name: COACHING_TOOL_NAME },
      })

      if (!Array.isArray(response.content) || response.content.length === 0) {
        return null
      }
      const toolBlock = response.content.find(
        (b): b is Extract<typeof b, { type: 'tool_use' }> =>
          b.type === 'tool_use' && b.name === COACHING_TOOL_NAME,
      )
      if (!toolBlock) return null

      const parsed = buildCoachingToolInput(toolBlock.input, tierKey)
      if (!parsed) return null

      const rendered = renderCoachingToolInputToMarkdown(parsed, tierKey)
      if (!rendered) return null

      const outputTokens = response.usage?.output_tokens ?? null
      return { markdown: rendered, toolInput: parsed, outputTokens }
    } catch (err) {
      console.error('analyze structured path failed:', err)
      return null
    }
  }

  // Fallback streaming path. Preserves the pre-tool-use behavior: buffer the
  // full response, strip the TIER_ASSESSMENT trailer, return the assessed tier
  // so the backfill UPDATE can record it. Per-tier max_tokens is now honored
  // here too (previously hardcoded to 1024).
  async function runFallback(): Promise<{
    markdown: string
    assessedTier: ReturnType<typeof parseTierAssessmentTrailer>['assessedTier']
    error: string | null
  }> {
    const messageStream = anthropic.messages.stream({
      model: 'claude-sonnet-4-6',
      max_tokens: maxTokens,
      system: fallbackSystemPrompt,
      messages: [{ role: 'user', content: prompt }],
    })

    let buffered = ''
    let error: string | null = null
    try {
      for await (const chunk of messageStream) {
        if (
          chunk.type === 'content_block_delta' &&
          chunk.delta.type === 'text_delta'
        ) {
          buffered += chunk.delta.text
        }
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Analysis stream failed'
    }
    const { assessedTier, stripped } = parseTierAssessmentTrailer(buffered)
    return { markdown: stripped, assessedTier, error }
  }

  // We buffer the full response before emitting, then parse + strip the
  // [TIER_ASSESSMENT: ...] trailer. Latency cost is the streaming UX (response
  // arrives in one shot instead of token-by-token), but the telemetry signal —
  // seeing what the model thought the tier was, even when the defanged
  // reconcile rule prevents it from acting — is worth far more than perceived
  // typing speed. Total generation is capped per-tier so the wait is bounded.
  const stream = new ReadableStream({
    async start(controller) {
      const structured = await tryStructured()

      let markdown: string
      let toolInput: CoachingToolInput | null
      let outputTokens: number | null
      let assessedTier:
        | 'beginner'
        | 'intermediate'
        | 'competitive'
        | 'advanced'
        | 'unknown'
        | null
      let streamError: string | null = null

      if (structured) {
        markdown = structured.markdown
        toolInput = structured.toolInput
        outputTokens = structured.outputTokens
        // tool_use forecloses model dissent on tier — by definition the model
        // coached to the tier we passed in. Null when we had no profile.
        assessedTier = tierKey
      } else {
        const fb = await runFallback()
        markdown = fb.markdown
        toolInput = null
        outputTokens = null
        assessedTier = fb.assessedTier
        streamError = fb.error
      }

      controller.enqueue(encoder.encode(markdown))
      if (streamError) {
        controller.enqueue(encoder.encode(`${ERROR_PREFIX}${streamError}`))
      }
      controller.close()

      // Backfill the parsed assessment + downgrade flag, plus the new output
      // metrics. Wrapped in after() so Vercel keeps the invocation live past
      // controller.close() — without this the DB UPDATE gets dropped when the
      // function freezes.
      if (eventId) {
        const backfillCoached =
          coachedTier ?? (assessedTier && assessedTier !== 'unknown' ? assessedTier : null)
        const downgrade = isTierDowngrade(backfillCoached, assessedTier)
        const metrics = extractResponseMetrics({
          tier: tierKey,
          toolInput,
          markdownText: markdown,
          outputTokens,
        })
        after(async () => {
          const { error } = await supabaseAdmin
            .from('analysis_events')
            .update({
              llm_assessed_tier: assessedTier,
              llm_coached_tier: backfillCoached,
              llm_tier_downgrade: downgrade,
              response_token_count: metrics.response_token_count,
              response_tip_count: metrics.response_tip_count,
              response_char_count: metrics.response_char_count,
              used_baseline_template: metrics.used_baseline_template,
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
