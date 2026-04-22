// Prompt builder for POST /api/live-coach. Kept separate from the /analyze
// prompts because live coaching has a fundamentally different output shape:
// ONE short sentence meant to be spoken through earbuds between points, not a
// markdown-structured review. The tier-rules infrastructure from lib/profile
// still applies — we just layer TTS-friendly brevity rules on top.

import type { SkillTier, UserProfile } from './profile'

export const LIVE_SYSTEM_PROMPT = `You are a tennis coach yelling one short cue across the net between points. The player hears this through earbuds while hitting — they cannot stop to read anything.

STRICT RULES (non-negotiable):
- Respond with exactly ONE sentence.
- At most 25 words.
- No markdown. No lists. No numbers. No headings. No degrees or measurement units.
- Feel-based and external-focus. Talk like a coach giving a quick cue, not a writer.
- Summarize the whole batch as one observation — never walk through the swings one by one.
- Never greet. Never sign off. No meta-commentary like "here's your cue".`

// Short live-mode guidance per tier. Kept separate from TIER_RULES in
// lib/profile.ts (which shape multi-section markdown output) so the live
// system prompt stays blunt and a single paragraph.
const LIVE_TIER_CUES: Record<SkillTier, string> = {
  beginner:
    'Tier is beginner. Use one external-focus cue — where to push the ball, how to finish the racket, where to feel the weight.',
  intermediate:
    'Tier is intermediate. Use one feel-based cue they can try on the next ball — load, turn, contact, or finish.',
  competitive:
    'Tier is competitive. Use one execution cue focused on matchplay polish — timing, target, or finish direction.',
  advanced:
    "Tier is advanced. Default to reinforcing what's clean. Only give a cue if you see a concrete micro-refinement.",
}

export interface LivePromptSwing {
  angleSummary: string
  startMs: number
  endMs: number
}

export interface LivePromptContext {
  profile: UserProfile | null
  skipped: boolean
  shotType: string
  swings: LivePromptSwing[]
  baselineSummary?: string | null
  baselineLabel?: string | null
}

function formatMs(ms: number): string {
  return (Math.round(ms / 100) / 10).toFixed(1) + 's'
}

function buildPlayerLine(profile: UserProfile): string {
  const hand = profile.dominant_hand === 'right' ? 'right-handed' : 'left-handed'
  const backhand =
    profile.backhand_style === 'one_handed' ? 'one-handed backhand' : 'two-handed backhand'
  const goalNote =
    profile.primary_goal === 'other' && profile.primary_goal_note
      ? ` ("${profile.primary_goal_note}")`
      : ''
  return `Player is ${hand}, ${backhand}. Stated focus: ${profile.primary_goal}${goalNote}.`
}

export function buildLiveCoachingPrompt(ctx: LivePromptContext): string {
  const { profile, skipped, shotType, swings, baselineSummary, baselineLabel } = ctx

  const tier: SkillTier | null = profile?.skill_tier ?? null
  const tierLine = tier
    ? LIVE_TIER_CUES[tier]
    : skipped
      ? 'Tier unknown because the player skipped onboarding. Infer tier from the data and coach to it.'
      : 'Tier unknown. Keep the cue broadly applicable.'

  const baselineBlock = baselineSummary
    ? `BASELINE (${baselineLabel ?? 'your best day'}):\n${baselineSummary}`
    : ''

  const swingLines = swings
    .map(
      (s, i) =>
        `SWING ${i + 1} (${formatMs(s.startMs)}–${formatMs(s.endMs)}):\n${s.angleSummary}`,
    )
    .join('\n\n')

  const countWord = swings.length === 1 ? shotType : `${shotType}s`

  const sections = [
    `They just hit ${swings.length} ${countWord} in a row.`,
    tierLine,
    profile ? buildPlayerLine(profile) : null,
    baselineBlock || null,
    `LAST ${swings.length} ${swings.length === 1 ? 'SWING' : 'SWINGS'}:\n${swingLines}`,
    'Give ONE cue for the next ball. One sentence. Max 25 words.',
  ].filter((s): s is string => typeof s === 'string' && s.length > 0)

  return sections.join('\n\n')
}
