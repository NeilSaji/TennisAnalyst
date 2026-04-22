import { describe, it, expect } from 'vitest'
import {
  buildLiveCoachingPrompt,
  LIVE_SYSTEM_PROMPT,
  type LivePromptSwing,
} from '@/lib/livePrompt'
import type { UserProfile } from '@/lib/profile'

const intermediateProfile: UserProfile = {
  skill_tier: 'intermediate',
  dominant_hand: 'right',
  backhand_style: 'two_handed',
  primary_goal: 'consistency',
  primary_goal_note: null,
  onboarded_at: '2026-04-20T00:00:00Z',
}

const advancedProfile: UserProfile = {
  skill_tier: 'advanced',
  dominant_hand: 'left',
  backhand_style: 'one_handed',
  primary_goal: 'other',
  primary_goal_note: 'reduce forehand pronation',
  onboarded_at: '2026-04-20T00:00:00Z',
}

function makeSwing(i: number): LivePromptSwing {
  return {
    angleSummary: `preparation: elbow_R=135° shoulder_R=100° ...\ncontact: elbow_R=160° ...`,
    startMs: i * 2000,
    endMs: i * 2000 + 600,
  }
}

describe('LIVE_SYSTEM_PROMPT', () => {
  it('enforces TTS-friendly brevity rules', () => {
    expect(LIVE_SYSTEM_PROMPT).toMatch(/ONE sentence/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/25 words/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/no markdown/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/no lists/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/external-focus/i)
  })
})

describe('buildLiveCoachingPrompt', () => {
  it('names the tier when profile is present', () => {
    const prompt = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1), makeSwing(2), makeSwing(3)],
    })
    expect(prompt).toMatch(/intermediate/i)
    expect(prompt).toMatch(/right-handed/i)
    expect(prompt).toMatch(/two-handed backhand/i)
    expect(prompt).toMatch(/consistency/i)
    expect(prompt).toMatch(/4 forehands in a row/i)
    expect(prompt).toMatch(/LAST 4 SWINGS/i)
    expect(prompt).toMatch(/SWING 1/i)
    expect(prompt).toMatch(/SWING 4/i)
  })

  it('handles the advanced tier and primary_goal_note', () => {
    const prompt = buildLiveCoachingPrompt({
      profile: advancedProfile,
      skipped: false,
      shotType: 'backhand',
      swings: [makeSwing(0), makeSwing(1)],
    })
    expect(prompt).toMatch(/advanced/i)
    expect(prompt).toMatch(/left-handed/i)
    expect(prompt).toMatch(/one-handed backhand/i)
    expect(prompt).toMatch(/reduce forehand pronation/)
  })

  it('uses inferred-tier language for skipped users', () => {
    const prompt = buildLiveCoachingPrompt({
      profile: null,
      skipped: true,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1), makeSwing(2)],
    })
    expect(prompt).toMatch(/skipped onboarding/i)
    expect(prompt).toMatch(/infer tier/i)
    expect(prompt).not.toMatch(/Player is/)
  })

  it('uses generic language when neither profile nor skipped flag is present', () => {
    const prompt = buildLiveCoachingPrompt({
      profile: null,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1)],
    })
    expect(prompt).toMatch(/tier unknown/i)
    expect(prompt).toMatch(/broadly applicable/i)
  })

  it('includes the baseline block only when baselineSummary is provided', () => {
    const withBaseline = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
      baselineSummary: 'preparation: elbow_R=120° ...',
      baselineLabel: 'June 14 winner',
    })
    expect(withBaseline).toMatch(/BASELINE \(June 14 winner\)/)
    expect(withBaseline).toMatch(/elbow_R=120°/)

    const withoutBaseline = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
    })
    expect(withoutBaseline).not.toMatch(/BASELINE/)
  })

  it('asks for a single cue at the end', () => {
    const prompt = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1), makeSwing(2)],
    })
    expect(prompt.trimEnd()).toMatch(/ONE cue for the next ball\. One sentence\. Max 25 words\.$/)
  })

  it('singular shot language when only one swing', () => {
    const prompt = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
    })
    expect(prompt).toMatch(/hit 1 forehand in a row/i)
    expect(prompt).toMatch(/LAST 1 SWING:/)
  })
})
