import { describe, it, expect } from 'vitest'
import {
  buildInferredTierCoachingBlock,
  buildTierCoachingBlock,
  getCoachingContext,
  getProfile,
  hasCompletedOnboardingFlow,
  isOnboarded,
  parseProfile,
  wasSkipped,
  type UserProfile,
  type SkillTier,
} from '@/lib/profile'

function baseProfile(overrides: Partial<UserProfile> = {}): UserProfile {
  return {
    skill_tier: 'intermediate',
    dominant_hand: 'right',
    backhand_style: 'two_handed',
    primary_goal: 'consistency',
    primary_goal_note: null,
    onboarded_at: '2024-01-01T00:00:00.000Z',
    ...overrides,
  }
}

describe('buildTierCoachingBlock', () => {
  const tiers: SkillTier[] = ['beginner', 'intermediate', 'competitive', 'advanced']

  it.each(tiers)('returns a non-empty coaching block for %s tier', (tier) => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: tier }))
    expect(block.length).toBeGreaterThan(100)
  })

  it('surfaces the beginner "foundation" rule', () => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: 'beginner' }))
    expect(block).toMatch(/one foundation cue/i)
    expect(block).toMatch(/lead with what's working/i)
  })

  it('tells intermediate to mix foundations and refinements', () => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: 'intermediate' }))
    expect(block).toMatch(/foundations and refinements/i)
  })

  it('tells competitive to focus on execution not foundations', () => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: 'competitive' }))
    expect(block).toMatch(/execution and refinement/i)
  })

  it('tells advanced there is nothing to fix by default', () => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: 'advanced' }))
    expect(block).toMatch(/nothing to fix/i)
    expect(block).toMatch(/baseline/i)
  })

  it.each(tiers)('includes the reconcile rule on the %s block', (tier) => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: tier }))
    expect(block).toMatch(/RECONCILE RULE/)
    expect(block).toMatch(/lock this in before we refine higher up/i)
  })

  it('mentions the right side for right-handed players', () => {
    const block = buildTierCoachingBlock(
      baseProfile({ dominant_hand: 'right', backhand_style: 'two_handed' }),
    )
    expect(block).toMatch(/right-handed/i)
    expect(block).toMatch(/dominant arm is the right/i)
    expect(block).toMatch(/two-handed backhand/i)
  })

  it('flips dominant side for left-handed players', () => {
    const block = buildTierCoachingBlock(
      baseProfile({ dominant_hand: 'left', backhand_style: 'one_handed' }),
    )
    expect(block).toMatch(/left-handed/i)
    expect(block).toMatch(/dominant arm is the left/i)
    expect(block).toMatch(/one-handed backhand/i)
  })

  it('weights the block toward the stated primary goal', () => {
    const block = buildTierCoachingBlock(baseProfile({ primary_goal: 'power' }))
    expect(block).toMatch(/GOAL WEIGHTING/)
    expect(block).toMatch(/More power/)
  })

  it('includes the free-text note when primary goal is other', () => {
    const block = buildTierCoachingBlock(
      baseProfile({ primary_goal: 'other', primary_goal_note: 'serve placement' }),
    )
    expect(block).toMatch(/serve placement/)
  })

  it('returns a generic fallback block when profile is null', () => {
    const block = buildTierCoachingBlock(null)
    expect(block).toMatch(/READ THE SKILL LEVEL FIRST/i)
    // The fallback intentionally omits the per-tier / per-player metadata
    // that only makes sense for an onboarded user.
    expect(block).not.toMatch(/RECONCILE RULE/)
    expect(block).not.toMatch(/GOAL WEIGHTING/)
  })
})

describe('parseProfile edge cases', () => {
  it('returns null for null metadata', () => {
    expect(parseProfile(null)).toBeNull()
    expect(parseProfile(undefined)).toBeNull()
  })

  it('returns null when metadata is not an object', () => {
    // Cast needed because parseProfile's type narrows to object; the runtime
    // defence still matters for raw Supabase payloads.
    expect(parseProfile('oops' as unknown as Record<string, unknown>)).toBeNull()
  })

  it('returns null when skill_tier is unknown', () => {
    expect(
      parseProfile({
        skill_tier: 'grandmaster',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'power',
        onboarded_at: '2024-01-01T00:00:00.000Z',
      }),
    ).toBeNull()
  })

  it('returns null when a required field is missing', () => {
    expect(
      parseProfile({
        skill_tier: 'beginner',
        dominant_hand: 'right',
        // backhand_style omitted
        primary_goal: 'power',
        onboarded_at: '2024-01-01T00:00:00.000Z',
      }),
    ).toBeNull()
  })

  it('returns null when onboarded_at is empty', () => {
    expect(
      parseProfile({
        skill_tier: 'beginner',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'power',
        onboarded_at: '',
      }),
    ).toBeNull()
  })

  it('normalizes primary_goal_note to null when goal is not "other"', () => {
    const result = parseProfile({
      skill_tier: 'beginner',
      dominant_hand: 'right',
      backhand_style: 'two_handed',
      primary_goal: 'power',
      primary_goal_note: 'should be dropped',
      onboarded_at: '2024-01-01T00:00:00.000Z',
    })
    expect(result?.primary_goal_note).toBeNull()
  })

  it('truncates primary_goal_note at 120 characters', () => {
    const long = 'a'.repeat(500)
    const result = parseProfile({
      skill_tier: 'advanced',
      dominant_hand: 'left',
      backhand_style: 'one_handed',
      primary_goal: 'other',
      primary_goal_note: long,
      onboarded_at: '2024-01-01T00:00:00.000Z',
    })
    expect(result?.primary_goal_note?.length).toBe(120)
  })

  it('drops whitespace-only notes for "other" goal', () => {
    const result = parseProfile({
      skill_tier: 'advanced',
      dominant_hand: 'left',
      backhand_style: 'one_handed',
      primary_goal: 'other',
      primary_goal_note: '   ',
      onboarded_at: '2024-01-01T00:00:00.000Z',
    })
    expect(result?.primary_goal_note).toBeNull()
  })
})

describe('isOnboarded', () => {
  it('is false for malformed metadata', () => {
    expect(isOnboarded({ skill_tier: 'wrong' })).toBe(false)
  })

  it('is true for a valid profile payload', () => {
    expect(
      isOnboarded({
        skill_tier: 'intermediate',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'consistency',
        onboarded_at: '2024-01-01T00:00:00.000Z',
      }),
    ).toBe(true)
  })
})

describe('wasSkipped', () => {
  it('is true when skipped_onboarding_at is a non-empty string', () => {
    expect(wasSkipped({ skipped_onboarding_at: '2024-06-01T00:00:00.000Z' })).toBe(true)
  })

  it('is false when skipped_onboarding_at is missing', () => {
    expect(wasSkipped({})).toBe(false)
  })

  it('is false when skipped_onboarding_at is empty', () => {
    expect(wasSkipped({ skipped_onboarding_at: '' })).toBe(false)
  })

  it('is false when skipped_onboarding_at is not a string', () => {
    expect(wasSkipped({ skipped_onboarding_at: 12345 })).toBe(false)
    expect(wasSkipped({ skipped_onboarding_at: true })).toBe(false)
  })

  it('is false for null / undefined metadata', () => {
    expect(wasSkipped(null)).toBe(false)
    expect(wasSkipped(undefined)).toBe(false)
  })
})

describe('hasCompletedOnboardingFlow', () => {
  it('is true when the user has a full profile', () => {
    expect(
      hasCompletedOnboardingFlow({
        skill_tier: 'intermediate',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'consistency',
        onboarded_at: '2024-01-01T00:00:00.000Z',
      }),
    ).toBe(true)
  })

  it('is true when the user has skipped', () => {
    expect(hasCompletedOnboardingFlow({ skipped_onboarding_at: '2024-06-01T00:00:00.000Z' })).toBe(
      true,
    )
  })

  it('is false when the user has neither onboarded nor skipped', () => {
    expect(hasCompletedOnboardingFlow({})).toBe(false)
    expect(hasCompletedOnboardingFlow(null)).toBe(false)
  })
})

describe('buildInferredTierCoachingBlock', () => {
  const block = buildInferredTierCoachingBlock()

  it('returns a non-empty block', () => {
    expect(block.length).toBeGreaterThan(100)
  })

  it('names all four tiers', () => {
    expect(block).toMatch(/beginner/i)
    expect(block).toMatch(/intermediate/i)
    expect(block).toMatch(/competitive/i)
    expect(block).toMatch(/advanced/i)
  })

  it('tells the LLM to name its inferred tier at the top', () => {
    expect(block).toMatch(/NAME YOUR INFERRED TIER/)
    expect(block).toMatch(/italic parentheses/i)
  })

  it('includes the reconcile rule so the model can shift mid-response', () => {
    expect(block).toMatch(/RECONCILE RULE/)
  })
})

describe('getCoachingContext', () => {
  it('returns the parsed profile with skipped=false for an onboarded user', async () => {
    const client = {
      auth: {
        getUser: async () => ({
          data: {
            user: {
              user_metadata: {
                skill_tier: 'competitive',
                dominant_hand: 'left',
                backhand_style: 'one_handed',
                primary_goal: 'topspin',
                onboarded_at: '2024-05-01T00:00:00.000Z',
              },
            },
          },
        }),
      },
    }
    const ctx = await getCoachingContext(client)
    expect(ctx.profile?.skill_tier).toBe('competitive')
    expect(ctx.skipped).toBe(false)
  })

  it('returns profile=null and skipped=true when the user only skipped', async () => {
    const client = {
      auth: {
        getUser: async () => ({
          data: {
            user: {
              user_metadata: { skipped_onboarding_at: '2024-06-01T00:00:00.000Z' },
            },
          },
        }),
      },
    }
    const ctx = await getCoachingContext(client)
    expect(ctx.profile).toBeNull()
    expect(ctx.skipped).toBe(true)
  })

  it('returns null profile and false skipped when the user has neither', async () => {
    const client = {
      auth: {
        getUser: async () => ({ data: { user: { user_metadata: {} } } }),
      },
    }
    const ctx = await getCoachingContext(client)
    expect(ctx.profile).toBeNull()
    expect(ctx.skipped).toBe(false)
  })

  it('returns null profile and false skipped when getUser rejects', async () => {
    const client = {
      auth: {
        getUser: async () => {
          throw new Error('network down')
        },
      },
    }
    const ctx = await getCoachingContext(client)
    expect(ctx.profile).toBeNull()
    expect(ctx.skipped).toBe(false)
  })
})

describe('getProfile', () => {
  it('returns null when the Supabase client has no user', async () => {
    const client = {
      auth: {
        getUser: async () => ({ data: { user: null } }),
      },
    }
    const profile = await getProfile(client)
    expect(profile).toBeNull()
  })

  it('parses metadata into a profile when the user is present', async () => {
    const client = {
      auth: {
        getUser: async () => ({
          data: {
            user: {
              user_metadata: {
                skill_tier: 'competitive',
                dominant_hand: 'left',
                backhand_style: 'one_handed',
                primary_goal: 'topspin',
                onboarded_at: '2024-05-01T00:00:00.000Z',
              },
            },
          },
        }),
      },
    }
    const profile = await getProfile(client)
    expect(profile?.skill_tier).toBe('competitive')
    expect(profile?.dominant_hand).toBe('left')
  })

  it('returns null when getUser rejects', async () => {
    const client = {
      auth: {
        getUser: async () => {
          throw new Error('network down')
        },
      },
    }
    const profile = await getProfile(client)
    expect(profile).toBeNull()
  })
})
