import { describe, it, expect } from 'vitest'
import {
  ADVANCED_BASELINE_TEMPLATE,
  buildInferredTierCoachingBlock,
  buildTierCoachingBlock,
  COACHING_TOOL_NAME,
  COACHING_TOOL_SCHEMAS,
  getCoachingContext,
  getProfile,
  hasCompletedOnboardingFlow,
  isOnboarded,
  isTierDowngrade,
  parseProfile,
  parseTierAssessmentTrailer,
  TIER_MAX_TOKENS,
  TIER_RULES,
  tierRank,
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

  it('surfaces the beginner external-focus rule with a word cap', () => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: 'beginner' }))
    expect(block).toMatch(/2 or 3 external-focus coaching cues/i)
    expect(block).toMatch(/under 120 words/i)
    expect(block).toMatch(/lead with ONE sentence about a strength/i)
  })

  it('tells intermediate to mix a foundation tune-up with a refinement under 180 words', () => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: 'intermediate' }))
    expect(block).toMatch(/under 180 words/i)
    expect(block).toMatch(/2 or 3 coaching cues/i)
  })

  it('tells competitive to emit exactly 3 execution cues under 220 words', () => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: 'competitive' }))
    expect(block).toMatch(/exactly 3 execution cues/i)
    expect(block).toMatch(/under 220 words/i)
  })

  it('tells advanced to default to the baseline template and keep the frame positive', () => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: 'advanced' }))
    expect(block).toContain(ADVANCED_BASELINE_TEMPLATE)
    expect(block).toMatch(/under 60 words/i)
    expect(block).toMatch(/Keep the frame positive/i)
  })

  it.each(tiers)('includes the defanged reconcile rule on the %s block', (tier) => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: tier }))
    expect(block).toMatch(/RECONCILE RULE/)
    // New "coach to the contract" phrasing replaces the old downgrade-permission.
    expect(block).toMatch(/self-reported tier is a CONTRACT/)
  })

  it.each(tiers)('does NOT contain downgrade-permission phrasing on the %s block', (tier) => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: tier }))
    // Old phrasings that invited a downgrade mid-response are gone.
    expect(block).not.toMatch(/lock this in before we refine higher up/i)
    expect(block).not.toMatch(/shift into foundation coaching/i)
    expect(block).not.toMatch(/meaningfully more elementary/i)
    // Explicit anti-downgrade language is present.
    expect(block).toMatch(/do not downgrade/i)
    expect(block).toMatch(/camera geometry/i)
  })

  it.each(tiers)('includes the TIER_ASSESSMENT trailer instruction on the %s block', (tier) => {
    const block = buildTierCoachingBlock(baseProfile({ skill_tier: tier }))
    expect(block).toMatch(/\[TIER_ASSESSMENT:/)
    expect(block).toMatch(/server parses and strips/i)
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

describe('TIER_RULES phrasing invariants', () => {
  const tiers: SkillTier[] = ['beginner', 'intermediate', 'competitive', 'advanced']

  it.each(tiers)('does not use negation phrasing in the %s tier rule', (tier) => {
    // Scope assertions to the raw TIER_RULES string — buildTierCoachingBlock
    // includes RECONCILE_RULE, which is allowed to keep its negations.
    const rule = TIER_RULES[tier]
    expect(rule).not.toMatch(/\bnever\b/i)
    expect(rule).not.toMatch(/\bdon't\b/i)
    expect(rule).not.toMatch(/\bdo not\b/i)
    expect(rule).not.toMatch(/\bnothing to\b/i)
  })

  it('beginner rule uses external-focus cue examples', () => {
    const rule = TIER_RULES.beginner
    expect(rule).toMatch(/push the ball|push through|racket up by your ear|finish with/i)
    expect(rule).not.toMatch(/rotate your shoulder/i)
    expect(rule).not.toMatch(/bend your knee/i)
  })

  it('advanced rule contains the literal baseline template', () => {
    expect(TIER_RULES.advanced).toContain(ADVANCED_BASELINE_TEMPLATE)
  })
})

describe('TIER_MAX_TOKENS', () => {
  it('is monotonic (advanced < beginner < intermediate < competitive) and has expected values', () => {
    // Bumped after adding per-cue exercises (1-2 drills each) — the
    // previous ceiling truncated 3-cue responses mid-sentence. See
    // lib/profile.ts:TIER_MAX_TOKENS for the rationale.
    expect(TIER_MAX_TOKENS.advanced).toBe(350)
    expect(TIER_MAX_TOKENS.beginner).toBe(750)
    expect(TIER_MAX_TOKENS.intermediate).toBe(1000)
    expect(TIER_MAX_TOKENS.competitive).toBe(1300)
    expect(TIER_MAX_TOKENS.advanced).toBeLessThan(TIER_MAX_TOKENS.beginner)
    expect(TIER_MAX_TOKENS.beginner).toBeLessThan(TIER_MAX_TOKENS.intermediate)
    expect(TIER_MAX_TOKENS.intermediate).toBeLessThan(TIER_MAX_TOKENS.competitive)
  })
})

describe('COACHING_TOOL_SCHEMAS', () => {
  function cuesSchemaFor(tier: SkillTier): { minItems: number; maxItems: number } {
    const schema = COACHING_TOOL_SCHEMAS[tier].input_schema as {
      properties: { cues: { minItems: number; maxItems: number } }
    }
    return schema.properties.cues
  }

  it('enforces per-tier cue counts', () => {
    expect(cuesSchemaFor('beginner')).toMatchObject({ minItems: 2, maxItems: 3 })
    expect(cuesSchemaFor('intermediate')).toMatchObject({ minItems: 2, maxItems: 3 })
    expect(cuesSchemaFor('competitive')).toMatchObject({ minItems: 3, maxItems: 3 })
    expect(cuesSchemaFor('advanced')).toMatchObject({ minItems: 0, maxItems: 1 })
  })

  it('all schemas use the same tool name', () => {
    const tiers: SkillTier[] = ['beginner', 'intermediate', 'competitive', 'advanced']
    for (const tier of tiers) {
      expect(COACHING_TOOL_SCHEMAS[tier].name).toBe('emit_coaching')
      expect(COACHING_TOOL_SCHEMAS[tier].name).toBe(COACHING_TOOL_NAME)
    }
  })

  it('all schemas reject additionalProperties', () => {
    const tiers: SkillTier[] = ['beginner', 'intermediate', 'competitive', 'advanced']
    for (const tier of tiers) {
      const schema = COACHING_TOOL_SCHEMAS[tier].input_schema as Record<string, unknown>
      expect(schema.additionalProperties).toBe(false)
    }
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

  it('keeps the three-signal override gate in the skipped-user path', () => {
    // Skipped users have no self-report contract to protect, so the LLM is
    // explicitly allowed to shift tier when multiple independent signals agree.
    expect(block).toMatch(/three-signal override/i)
    expect(block).toMatch(/at least THREE independent signals/i)
  })

  it('includes the TIER_ASSESSMENT trailer instruction', () => {
    expect(block).toMatch(/\[TIER_ASSESSMENT:/)
    expect(block).toMatch(/server parses and strips/i)
  })
})

describe('parseTierAssessmentTrailer', () => {
  it('returns null assessed tier and original text when no trailer is present', () => {
    const r = parseTierAssessmentTrailer('Here is some coaching advice.')
    expect(r.assessedTier).toBeNull()
    expect(r.stripped).toBe('Here is some coaching advice.')
  })

  it('extracts each recognized tier', () => {
    const tiers = ['beginner', 'intermediate', 'competitive', 'advanced', 'unknown'] as const
    for (const t of tiers) {
      const r = parseTierAssessmentTrailer(`Coach body here.\n\n[TIER_ASSESSMENT: ${t}]`)
      expect(r.assessedTier).toBe(t)
      expect(r.stripped).toBe('Coach body here.')
    }
  })

  it('is case-insensitive on the tier value', () => {
    const r = parseTierAssessmentTrailer('body\n[TIER_ASSESSMENT: Competitive]')
    expect(r.assessedTier).toBe('competitive')
  })

  it('strips trailing whitespace after the trailer', () => {
    const r = parseTierAssessmentTrailer('body text\n\n[TIER_ASSESSMENT: advanced]   \n')
    expect(r.assessedTier).toBe('advanced')
    expect(r.stripped).toBe('body text')
  })

  it('strips the trailer even when the model appends trailing text after it', () => {
    // Regex is intentionally unanchored so a stray period, emoji, or extra
    // sentence after the tag doesn't cause the raw trailer to leak to the user.
    const r = parseTierAssessmentTrailer(
      'Nice work.\n\n[TIER_ASSESSMENT: advanced] 🎾',
    )
    expect(r.assessedTier).toBe('advanced')
    expect(r.stripped).not.toContain('TIER_ASSESSMENT')
    expect(r.stripped).toContain('Nice work.')
  })

  it('uses the LAST trailer match when the model emits the format more than once', () => {
    const r = parseTierAssessmentTrailer(
      'Earlier I thought [TIER_ASSESSMENT: intermediate] but now\n\n[TIER_ASSESSMENT: advanced]',
    )
    expect(r.assessedTier).toBe('advanced')
    expect(r.stripped).not.toContain('TIER_ASSESSMENT')
  })
})

describe('tierRank', () => {
  it('orders tiers correctly', () => {
    expect(tierRank('beginner')).toBeLessThan(tierRank('intermediate'))
    expect(tierRank('intermediate')).toBeLessThan(tierRank('competitive'))
    expect(tierRank('competitive')).toBeLessThan(tierRank('advanced'))
  })
})

describe('isTierDowngrade', () => {
  it('is true when assessed tier is strictly lower than coached tier', () => {
    expect(isTierDowngrade('advanced', 'intermediate')).toBe(true)
    expect(isTierDowngrade('competitive', 'beginner')).toBe(true)
  })

  it('is false when assessed tier matches the coached tier', () => {
    expect(isTierDowngrade('intermediate', 'intermediate')).toBe(false)
  })

  it('is false when assessed tier is higher than coached tier', () => {
    expect(isTierDowngrade('beginner', 'advanced')).toBe(false)
  })

  it('is false when either side is null', () => {
    expect(isTierDowngrade(null, 'advanced')).toBe(false)
    expect(isTierDowngrade('advanced', null)).toBe(false)
    expect(isTierDowngrade(null, null)).toBe(false)
  })

  it('is false when the assessed tier is unknown', () => {
    // "unknown" means the LLM couldn't tell — it shouldn't count as downgrade signal.
    expect(isTierDowngrade('advanced', 'unknown')).toBe(false)
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
