// User profile stored on Supabase `auth.users.user_metadata`.
//
// Written client-side via `supabase.auth.updateUser({ data: {...} })`.
// Read server-side via `supabase.auth.getUser()` then `parseProfile()`.
//
// Skill tier drives how much and what kind of coaching the LLM produces
// (see `app/api/analyze/route.ts`). Handedness + backhand style change
// which dominant-arm joints the prompt emphasizes. Primary goal weights
// which observations get surfaced first.

export type SkillTier =
  | 'beginner' // New to tennis
  | 'intermediate' // Recreational player who rallies consistently
  | 'competitive' // Plays matches, tournaments, strong club level
  | 'advanced' // Top club / college / pro

export type DominantHand = 'right' | 'left'
export type BackhandStyle = 'one_handed' | 'two_handed'

export type PrimaryGoal =
  | 'power'
  | 'consistency'
  | 'topspin'
  | 'slice'
  | 'learning'
  | 'other'

export interface UserProfile {
  skill_tier: SkillTier
  dominant_hand: DominantHand
  backhand_style: BackhandStyle
  primary_goal: PrimaryGoal
  // Only set when primary_goal === 'other'. Bounded to 120 chars to keep
  // the prompt budget sane.
  primary_goal_note: string | null
  onboarded_at: string // ISO 8601
}

const SKILL_TIERS: readonly SkillTier[] = ['beginner', 'intermediate', 'competitive', 'advanced']
const DOMINANT_HANDS: readonly DominantHand[] = ['right', 'left']
const BACKHAND_STYLES: readonly BackhandStyle[] = ['one_handed', 'two_handed']
const PRIMARY_GOALS: readonly PrimaryGoal[] = [
  'power',
  'consistency',
  'topspin',
  'slice',
  'learning',
  'other',
]

export const SKILL_TIER_LABELS: Record<SkillTier, string> = {
  beginner: 'New to tennis',
  intermediate: 'Intermediate',
  competitive: 'Competitive',
  advanced: 'Advanced or pro',
}

export const PRIMARY_GOAL_LABELS: Record<PrimaryGoal, string> = {
  power: 'More power',
  consistency: 'More consistency',
  topspin: 'Cleaner topspin',
  slice: 'Fix my slice',
  learning: 'Just learning',
  other: 'Something else',
}

// Parses raw user_metadata into a validated UserProfile, or null if any
// required field is missing or invalid. Callers use this to decide both
// "is the user onboarded?" and "do we have a profile to consume?"
export function parseProfile(
  metadata: Record<string, unknown> | null | undefined,
): UserProfile | null {
  if (!metadata || typeof metadata !== 'object') return null

  const skill_tier = metadata.skill_tier
  const dominant_hand = metadata.dominant_hand
  const backhand_style = metadata.backhand_style
  const primary_goal = metadata.primary_goal
  const primary_goal_note = metadata.primary_goal_note
  const onboarded_at = metadata.onboarded_at

  if (typeof skill_tier !== 'string' || !SKILL_TIERS.includes(skill_tier as SkillTier)) {
    return null
  }
  if (
    typeof dominant_hand !== 'string' ||
    !DOMINANT_HANDS.includes(dominant_hand as DominantHand)
  ) {
    return null
  }
  if (
    typeof backhand_style !== 'string' ||
    !BACKHAND_STYLES.includes(backhand_style as BackhandStyle)
  ) {
    return null
  }
  if (typeof primary_goal !== 'string' || !PRIMARY_GOALS.includes(primary_goal as PrimaryGoal)) {
    return null
  }
  if (typeof onboarded_at !== 'string' || !onboarded_at) {
    return null
  }

  // primary_goal_note is only meaningful when goal === 'other'; otherwise normalize to null
  let note: string | null = null
  if (primary_goal === 'other') {
    if (typeof primary_goal_note === 'string' && primary_goal_note.trim()) {
      note = primary_goal_note.trim().slice(0, 120)
    }
  }

  return {
    skill_tier: skill_tier as SkillTier,
    dominant_hand: dominant_hand as DominantHand,
    backhand_style: backhand_style as BackhandStyle,
    primary_goal: primary_goal as PrimaryGoal,
    primary_goal_note: note,
    onboarded_at,
  }
}

// Convenience for the routing gate: does the signed-in user have a
// complete profile? Treats malformed metadata as "not onboarded" so the
// gate heals broken entries by re-running onboarding.
export function isOnboarded(
  metadata: Record<string, unknown> | null | undefined,
): boolean {
  return parseProfile(metadata) !== null
}

// Did the user explicitly skip the onboarding form? Skipped users get a
// different prompt shape (LLM infers their tier from swing data) but still
// pass through the middleware gate so we don't nag them forever.
export function wasSkipped(
  metadata: Record<string, unknown> | null | undefined,
): boolean {
  if (!metadata || typeof metadata !== 'object') return false
  const t = (metadata as Record<string, unknown>).skipped_onboarding_at
  return typeof t === 'string' && t.length > 0
}

// Has the user made any onboarding decision — either completing or skipping?
// Used by middleware to decide whether to redirect to /onboarding. A true
// return here means "pass through"; false means "gate to onboarding".
export function hasCompletedOnboardingFlow(
  metadata: Record<string, unknown> | null | undefined,
): boolean {
  return isOnboarded(metadata) || wasSkipped(metadata)
}

// ---------------------------------------------------------------------------
// Server-side additions. Kept below the shared contract so the type/parser
// surface stays untouched.
// ---------------------------------------------------------------------------

// Minimal shape of the Supabase server client we need. Declared locally so
// this module stays decoupled from @supabase/ssr — the real client satisfies
// it, and tests can pass in a hand-rolled stub.
interface SupabaseLikeClient {
  auth: {
    getUser: () => Promise<{
      data: { user: { user_metadata?: Record<string, unknown> | null } | null }
      error?: unknown
    }>
  }
}

// Fetches the signed-in user and returns their parsed profile, or null if
// the caller is anonymous or their metadata is malformed. Swallowing the
// Supabase error is intentional: the LLM routes fall back to generic
// coaching when there's no profile, and we never want auth hiccups to take
// down the coaching stream.
export async function getProfile(
  client: SupabaseLikeClient,
): Promise<UserProfile | null> {
  try {
    const { data } = await client.auth.getUser()
    const metadata = data.user?.user_metadata
    return parseProfile(metadata)
  } catch {
    return null
  }
}

// Human label for the stated goal, including the free-text note for 'other'.
function describeGoal(profile: UserProfile): string {
  const label = PRIMARY_GOAL_LABELS[profile.primary_goal]
  if (profile.primary_goal === 'other' && profile.primary_goal_note) {
    return `${label} ("${profile.primary_goal_note}")`
  }
  return label
}

// Defanged reconcile rule used only for self-reported-tier players. Previously
// this told the LLM to downgrade mid-response if the swing "looked elementary",
// which produced the Djokovic regression: a single-camera pose summary is
// nowhere near enough signal to override what the player said about themselves.
// New contract: coach to the self-report, full stop. If pose numbers look
// inconsistent with the stated tier, assume camera geometry or a bad take.
const RECONCILE_RULE = `RECONCILE RULE: The self-reported tier is a CONTRACT. Coach to it. Do NOT downgrade the player mid-response because the pose data looks rougher than the stated tier. Single-camera pose estimates are noisy — a weird elbow angle or scattered hip rotation is almost always camera geometry, occlusion, or a single off-take, NOT evidence that the player misreported their level. Trust the self-report. If the numbers surprise you, assume the camera is at fault, pick the cleanest frames to anchor on, and coach them at the tier they told you they're at. Never imply they overestimated themselves. Never phrase feedback like "let's lock in the basics first" unless their tier is beginner. Meet them exactly where they said they are.`

const TIER_RULES: Record<SkillTier, string> = {
  beginner: `TIER: Beginner (new to tennis). One foundation cue per analysis. Lead with what's working. Never list three problems — they'll quit. Use physical feel cues only.`,
  intermediate: `TIER: Intermediate (rallies consistently). 2–3 tips, mix of foundations and refinements. Encouraging tone. Assume they rally consistently but still need cues on timing and rotation.`,
  competitive: `TIER: Competitive (match/tournament player). Standard 3-tip structure. Assume they know the fundamentals. Focus on execution and refinement, not foundations.`,
  advanced: `TIER: Advanced (top club / college / pro). Default to 'this is clean, keep grooving it' — one micro-refinement max, or nothing to fix. Never invent problems. If the swing is solid, say so and stop. Point them toward saving this as a baseline for drift detection.`,
}

// Trailing instruction appended to every prompt branch so we can collect
// telemetry on what tier the LLM actually thought the swing was, even when the
// defanged RECONCILE_RULE prevents it from acting on that belief. The server
// parses and strips this line before the client sees the response.
const TIER_ASSESSMENT_TRAILER = `At the very END of your response, on its own final line, emit EXACTLY:
[TIER_ASSESSMENT: <beginner|intermediate|competitive|advanced|unknown>]
Do not explain this line. The server parses and strips it before the player sees it.`

// Exported so the analyze routes can parse + strip the trailer after the
// stream buffers. Unanchored (global) + case-insensitive — if the LLM appends
// a period/emoji/extra paragraph after the tag, we still match + strip it
// rather than leaking the raw `[TIER_ASSESSMENT: ...]` line to the user.
export const TIER_ASSESSMENT_REGEX =
  /\[TIER_ASSESSMENT:\s*(beginner|intermediate|competitive|advanced|unknown)\s*\]/gi

// Parse the trailer out of a buffered response. Returns the assessed tier
// (lowercase) and the response with the trailer stripped. If no trailer is
// found, returns null + the original text. We also strip any trailing blank
// lines left behind after the trailer is removed so the coaching text doesn't
// look truncated. Uses the LAST match in the buffer — the model occasionally
// quotes the format earlier in coaching, and we only care about the real tag.
export function parseTierAssessmentTrailer(buffered: string): {
  assessedTier: 'beginner' | 'intermediate' | 'competitive' | 'advanced' | 'unknown' | null
  stripped: string
} {
  const matches = Array.from(buffered.matchAll(TIER_ASSESSMENT_REGEX))
  if (matches.length === 0) {
    return { assessedTier: null, stripped: buffered }
  }
  const last = matches[matches.length - 1]
  const tier = last[1].toLowerCase() as
    | 'beginner'
    | 'intermediate'
    | 'competitive'
    | 'advanced'
    | 'unknown'
  // Strip EVERY trailer instance (defensive — the model occasionally emits
  // more than one), then clean up trailing whitespace.
  const stripped = buffered.replace(TIER_ASSESSMENT_REGEX, '').replace(/\s+$/, '')
  return { assessedTier: tier, stripped }
}

// Ordering used to decide whether the LLM's assessed tier represents a
// downgrade from the self-reported tier. 'unknown' is not a real tier; callers
// treat a null assessed tier as "no downgrade detected".
const TIER_RANK: Record<SkillTier, number> = {
  beginner: 0,
  intermediate: 1,
  competitive: 2,
  advanced: 3,
}

export function tierRank(tier: SkillTier): number {
  return TIER_RANK[tier]
}

// True when the LLM's assessed tier is strictly lower than the tier we coached
// to. Returns false whenever either side is null/unknown — missing data is not
// a downgrade signal.
export function isTierDowngrade(
  coachedTier: SkillTier | null,
  assessedTier: 'beginner' | 'intermediate' | 'competitive' | 'advanced' | 'unknown' | null,
): boolean {
  if (!coachedTier || !assessedTier || assessedTier === 'unknown') return false
  return TIER_RANK[assessedTier] < TIER_RANK[coachedTier]
}

// Fallback block when no profile is available (legacy users, anonymous
// sessions). Keeps the prompt coherent without assuming a tier — mirrors the
// old SKILL_CALIBRATION behavior. Trailer instruction appended so we still
// capture the assessed tier for anon sessions.
const GENERIC_CALIBRATION = `READ THE SKILL LEVEL FIRST:
Before giving any advice, look at the joint angles and phase timing. Judge how refined this swing already is.
- Polished / near-pro mechanics: give SUBTLE refinements, not rebuilds.
- Solid intermediate: point to the 2 or 3 biggest gaps and give practical drills.
- Still developing: focus on foundations, and pick the ONE thing that unlocks everything else.
Match the advice to what the swing actually needs. Never default to generic cues.

${TIER_ASSESSMENT_TRAILER}`

// Returns the prompt fragment that tier-specific coaching is built from.
// Designed to be interpolated into the route prompts — contains the tier
// rule, reconcile rule, handedness context, and goal weighting.
export function buildTierCoachingBlock(profile: UserProfile | null): string {
  if (!profile) return GENERIC_CALIBRATION

  const tierRule = TIER_RULES[profile.skill_tier]
  const handednessLabel = profile.dominant_hand === 'right' ? 'right-handed' : 'left-handed'
  // Dominant-side references are what the LLM uses to pick which joints to
  // name in its feedback; inverted for lefties.
  const dominantSide = profile.dominant_hand === 'right' ? 'right' : 'left'
  const nonDominantSide = profile.dominant_hand === 'right' ? 'left' : 'right'
  const backhandLabel =
    profile.backhand_style === 'one_handed' ? 'one-handed backhand' : 'two-handed backhand'
  const goalLabel = describeGoal(profile)

  return `${tierRule}

${RECONCILE_RULE}

PLAYER CONTEXT:
- Handedness: ${handednessLabel}. Their dominant arm is the ${dominantSide} one; reference "${dominantSide} shoulder / elbow / wrist" for swing-side cues and "${nonDominantSide}" for the off-hand.
- Backhand style: ${backhandLabel}. When discussing the backhand side, tailor cues to this grip.

GOAL WEIGHTING:
- The player's stated priority is: ${goalLabel}. Prioritize observations that move the needle on this goal and mention it explicitly in at least one cue.

${TIER_ASSESSMENT_TRAILER}`
}

// Prompt block for users who skipped onboarding. The LLM is told there is no
// self-reported tier, asked to classify the swing from the data, name its
// inferred tier in a short italic header, then coach using that tier's rules.
//
// Unlike the self-reported path (where RECONCILE_RULE is defanged into a
// "coach to the contract" instruction), the skipped path keeps the three-
// signal override gate: the LLM is allowed to shift tier mid-response only if
// multiple independent signals agree the swing is different from the initial
// guess. Without a self-report, there is no contract to preserve.
export function buildInferredTierCoachingBlock(): string {
  return `INFERRED TIER MODE: The player declined to self-report their skill level. Classify their swing from the joint angle and phase data into ONE of these four tiers, then coach to that tier:

- beginner: wild inconsistency across frames, arming the ball with little trunk rotation, phase timing scattered, no clear kinetic chain.
- intermediate: rallies consistently, recognizable stroke shape, but timing and rotation still need work.
- competitive: solid fundamentals, clean kinetic chain, mostly about execution polish and matchplay refinements.
- advanced: clean mechanics end to end, only micro-refinements left — groove the baseline rather than rebuild anything.

NAME YOUR INFERRED TIER: At the very top of your response, on its own line, emit the tier you picked in italic parentheses. Keep it short and non-intrusive, for example:
*(coaching you as intermediate — set your profile to recalibrate)*
Do this once, then move straight into the normal coaching sections.

TIER RULES (use the one matching the tier you picked):
- beginner: ${TIER_RULES.beginner}
- intermediate: ${TIER_RULES.intermediate}
- competitive: ${TIER_RULES.competitive}
- advanced: ${TIER_RULES.advanced}

RECONCILE RULE (three-signal override): Because there is no self-report to anchor on, you are allowed to shift your inferred tier mid-response if, and only if, at least THREE independent signals agree with the shift. Examples of independent signals: kinetic chain sequencing, trunk rotation magnitude, contact-point consistency across frames, racket-drop depth, phase-timing regularity. One weird number is not a signal. One off-frame is not a signal. If you shift, be explicit: "updating my read to <tier>" and continue from there. If only one or two signals disagree with your first guess, assume noise and stay the course.

${TIER_ASSESSMENT_TRAILER}`
}

// One-shot fetch of everything the coaching routes need from auth: the parsed
// profile plus whether the user explicitly skipped onboarding. Keeps the
// routes to a single getUser() round trip instead of calling getProfile()
// and a separate skipped-check.
export async function getCoachingContext(
  client: SupabaseLikeClient,
): Promise<{ profile: UserProfile | null; skipped: boolean }> {
  try {
    const { data } = await client.auth.getUser()
    const metadata = data.user?.user_metadata ?? null
    return {
      profile: parseProfile(metadata),
      skipped: wasSkipped(metadata),
    }
  } catch {
    return { profile: null, skipped: false }
  }
}
