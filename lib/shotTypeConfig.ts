import type { JointGroup } from './jointAngles'
import type { JointAngles } from './supabase'
import type { SwingPhase } from './syncAlignment'

/**
 * Shot-type-specific configuration for coaching analysis.
 * Defines which joints to emphasize, key angles to inspect per phase,
 * kinetic chain ordering, and common mistakes for each stroke type.
 *
 * Angle conventions match computeJointAngles() in jointAngles.ts.
 */

// ---------------------------------------------------------------------------
// Canonical shot type lists (single source of truth for validation)
// ---------------------------------------------------------------------------

export const VALID_SHOT_TYPES = ['forehand', 'backhand', 'serve', 'volley', 'slice'] as const
export const VALID_USER_SHOT_TYPES = [...VALID_SHOT_TYPES, 'unknown'] as const
export type ShotType = (typeof VALID_SHOT_TYPES)[number]
export type UserShotType = (typeof VALID_USER_SHOT_TYPES)[number]

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A key angle to compare between user and pro at a specific swing phase. */
export interface KeyAngleSpec {
  label: string
  angleKey: keyof JointAngles
  phase: SwingPhase
  /** Ideal range in degrees [min, max] based on biomechanics literature. */
  idealRange: [number, number]
  /** One-sentence coaching cue if the player is outside the range. */
  coachingCue: string
}

/** A detectable mistake. `detect` receives the angles at the relevant phase. */
export interface MistakeCheck {
  id: string
  label: string
  phase: SwingPhase
  detect: (angles: JointAngles) => boolean
  tip: string
}

export interface ShotTypeConfig {
  /** Human-readable label */
  label: string
  /** Joint groups to highlight in the pose overlay (primary = most important) */
  emphasizedJoints: JointGroup[]
  /** Secondary joints - shown but not as critical */
  secondaryJoints: JointGroup[]
  /** Keys from JointAngles to focus on during comparison */
  keyAngles: (keyof JointAngles)[]
  /** Phase-specific angle specs with ideal ranges and coaching cues */
  keyAngleSpecs: KeyAngleSpec[]
  /** Kinetic chain firing order (angle keys that should peak sequentially) */
  kineticChainOrder: (keyof JointAngles)[]
  /** Common technique mistakes detectable via pose data */
  commonMistakes: string[]
  /** Machine-detectable mistakes with threshold functions */
  mistakeChecks: MistakeCheck[]
}

// ---------------------------------------------------------------------------
// Deviation scoring
// ---------------------------------------------------------------------------

export type DeviationLevel = 'good' | 'moderate' | 'poor' | 'unknown'

/** Score user angle vs pro angle. Green <=15 deg, yellow <=30, red >30. */
export function scoreAngleDeviation(
  userAngle: number | undefined,
  proAngle: number | undefined
): { level: DeviationLevel; delta: number | null } {
  if (userAngle == null || proAngle == null) return { level: 'unknown' as const, delta: null }
  const delta = Math.abs(userAngle - proAngle)
  if (delta <= 15) return { level: 'good', delta }
  if (delta <= 30) return { level: 'moderate', delta }
  return { level: 'poor', delta }
}

/** Score user angle vs an ideal range. */
export function scoreAngleVsIdeal(
  angle: number | undefined,
  idealRange: [number, number]
): { level: DeviationLevel; delta: number | null } {
  if (angle == null) return { level: 'unknown' as const, delta: null }
  const [min, max] = idealRange
  if (angle >= min && angle <= max) return { level: 'good', delta: 0 }
  const delta = angle < min ? min - angle : angle - max
  if (delta <= 15) return { level: 'moderate', delta }
  return { level: 'poor', delta }
}

// ---------------------------------------------------------------------------
// Configs
// ---------------------------------------------------------------------------

export const SHOT_TYPE_CONFIGS: Record<string, ShotTypeConfig> = {
  forehand: {
    label: 'Forehand',
    emphasizedJoints: ['hips', 'shoulders', 'elbows', 'wrists'],
    secondaryJoints: ['knees', 'ankles'],
    keyAngles: [
      'right_elbow',
      'right_shoulder',
      'hip_rotation',
      'trunk_rotation',
      'right_knee',
    ],
    keyAngleSpecs: [
      {
        label: 'Hip Rotation',
        angleKey: 'hip_rotation',
        phase: 'backswing',
        idealRange: [45, 70],
        coachingCue: 'Turn your pocket toward the back fence - load those hips to store power.',
      },
      {
        label: 'Trunk Rotation',
        angleKey: 'trunk_rotation',
        phase: 'backswing',
        idealRange: [60, 110],
        coachingCue: 'Show your back to the net - shoulders should turn farther than hips.',
      },
      {
        label: 'Knee Bend',
        angleKey: 'right_knee',
        phase: 'backswing',
        idealRange: [130, 160],
        coachingCue: 'Sit into the shot - load your legs like settling into a low chair.',
      },
      {
        label: 'Elbow at Contact',
        angleKey: 'right_elbow',
        phase: 'contact',
        idealRange: [100, 140],
        coachingCue: 'Keep the elbow soft at contact - not locked straight, not jammed in.',
      },
      {
        label: 'Shoulder at Contact',
        angleKey: 'right_shoulder',
        phase: 'contact',
        idealRange: [70, 110],
        coachingCue: 'Let your chest rotate through - the arm rides with the body turn.',
      },
    ],
    kineticChainOrder: ['hip_rotation', 'trunk_rotation', 'right_shoulder', 'right_elbow', 'right_wrist'],
    commonMistakes: [
      'Insufficient hip rotation - hips stay square to the net throughout the swing',
      'Arm-only swing - trunk rotation less than 15 degrees between loading and contact',
      'Locked knees during loading - knee angle exceeds 170 degrees',
      'Late contact point - elbow too bent (under 90 degrees) at contact',
      'No hip-shoulder separation - hip and trunk rotation nearly identical throughout backswing',
    ],
    mistakeChecks: [
      {
        id: 'arm_only',
        label: 'Arm-only swing',
        phase: 'forward_swing',
        detect: (a) => (a.trunk_rotation ?? 180) > 150,
        tip: 'Your shoulders barely rotate - lead with your hips and let the body sling the arm through.',
      },
      {
        id: 'locked_knees',
        label: 'Locked knees',
        phase: 'backswing',
        detect: (a) => (a.right_knee ?? 180) > 170,
        tip: 'Straight legs are dead legs. Bend and feel the ground push you into the shot.',
      },
      {
        id: 'cramped_elbow',
        label: 'Cramped elbow at contact',
        phase: 'contact',
        detect: (a) => (a.right_elbow ?? 180) < 90,
        tip: 'You\'re hitting too close to your body - reach out and meet the ball in front.',
      },
      {
        id: 'over_extended',
        label: 'Over-extended arm',
        phase: 'contact',
        detect: (a) => (a.right_elbow ?? 0) > 175,
        tip: 'Your arm is locked straight at contact - keep a soft bend for control and spin.',
      },
    ],
  },

  backhand: {
    label: 'Backhand',
    emphasizedJoints: ['hips', 'shoulders', 'elbows', 'wrists'],
    secondaryJoints: ['knees', 'ankles'],
    keyAngles: [
      'left_elbow',
      'left_shoulder',
      'hip_rotation',
      'trunk_rotation',
      'left_knee',
    ],
    keyAngleSpecs: [
      {
        label: 'Trunk Rotation',
        angleKey: 'trunk_rotation',
        phase: 'backswing',
        idealRange: [70, 100],
        coachingCue: 'Turn your dominant shoulder toward the net - full coil sets up the shot.',
      },
      {
        label: 'Elbow at Contact',
        angleKey: 'left_elbow',
        phase: 'contact',
        idealRange: [140, 175],
        coachingCue: 'Extend through the ball - reach out and let the arm uncoil naturally.',
      },
      {
        label: 'Knee Bend',
        angleKey: 'left_knee',
        phase: 'backswing',
        idealRange: [130, 155],
        coachingCue: 'Load your legs and push up through the ball.',
      },
      {
        label: 'Hip Rotation',
        angleKey: 'hip_rotation',
        phase: 'backswing',
        idealRange: [50, 75],
        coachingCue: 'Coil the hips - they should lead the forward swing.',
      },
    ],
    kineticChainOrder: ['hip_rotation', 'trunk_rotation', 'left_shoulder', 'left_elbow', 'left_wrist'],
    commonMistakes: [
      'Insufficient shoulder turn - trunk rotation under 45 degrees during preparation',
      'Over-extended arm at contact - elbow locked at 175+ degrees',
      'Poor weight transfer - hips remain static throughout the swing',
      'Wrist flexion at contact - increases tennis elbow risk',
      'No follow-through - trunk rotation stops at contact',
    ],
    mistakeChecks: [
      {
        id: 'wrist_collapse',
        label: 'Wrist collapse',
        phase: 'contact',
        detect: (a) => (a.left_elbow ?? 180) < 120,
        tip: 'Your arm is too bent at contact - reach through the ball with a firm wrist.',
      },
      {
        id: 'no_trunk_turn',
        label: 'Insufficient shoulder turn',
        phase: 'backswing',
        detect: (a) => (a.trunk_rotation ?? 180) > 160,
        tip: 'Turn your shoulders more - show your chest to the back fence during preparation.',
      },
    ],
  },

  serve: {
    label: 'Serve',
    emphasizedJoints: ['knees', 'shoulders', 'elbows'],
    secondaryJoints: ['hips', 'wrists', 'ankles'],
    keyAngles: [
      'right_elbow',
      'right_shoulder',
      'right_knee',
      'trunk_rotation',
      'hip_rotation',
    ],
    keyAngleSpecs: [
      {
        label: 'Knee Bend (Trophy)',
        angleKey: 'right_knee',
        phase: 'backswing',
        idealRange: [130, 155],
        coachingCue: 'Sit and explode - deep knee bend, then launch up like dunking a basketball.',
      },
      {
        label: 'Shoulder (Trophy)',
        angleKey: 'right_shoulder',
        phase: 'backswing',
        idealRange: [85, 110],
        coachingCue: 'Get the arm up to the trophy position - elbow at shoulder height.',
      },
      {
        label: 'Elbow at Contact',
        angleKey: 'right_elbow',
        phase: 'contact',
        idealRange: [150, 175],
        coachingCue: 'Reach for the sky at contact - extend fully at the highest point.',
      },
      {
        label: 'Trunk at Contact',
        angleKey: 'trunk_rotation',
        phase: 'contact',
        idealRange: [35, 60],
        coachingCue: 'Lean into the court and extend upward - land inside the baseline.',
      },
    ],
    kineticChainOrder: ['right_knee', 'hip_rotation', 'trunk_rotation', 'right_shoulder', 'right_elbow', 'right_wrist'],
    commonMistakes: [
      'Insufficient knee bend - knee angle exceeds 160 degrees at trophy position',
      'Low contact point - arm not fully extended at contact',
      'No leg drive - knees stay straight through the motion',
      'Rushing the toss - no separation between toss and swing',
      'Falling backward - weight not transferred into the court',
    ],
    mistakeChecks: [
      {
        id: 'straight_legs',
        label: 'Insufficient knee bend',
        phase: 'backswing',
        detect: (a) => (a.right_knee ?? 180) > 165,
        tip: 'Your legs barely bend - this costs half your serve power. Think "sit then explode."',
      },
      {
        id: 'low_contact',
        label: 'Arm not fully extended',
        phase: 'contact',
        detect: (a) => (a.right_elbow ?? 0) < 140,
        tip: 'You\'re hitting with a bent arm - reach up and extend fully at contact.',
      },
    ],
  },

  volley: {
    label: 'Volley',
    emphasizedJoints: ['wrists', 'elbows', 'shoulders'],
    secondaryJoints: ['knees', 'hips', 'ankles'],
    keyAngles: [
      'right_elbow',
      'right_shoulder',
      'right_knee',
      'trunk_rotation',
    ],
    keyAngleSpecs: [
      {
        label: 'Elbow (Preparation)',
        angleKey: 'right_elbow',
        phase: 'preparation',
        idealRange: [100, 140],
        coachingCue: 'Keep the racket in front - short backswing, firm wrist.',
      },
      {
        label: 'Knee Bend',
        angleKey: 'right_knee',
        phase: 'preparation',
        idealRange: [140, 165],
        coachingCue: 'Stay low and athletic - bend those knees for quick reactions.',
      },
    ],
    kineticChainOrder: ['hip_rotation', 'trunk_rotation', 'right_shoulder', 'right_elbow', 'right_wrist'],
    commonMistakes: [
      'Taking a full swing - volley should be a compact punch',
      'Wrist breakdown - wrist collapses on contact instead of staying firm',
      'Standing too upright - knees not flexed for quick reactions',
      'No split step - feet planted before the opponent strikes',
    ],
    mistakeChecks: [
      {
        id: 'big_backswing',
        label: 'Excessive backswing',
        phase: 'backswing',
        detect: (a) => (a.trunk_rotation ?? 0) > 60,
        tip: 'Too much backswing for a volley - keep it compact. Punch, don\'t swing.',
      },
    ],
  },

  slice: {
    label: 'Slice',
    emphasizedJoints: ['wrists', 'elbows', 'shoulders'],
    secondaryJoints: ['knees', 'hips'],
    keyAngles: [
      'right_elbow',
      'right_shoulder',
      'trunk_rotation',
      'right_knee',
    ],
    keyAngleSpecs: [
      {
        label: 'Elbow at Contact',
        angleKey: 'right_elbow',
        phase: 'contact' as SwingPhase,
        idealRange: [110, 150] as [number, number],
        coachingCue: 'Keep the arm relaxed through contact - carve under the ball with a smooth path.',
      },
      {
        label: 'Shoulder at Contact',
        angleKey: 'right_shoulder',
        phase: 'contact' as SwingPhase,
        idealRange: [60, 100] as [number, number],
        coachingCue: 'Stay sideways through the slice - your chest should face the sideline at contact.',
      },
      {
        label: 'Knee Bend',
        angleKey: 'right_knee',
        phase: 'preparation' as SwingPhase,
        idealRange: [135, 160] as [number, number],
        coachingCue: 'Get low and stay balanced - the slice needs a stable base.',
      },
    ],
    kineticChainOrder: ['hip_rotation', 'trunk_rotation', 'right_shoulder', 'right_elbow', 'right_wrist'],
    commonMistakes: [
      'Chopping down on the ball instead of carving through it',
      'Open racket face too early - slice floats with no bite',
      'No shoulder turn - arm-only slice lacks depth and control',
      'Standing too tall - no knee bend robs the shot of stability',
    ],
    mistakeChecks: [
      {
        id: 'no_shoulder_turn',
        label: 'No shoulder turn',
        phase: 'backswing' as SwingPhase,
        detect: (a: JointAngles) => (a.trunk_rotation ?? 180) > 160,
        tip: 'Turn your shoulders sideways - a good slice starts with a full unit turn.',
      },
      {
        id: 'locked_arm',
        label: 'Locked arm at contact',
        phase: 'contact' as SwingPhase,
        detect: (a: JointAngles) => (a.right_elbow ?? 0) > 170,
        tip: 'Your arm is too stiff - keep a soft bend to guide the racket under the ball.',
      },
    ],
  },
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

/**
 * Get the config for a given shot type. Falls back to forehand config
 * for unknown types.
 */
export function getShotTypeConfig(shotType: string | null | undefined): ShotTypeConfig {
  const normalized = (shotType ?? 'forehand').toLowerCase().trim()
  return SHOT_TYPE_CONFIGS[normalized] ?? SHOT_TYPE_CONFIGS.forehand
}

/**
 * Get the recommended joint visibility map for a given shot type.
 * Emphasized and secondary joints are visible; others are hidden.
 *
 * Note: the racket-head trail is ALWAYS on for every shot type
 * (forehand / backhand / serve / volley / slice). It lives on the separate
 * `showRacket` flag in the joint store (not in this JointGroup map), so this
 * function doesn't toggle it -- any shot-type "focus" UI should leave
 * `showRacket` untouched when calling `setVisibility(map)`.
 */
export function getRecommendedVisibility(
  shotType: string | null | undefined
): Record<JointGroup, boolean> {
  const config = getShotTypeConfig(shotType)
  const allGroups: JointGroup[] = ['shoulders', 'elbows', 'wrists', 'hips', 'knees', 'ankles']
  const result = {} as Record<JointGroup, boolean>
  for (const g of allGroups) {
    result[g] = config.emphasizedJoints.includes(g) || config.secondaryJoints.includes(g)
  }
  return result
}
