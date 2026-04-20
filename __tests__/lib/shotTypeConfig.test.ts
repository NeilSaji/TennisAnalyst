import { describe, it, expect } from 'vitest'
import {
  SHOT_TYPE_CONFIGS,
  VALID_SHOT_TYPES,
  VALID_USER_SHOT_TYPES,
  getShotTypeConfig,
  getRecommendedVisibility,
  scoreAngleDeviation,
  scoreAngleVsIdeal,
  type ShotTypeConfig,
} from '@/lib/shotTypeConfig'
import { JOINT_GROUPS } from '@/lib/jointAngles'
import type { JointGroup } from '@/lib/jointAngles'
import type { JointAngles } from '@/lib/supabase'

// All valid JointGroup keys
const VALID_JOINT_GROUPS = Object.keys(JOINT_GROUPS) as JointGroup[]

// All valid JointAngles keys
const VALID_ANGLE_KEYS: (keyof JointAngles)[] = [
  'right_elbow',
  'left_elbow',
  'right_shoulder',
  'left_shoulder',
  'right_wrist',
  'left_wrist',
  'right_knee',
  'left_knee',
  'right_hip',
  'left_hip',
  'hip_rotation',
  'trunk_rotation',
]

// ---------------------------------------------------------------------------
// SHOT_TYPE_CONFIGS structure
// ---------------------------------------------------------------------------

describe('VALID_SHOT_TYPES constants', () => {
  it('VALID_SHOT_TYPES contains all 5 shot types', () => {
    expect(VALID_SHOT_TYPES).toEqual(['forehand', 'backhand', 'serve', 'volley', 'slice'])
  })

  it('VALID_USER_SHOT_TYPES extends shot types with unknown', () => {
    expect(VALID_USER_SHOT_TYPES).toEqual(['forehand', 'backhand', 'serve', 'volley', 'slice', 'unknown'])
  })

  it('every VALID_SHOT_TYPE has a matching config', () => {
    for (const st of VALID_SHOT_TYPES) {
      expect(SHOT_TYPE_CONFIGS[st]).toBeDefined()
    }
  })
})

describe('SHOT_TYPE_CONFIGS', () => {
  it('has entries for forehand, backhand, serve, volley, and slice', () => {
    expect(SHOT_TYPE_CONFIGS).toHaveProperty('forehand')
    expect(SHOT_TYPE_CONFIGS).toHaveProperty('backhand')
    expect(SHOT_TYPE_CONFIGS).toHaveProperty('serve')
    expect(SHOT_TYPE_CONFIGS).toHaveProperty('volley')
    expect(SHOT_TYPE_CONFIGS).toHaveProperty('slice')
  })

  it.each(['forehand', 'backhand', 'serve', 'volley', 'slice'])(
    '%s config has all required fields',
    (shotType) => {
      const config = SHOT_TYPE_CONFIGS[shotType]
      expect(config.label).toBeDefined()
      expect(typeof config.label).toBe('string')
      expect(config.label.length).toBeGreaterThan(0)

      expect(Array.isArray(config.emphasizedJoints)).toBe(true)
      expect(config.emphasizedJoints.length).toBeGreaterThan(0)

      expect(Array.isArray(config.secondaryJoints)).toBe(true)

      expect(Array.isArray(config.keyAngles)).toBe(true)
      expect(config.keyAngles.length).toBeGreaterThan(0)

      expect(Array.isArray(config.keyAngleSpecs)).toBe(true)
      expect(config.keyAngleSpecs.length).toBeGreaterThan(0)

      expect(Array.isArray(config.kineticChainOrder)).toBe(true)
      expect(config.kineticChainOrder.length).toBeGreaterThan(0)

      expect(Array.isArray(config.commonMistakes)).toBe(true)
      expect(config.commonMistakes.length).toBeGreaterThan(0)

      expect(Array.isArray(config.mistakeChecks)).toBe(true)
    }
  )

  it.each(['forehand', 'backhand', 'serve', 'volley', 'slice'])(
    '%s emphasizedJoints are all valid JointGroup values',
    (shotType) => {
      const config = SHOT_TYPE_CONFIGS[shotType]
      for (const joint of config.emphasizedJoints) {
        expect(VALID_JOINT_GROUPS).toContain(joint)
      }
    }
  )

  it.each(['forehand', 'backhand', 'serve', 'volley', 'slice'])(
    '%s secondaryJoints are all valid JointGroup values',
    (shotType) => {
      const config = SHOT_TYPE_CONFIGS[shotType]
      for (const joint of config.secondaryJoints) {
        expect(VALID_JOINT_GROUPS).toContain(joint)
      }
    }
  )

  it.each(['forehand', 'backhand', 'serve', 'volley', 'slice'])(
    '%s emphasized and secondary joints do not overlap',
    (shotType) => {
      const config = SHOT_TYPE_CONFIGS[shotType]
      const overlap = config.emphasizedJoints.filter((j) =>
        config.secondaryJoints.includes(j)
      )
      expect(overlap).toHaveLength(0)
    }
  )

  it.each(['forehand', 'backhand', 'serve', 'volley', 'slice'])(
    '%s keyAngles are all valid JointAngles keys',
    (shotType) => {
      const config = SHOT_TYPE_CONFIGS[shotType]
      for (const angleKey of config.keyAngles) {
        expect(VALID_ANGLE_KEYS).toContain(angleKey)
      }
    }
  )

  it.each(['forehand', 'backhand', 'serve', 'volley', 'slice'])(
    '%s kineticChainOrder elements are valid JointAngles keys',
    (shotType) => {
      const config = SHOT_TYPE_CONFIGS[shotType]
      for (const angleKey of config.kineticChainOrder) {
        expect(VALID_ANGLE_KEYS).toContain(angleKey)
      }
    }
  )

  it.each(['forehand', 'backhand', 'serve', 'volley', 'slice'])(
    '%s commonMistakes are non-empty strings',
    (shotType) => {
      const config = SHOT_TYPE_CONFIGS[shotType]
      for (const mistake of config.commonMistakes) {
        expect(typeof mistake).toBe('string')
        expect(mistake.length).toBeGreaterThan(0)
      }
    }
  )

  it('forehand config emphasizes hip rotation and trunk rotation', () => {
    const config = SHOT_TYPE_CONFIGS.forehand
    expect(config.keyAngles).toContain('hip_rotation')
    expect(config.keyAngles).toContain('trunk_rotation')
  })

  it('serve config emphasizes knee angle for leg drive', () => {
    const config = SHOT_TYPE_CONFIGS.serve
    expect(config.keyAngles).toContain('right_knee')
  })

  it('forehand kinetic chain starts with hips and ends with wrist', () => {
    const chain = SHOT_TYPE_CONFIGS.forehand.kineticChainOrder
    expect(chain[0]).toBe('hip_rotation')
    expect(chain[chain.length - 1]).toBe('right_wrist')
    // Elbow must still precede the wrist link (proximal-to-distal).
    expect(chain[chain.length - 2]).toBe('right_elbow')
  })
})

// ---------------------------------------------------------------------------
// keyAngleSpecs
// ---------------------------------------------------------------------------

describe('keyAngleSpecs', () => {
  it.each(['forehand', 'backhand', 'serve', 'volley', 'slice'])(
    '%s specs have valid angle keys and ideal ranges',
    (shotType) => {
      const config = SHOT_TYPE_CONFIGS[shotType]
      for (const spec of config.keyAngleSpecs) {
        expect(VALID_ANGLE_KEYS).toContain(spec.angleKey)
        expect(spec.idealRange).toHaveLength(2)
        expect(spec.idealRange[0]).toBeLessThanOrEqual(spec.idealRange[1])
        expect(spec.label.length).toBeGreaterThan(0)
        expect(spec.coachingCue.length).toBeGreaterThan(0)
      }
    }
  )

  it('forehand has specs for both backswing and contact phases', () => {
    const phases = SHOT_TYPE_CONFIGS.forehand.keyAngleSpecs.map((s) => s.phase)
    expect(phases).toContain('backswing')
    expect(phases).toContain('contact')
  })
})

// ---------------------------------------------------------------------------
// mistakeChecks
// ---------------------------------------------------------------------------

describe('mistakeChecks', () => {
  it.each(['forehand', 'backhand', 'serve', 'volley', 'slice'])(
    '%s mistake checks have valid structure',
    (shotType) => {
      const config = SHOT_TYPE_CONFIGS[shotType]
      for (const check of config.mistakeChecks) {
        expect(check.id.length).toBeGreaterThan(0)
        expect(check.label.length).toBeGreaterThan(0)
        expect(check.tip.length).toBeGreaterThan(0)
        expect(typeof check.detect).toBe('function')
      }
    }
  )

  it('forehand locked_knees check triggers when knee > 170', () => {
    const check = SHOT_TYPE_CONFIGS.forehand.mistakeChecks.find(
      (c) => c.id === 'locked_knees'
    )
    expect(check).toBeDefined()
    expect(check!.detect({ right_knee: 175 })).toBe(true)
    expect(check!.detect({ right_knee: 150 })).toBe(false)
  })

  it('forehand cramped_elbow check triggers when elbow < 90', () => {
    const check = SHOT_TYPE_CONFIGS.forehand.mistakeChecks.find(
      (c) => c.id === 'cramped_elbow'
    )
    expect(check).toBeDefined()
    expect(check!.detect({ right_elbow: 80 })).toBe(true)
    expect(check!.detect({ right_elbow: 110 })).toBe(false)
  })

  it('forehand over_extended check triggers when elbow > 175', () => {
    const check = SHOT_TYPE_CONFIGS.forehand.mistakeChecks.find(
      (c) => c.id === 'over_extended'
    )
    expect(check).toBeDefined()
    expect(check!.detect({ right_elbow: 178 })).toBe(true)
    expect(check!.detect({ right_elbow: 140 })).toBe(false)
  })

  it('serve straight_legs check triggers when knee > 165', () => {
    const check = SHOT_TYPE_CONFIGS.serve.mistakeChecks.find(
      (c) => c.id === 'straight_legs'
    )
    expect(check).toBeDefined()
    expect(check!.detect({ right_knee: 170 })).toBe(true)
    expect(check!.detect({ right_knee: 140 })).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// getShotTypeConfig
// ---------------------------------------------------------------------------

describe('getShotTypeConfig', () => {
  it('returns the correct config for known shot types', () => {
    expect(getShotTypeConfig('forehand')).toBe(SHOT_TYPE_CONFIGS.forehand)
    expect(getShotTypeConfig('backhand')).toBe(SHOT_TYPE_CONFIGS.backhand)
    expect(getShotTypeConfig('serve')).toBe(SHOT_TYPE_CONFIGS.serve)
    expect(getShotTypeConfig('volley')).toBe(SHOT_TYPE_CONFIGS.volley)
  })

  it('normalizes case and whitespace', () => {
    expect(getShotTypeConfig('Forehand')).toBe(SHOT_TYPE_CONFIGS.forehand)
    expect(getShotTypeConfig('SERVE')).toBe(SHOT_TYPE_CONFIGS.serve)
    expect(getShotTypeConfig('  backhand  ')).toBe(SHOT_TYPE_CONFIGS.backhand)
  })

  it('falls back to forehand config for unknown shot types', () => {
    expect(getShotTypeConfig('dropshot')).toBe(SHOT_TYPE_CONFIGS.forehand)
    expect(getShotTypeConfig('lob')).toBe(SHOT_TYPE_CONFIGS.forehand)
  })

  it('handles null and undefined by falling back to forehand', () => {
    expect(getShotTypeConfig(null)).toBe(SHOT_TYPE_CONFIGS.forehand)
    expect(getShotTypeConfig(undefined)).toBe(SHOT_TYPE_CONFIGS.forehand)
  })

  it('returns a valid ShotTypeConfig for any input', () => {
    const config = getShotTypeConfig('unknown_shot_type')
    expect(config.label).toBeDefined()
    expect(config.emphasizedJoints.length).toBeGreaterThan(0)
    expect(config.keyAngles.length).toBeGreaterThan(0)
    expect(config.commonMistakes.length).toBeGreaterThan(0)
  })
})

// ---------------------------------------------------------------------------
// getRecommendedVisibility
// ---------------------------------------------------------------------------

describe('getRecommendedVisibility', () => {
  it('returns all joint groups as keys', () => {
    const vis = getRecommendedVisibility('forehand')
    for (const group of VALID_JOINT_GROUPS) {
      expect(group in vis).toBe(true)
    }
  })

  it('marks emphasized and secondary joints as visible', () => {
    const config = SHOT_TYPE_CONFIGS.forehand
    const vis = getRecommendedVisibility('forehand')

    for (const joint of config.emphasizedJoints) {
      expect(vis[joint]).toBe(true)
    }
    for (const joint of config.secondaryJoints) {
      expect(vis[joint]).toBe(true)
    }
  })

  it('handles null/undefined shot type', () => {
    const vis = getRecommendedVisibility(null)
    // Should fall back to forehand
    const config = SHOT_TYPE_CONFIGS.forehand
    for (const joint of config.emphasizedJoints) {
      expect(vis[joint]).toBe(true)
    }
  })
})

// ---------------------------------------------------------------------------
// scoreAngleDeviation
// ---------------------------------------------------------------------------

describe('scoreAngleDeviation', () => {
  it('returns good for difference <= 15 degrees', () => {
    expect(scoreAngleDeviation(100, 110)).toEqual({ level: 'good', delta: 10 })
    expect(scoreAngleDeviation(90, 90)).toEqual({ level: 'good', delta: 0 })
    expect(scoreAngleDeviation(100, 115)).toEqual({ level: 'good', delta: 15 })
  })

  it('returns moderate for difference 16-30 degrees', () => {
    expect(scoreAngleDeviation(100, 125)).toEqual({ level: 'moderate', delta: 25 })
    expect(scoreAngleDeviation(100, 130)).toEqual({ level: 'moderate', delta: 30 })
  })

  it('returns poor for difference > 30 degrees', () => {
    expect(scoreAngleDeviation(100, 140)).toEqual({ level: 'poor', delta: 40 })
    expect(scoreAngleDeviation(50, 100)).toEqual({ level: 'poor', delta: 50 })
  })

  it('returns unknown with null delta when either angle is undefined', () => {
    expect(scoreAngleDeviation(undefined, 100)).toEqual({ level: 'unknown', delta: null })
    expect(scoreAngleDeviation(100, undefined)).toEqual({ level: 'unknown', delta: null })
    expect(scoreAngleDeviation(undefined, undefined)).toEqual({ level: 'unknown', delta: null })
  })
})

// ---------------------------------------------------------------------------
// scoreAngleVsIdeal
// ---------------------------------------------------------------------------

describe('scoreAngleVsIdeal', () => {
  it('returns good with delta 0 when angle is within ideal range', () => {
    expect(scoreAngleVsIdeal(120, [100, 140])).toEqual({ level: 'good', delta: 0 })
    expect(scoreAngleVsIdeal(100, [100, 140])).toEqual({ level: 'good', delta: 0 })
    expect(scoreAngleVsIdeal(140, [100, 140])).toEqual({ level: 'good', delta: 0 })
  })

  it('returns moderate when angle is outside range by <= 15 degrees', () => {
    expect(scoreAngleVsIdeal(90, [100, 140])).toEqual({ level: 'moderate', delta: 10 })
    expect(scoreAngleVsIdeal(150, [100, 140])).toEqual({ level: 'moderate', delta: 10 })
  })

  it('returns poor when angle is outside range by > 15 degrees', () => {
    expect(scoreAngleVsIdeal(70, [100, 140])).toEqual({ level: 'poor', delta: 30 })
    expect(scoreAngleVsIdeal(170, [100, 140])).toEqual({ level: 'poor', delta: 30 })
  })

  it('returns unknown with null delta when angle is undefined', () => {
    expect(scoreAngleVsIdeal(undefined, [100, 140])).toEqual({ level: 'unknown', delta: null })
  })
})
