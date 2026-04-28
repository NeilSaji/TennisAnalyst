import { describe, it, expect } from 'vitest'
import { classifyCameraAngle } from '@/lib/cameraAngle'
import { LANDMARK_INDICES } from '@/lib/jointAngles'
import { makeLandmark } from '../helpers'
import type { Landmark } from '@/lib/supabase'

// ---------------------------------------------------------------------------
// classifyCameraAngle
// ---------------------------------------------------------------------------
//
// Synthetic landmark fixtures: each test only populates the four landmarks
// the classifier inspects (both shoulders + both hips) plus whatever else
// we want to sanity-check. The classifier ignores other landmarks.

function makeShouldersAndHips(opts: {
  lShoulderX: number
  lShoulderVis?: number
  rShoulderX: number
  rShoulderVis?: number
  lHipX?: number
  lHipVis?: number
  rHipX?: number
  rHipVis?: number
  // Optional y override for the shoulders / hips. Defaults to a natural
  // standing pose layout (shoulders ~0.25, hips ~0.65).
  shoulderY?: number
  hipY?: number
}): Landmark[] {
  const {
    lShoulderX,
    rShoulderX,
    lShoulderVis = 0.95,
    rShoulderVis = 0.95,
    lHipX = lShoulderX,
    rHipX = rShoulderX,
    lHipVis = 0.95,
    rHipVis = 0.95,
    shoulderY = 0.25,
    hipY = 0.65,
  } = opts

  return [
    makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, lShoulderX, shoulderY, lShoulderVis),
    makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, rShoulderX, shoulderY, rShoulderVis),
    makeLandmark(LANDMARK_INDICES.LEFT_HIP, lHipX, hipY, lHipVis),
    makeLandmark(LANDMARK_INDICES.RIGHT_HIP, rHipX, hipY, rHipVis),
  ]
}

describe('classifyCameraAngle', () => {
  it('returns "unknown" for empty landmarks', () => {
    expect(classifyCameraAngle([])).toBe('unknown')
  })

  it('returns "unknown" when one shoulder is missing', () => {
    // Only left shoulder + both hips — no right shoulder at all.
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.45, 0.25, 0.95),
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.47, 0.65, 0.95),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.53, 0.65, 0.95),
    ]
    expect(classifyCameraAngle(landmarks)).toBe('unknown')
  })

  it('returns "unknown" when one shoulder visibility is below 0.5', () => {
    const landmarks = makeShouldersAndHips({
      lShoulderX: 0.4,
      lShoulderVis: 0.3, // below VIS_FLOOR
      rShoulderX: 0.6,
      rShoulderVis: 0.95,
    })
    expect(classifyCameraAngle(landmarks)).toBe('unknown')
  })

  it('returns "unknown" when one hip visibility is below 0.5', () => {
    const landmarks = makeShouldersAndHips({
      lShoulderX: 0.4,
      rShoulderX: 0.6,
      lHipVis: 0.95,
      rHipVis: 0.2, // below VIS_FLOOR
    })
    expect(classifyCameraAngle(landmarks)).toBe('unknown')
  })

  it('returns "side-on" when one shoulder occludes the other (asymmetric visibility)', () => {
    // Side-on filming: near shoulder confidently tracked, far shoulder
    // hidden behind the torso so MediaPipe gives it a much lower
    // visibility. The two shoulders project to nearly the same x.
    const landmarks = makeShouldersAndHips({
      lShoulderX: 0.49,
      lShoulderVis: 0.95,
      rShoulderX: 0.51,
      rShoulderVis: 0.5, // big asymmetry vs left, but >= VIS_FLOOR
      lHipX: 0.48,
      rHipX: 0.52,
    })
    expect(classifyCameraAngle(landmarks)).toBe('side-on')
  })

  it('returns "side-on" when shoulders project to nearly the same x (narrow spread)', () => {
    // Both shoulders symmetrically visible (rare but possible if the
    // far shoulder is just barely peeking through), but the x-spread is
    // tiny — clearly a profile shot.
    const landmarks = makeShouldersAndHips({
      lShoulderX: 0.49,
      lShoulderVis: 0.9,
      rShoulderX: 0.51,
      rShoulderVis: 0.9,
    })
    // Spread = 0.02, below SIDE_ON_MAX_SHOULDER_SPREAD (0.06).
    expect(classifyCameraAngle(landmarks)).toBe('side-on')
  })

  it('returns "front-on" with wide shoulder spread and symmetric visibility', () => {
    // Player squarely facing the camera: shoulders splay wide, both
    // tracked confidently.
    const landmarks = makeShouldersAndHips({
      lShoulderX: 0.35,
      lShoulderVis: 0.95,
      rShoulderX: 0.65,
      rShoulderVis: 0.95,
      lHipX: 0.42,
      rHipX: 0.58,
    })
    // Spread = 0.30, above FRONT_ON_MIN_SHOULDER_SPREAD (0.18).
    // Asymmetry = 0.0, below FRONT_ON_MAX_VIS_ASYMMETRY (0.2).
    expect(classifyCameraAngle(landmarks)).toBe('front-on')
  })

  it('returns "oblique" for a three-quarter view (medium spread, both visible)', () => {
    // Player turned ~45 degrees: shoulders separated but not as wide as
    // a true front-on, both still tracked confidently.
    const landmarks = makeShouldersAndHips({
      lShoulderX: 0.44,
      lShoulderVis: 0.9,
      rShoulderX: 0.56,
      rShoulderVis: 0.9,
      lHipX: 0.46,
      rHipX: 0.54,
    })
    // Spread = 0.12 — not narrow (>0.06), not wide (<0.18). Asymmetry
    // is tiny, so it's not flagged as side-on either. -> oblique.
    expect(classifyCameraAngle(landmarks)).toBe('oblique')
  })

  it('returns "oblique" when wide spread but visibility is moderately asymmetric', () => {
    // Wide spread but the asymmetry rules out a clean front-on call.
    const landmarks = makeShouldersAndHips({
      lShoulderX: 0.32,
      lShoulderVis: 0.95,
      rShoulderX: 0.62,
      rShoulderVis: 0.65, // asymmetry 0.30 -> above FRONT_ON_MAX_VIS_ASYMMETRY (0.2)
      lHipX: 0.42,
      rHipX: 0.58,
    })
    // Asymmetry 0.30 is also below the side-on threshold (0.4), and the
    // spread is well above the side-on cap. -> oblique (the in-between).
    expect(classifyCameraAngle(landmarks)).toBe('oblique')
  })

  it('classifies an arms-forward edge case (shoulders identical x, both highly visible) as "side-on"', () => {
    // Documented behavior of the edge case called out in the spec: a
    // player facing the camera with arms held directly forward can in
    // theory produce a near-zero shoulder x-spread with both shoulders
    // confidently tracked. Geometrically this is "front-on" but with a
    // collapsed shoulder line. We choose to classify this as 'side-on'
    // because the shoulder-spread heuristic dominates — this trades a
    // rare false-side-on for a far more common false-front-on (a real
    // profile shot would be wrongly let through if we relaxed it).
    const landmarks = makeShouldersAndHips({
      lShoulderX: 0.5,
      lShoulderVis: 0.95,
      rShoulderX: 0.5,
      rShoulderVis: 0.95,
      lHipX: 0.45,
      rHipX: 0.55,
    })
    expect(classifyCameraAngle(landmarks)).toBe('side-on')
  })
})
