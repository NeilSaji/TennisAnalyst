import { describe, it, expect } from 'vitest'
import {
  analyzeKineticChain,
  FOREHAND_KINETIC_CHAIN,
  getChainTimings,
  type KineticChainResult,
} from '@/lib/kineticChain'
import type { PoseFrame, JointAngles } from '@/lib/supabase'
import { makeFrame, makeStandingPose } from '../helpers'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build frames with angles that produce peak velocities in a controlled order.
 * Each segment has a spike at a specific frame to test sequence detection.
 */
function makeChainFrames(
  peakOrder: { angleKey: keyof JointAngles; peakAtFrame: number }[],
  totalFrames: number,
  intervalMs = 33
): PoseFrame[] {
  const frames: PoseFrame[] = []
  for (let i = 0; i < totalFrames; i++) {
    const angles: JointAngles = {
      hip_rotation: 10,
      trunk_rotation: 10,
      right_shoulder: 45,
      right_elbow: 120,
      right_knee: 170,
      left_knee: 170,
    }

    // For each segment, create a large angle change at the specified frame
    for (const { angleKey, peakAtFrame } of peakOrder) {
      if (i === peakAtFrame) {
        // Big spike: add 60 degrees in one frame
        angles[angleKey] = (angles[angleKey] ?? 0) + 60
      } else if (i === peakAtFrame + 1) {
        // Return to baseline after spike
        angles[angleKey] = (angles[angleKey] ?? 0)
      }
    }

    frames.push(makeFrame(i, i * intervalMs, makeStandingPose(), angles))
  }
  return frames
}

/**
 * Build frames that simulate a realistic forehand kinetic chain.
 * Each segment peaks at progressively later times.
 */
function makeCorrectSequenceFrames(totalFrames = 40): PoseFrame[] {
  const frames: PoseFrame[] = []
  for (let i = 0; i < totalFrames; i++) {
    const t = i / (totalFrames - 1) // 0..1
    const ts = i * 33

    // Simulate proper chain: hips peak around t=0.3, trunk t=0.4,
    // shoulder t=0.5, elbow t=0.6
    const hipRate = Math.exp(-((t - 0.3) ** 2) / 0.01) * 50
    const trunkRate = Math.exp(-((t - 0.4) ** 2) / 0.01) * 50
    const shoulderRate = Math.exp(-((t - 0.5) ** 2) / 0.01) * 50
    const elbowRate = Math.exp(-((t - 0.6) ** 2) / 0.01) * 50

    // Integrate rates to get angles
    let hipAngle = 10
    let trunkAngle = 10
    let shoulderAngle = 45
    let elbowAngle = 120

    for (let j = 0; j <= i; j++) {
      const jt = j / (totalFrames - 1)
      hipAngle += Math.exp(-((jt - 0.3) ** 2) / 0.01) * 2
      trunkAngle += Math.exp(-((jt - 0.4) ** 2) / 0.01) * 2
      shoulderAngle += Math.exp(-((jt - 0.5) ** 2) / 0.01) * 2
      elbowAngle += Math.exp(-((jt - 0.6) ** 2) / 0.01) * 2
    }

    const angles: JointAngles = {
      hip_rotation: hipAngle,
      trunk_rotation: trunkAngle,
      right_shoulder: shoulderAngle,
      right_elbow: elbowAngle,
      right_knee: 170,
      left_knee: 170,
    }

    frames.push(makeFrame(i, ts, makeStandingPose(), angles))
  }
  return frames
}

/**
 * Build frames with reversed kinetic chain (elbow fires before hips).
 */
function makeReversedSequenceFrames(totalFrames = 40): PoseFrame[] {
  const frames: PoseFrame[] = []
  for (let i = 0; i < totalFrames; i++) {
    const t = i / (totalFrames - 1)
    const ts = i * 33

    // Reversed: elbow peaks first (t=0.3), then shoulder, trunk, hips
    let hipAngle = 10
    let trunkAngle = 10
    let shoulderAngle = 45
    let elbowAngle = 120

    for (let j = 0; j <= i; j++) {
      const jt = j / (totalFrames - 1)
      hipAngle += Math.exp(-((jt - 0.6) ** 2) / 0.01) * 2
      trunkAngle += Math.exp(-((jt - 0.5) ** 2) / 0.01) * 2
      shoulderAngle += Math.exp(-((jt - 0.4) ** 2) / 0.01) * 2
      elbowAngle += Math.exp(-((jt - 0.3) ** 2) / 0.01) * 2
    }

    const angles: JointAngles = {
      hip_rotation: hipAngle,
      trunk_rotation: trunkAngle,
      right_shoulder: shoulderAngle,
      right_elbow: elbowAngle,
      right_knee: 170,
      left_knee: 170,
    }

    frames.push(makeFrame(i, ts, makeStandingPose(), angles))
  }
  return frames
}

// ---------------------------------------------------------------------------
// analyzeKineticChain
// ---------------------------------------------------------------------------

describe('analyzeKineticChain', () => {
  it('returns all segments with -1 peakFrame when given < 2 frames', () => {
    const result = analyzeKineticChain([])
    expect(result.segments).toHaveLength(FOREHAND_KINETIC_CHAIN.length)
    expect(result.isSequenceCorrect).toBe(false)
    expect(result.sequenceDescription).toContain('Insufficient frames')
    for (const seg of result.segments) {
      expect(seg.peakFrame).toBe(-1)
      expect(seg.peakTimestampMs).toBe(-1)
      expect(seg.peakVelocity).toBe(0)
    }
  })

  it('returns insufficient result for a single frame', () => {
    const frames = [makeFrame(0, 0, makeStandingPose(), { hip_rotation: 10 })]
    const result = analyzeKineticChain(frames)
    expect(result.isSequenceCorrect).toBe(false)
    expect(result.sequenceDescription).toContain('Insufficient frames')
  })

  it('detects correct proximal-to-distal sequence', () => {
    const frames = makeCorrectSequenceFrames(40)
    const result = analyzeKineticChain(frames)

    // Check that detected segments have valid peak frames
    const detected = result.segments.filter((s) => s.peakFrame >= 0)
    expect(detected.length).toBeGreaterThanOrEqual(2)

    // Verify the sequence is correct
    expect(result.isSequenceCorrect).toBe(true)
    expect(result.outOfOrderSegments).toHaveLength(0)
    expect(result.sequenceDescription).toContain('correct')
  })

  it('detects reversed (incorrect) kinetic chain', () => {
    const frames = makeReversedSequenceFrames(40)
    const result = analyzeKineticChain(frames)

    expect(result.isSequenceCorrect).toBe(false)
    expect(result.outOfOrderSegments.length).toBeGreaterThan(0)
  })

  it('handles frames with partial angle data', () => {
    // Only provide hip_rotation and trunk_rotation, no shoulder or elbow
    const frames: PoseFrame[] = []
    for (let i = 0; i < 20; i++) {
      const angles: JointAngles = {
        hip_rotation: 10 + (i < 10 ? i * 5 : 50),
        trunk_rotation: 10 + (i < 15 ? i * 3 : 45),
      }
      frames.push(makeFrame(i, i * 33, makeStandingPose(), angles))
    }

    const result = analyzeKineticChain(frames)

    // Should detect hips and trunk but not shoulder or elbow
    const hips = result.segments.find((s) => s.name === 'hips')
    const trunk = result.segments.find((s) => s.name === 'trunk')
    const shoulder = result.segments.find((s) => s.name === 'shoulder')
    const elbow = result.segments.find((s) => s.name === 'elbow')

    expect(hips?.peakFrame).toBeGreaterThanOrEqual(0)
    expect(trunk?.peakFrame).toBeGreaterThanOrEqual(0)
    expect(shoulder?.peakFrame).toBe(-1)
    expect(elbow?.peakFrame).toBe(-1)
  })

  it('returns meaningful description when only one segment detected', () => {
    // Only provide hip_rotation data
    const frames: PoseFrame[] = []
    for (let i = 0; i < 10; i++) {
      const angles: JointAngles = {
        hip_rotation: 10 + i * 5,
      }
      frames.push(makeFrame(i, i * 33, makeStandingPose(), angles))
    }

    const result = analyzeKineticChain(frames)
    expect(result.isSequenceCorrect).toBe(false)
    expect(result.sequenceDescription).toContain('hips')
    expect(result.sequenceDescription).toContain('at least 2 segments')
  })

  it('uses custom chain definition when provided', () => {
    const customChain = [
      { name: 'hips', angleKey: 'hip_rotation' as keyof JointAngles },
      { name: 'arm', angleKey: 'right_elbow' as keyof JointAngles },
    ]

    const frames: PoseFrame[] = []
    for (let i = 0; i < 20; i++) {
      const angles: JointAngles = {
        hip_rotation: 10 + (i < 8 ? i * 6 : 50),
        right_elbow: 120 + (i < 12 ? 0 : (i - 12) * 8),
      }
      frames.push(makeFrame(i, i * 33, makeStandingPose(), angles))
    }

    const result = analyzeKineticChain(frames, customChain)
    expect(result.segments).toHaveLength(2)
    expect(result.segments[0].name).toBe('hips')
    expect(result.segments[1].name).toBe('arm')
  })

  it('each detected segment has positive peak velocity', () => {
    const frames = makeCorrectSequenceFrames(40)
    const result = analyzeKineticChain(frames)

    for (const seg of result.segments) {
      if (seg.peakFrame >= 0) {
        expect(seg.peakVelocity).toBeGreaterThan(0)
      }
    }
  })

  it('peak timestamps are within the frame range', () => {
    const frames = makeCorrectSequenceFrames(40)
    const maxTs = frames[frames.length - 1].timestamp_ms
    const result = analyzeKineticChain(frames)

    for (const seg of result.segments) {
      if (seg.peakFrame >= 0) {
        expect(seg.peakTimestampMs).toBeGreaterThanOrEqual(0)
        expect(seg.peakTimestampMs).toBeLessThanOrEqual(maxTs)
      }
    }
  })

  it('handles uniform angle data (no velocity spikes)', () => {
    const frames: PoseFrame[] = []
    for (let i = 0; i < 20; i++) {
      const angles: JointAngles = {
        hip_rotation: 10,
        trunk_rotation: 10,
        right_shoulder: 45,
        right_elbow: 120,
      }
      frames.push(makeFrame(i, i * 33, makeStandingPose(), angles))
    }

    const result = analyzeKineticChain(frames)

    // No angular velocity changes => no peaks detected
    for (const seg of result.segments) {
      expect(seg.peakVelocity).toBe(0)
      expect(seg.peakFrame).toBe(-1)
    }
    expect(result.isSequenceCorrect).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// FOREHAND_KINETIC_CHAIN
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// getChainTimings
// ---------------------------------------------------------------------------

describe('getChainTimings', () => {
  it('returns peak timestamps in the order the shot config defines', () => {
    const frames = makeCorrectSequenceFrames(40)
    const timings = getChainTimings(frames, 'forehand')

    // At least hips/trunk/shoulder/elbow should resolve for this synthetic data.
    expect(timings.length).toBeGreaterThanOrEqual(2)

    // Every timing must be a valid ms within the clip's duration.
    const maxTs = frames[frames.length - 1].timestamp_ms
    for (const t of timings) {
      expect(t.peakMs).toBeGreaterThanOrEqual(0)
      expect(t.peakMs).toBeLessThanOrEqual(maxTs)
      expect(typeof t.joint).toBe('string')
    }

    // Hips peak should arrive before the shoulder peak (proximal-to-distal).
    const hips = timings.find((t) => t.joint === 'hip_rotation')
    const shoulder = timings.find((t) => t.joint === 'right_shoulder')
    if (hips && shoulder) {
      expect(hips.peakMs).toBeLessThan(shoulder.peakMs)
    }
  })

  it('omits links with no detectable peak', () => {
    // Only hip_rotation has any motion; the rest stay flat.
    const frames: PoseFrame[] = []
    for (let i = 0; i < 20; i++) {
      const angles: JointAngles = {
        hip_rotation: 10 + (i < 10 ? i * 5 : 50),
      }
      frames.push(makeFrame(i, i * 33, makeStandingPose(), angles))
    }

    const timings = getChainTimings(frames, 'forehand')
    expect(timings.map((t) => t.joint)).toEqual(['hip_rotation'])
  })
})

describe('FOREHAND_KINETIC_CHAIN', () => {
  it('defines the correct proximal-to-distal order', () => {
    const names = FOREHAND_KINETIC_CHAIN.map((c) => c.name)
    expect(names).toEqual(['hips', 'trunk', 'shoulder', 'elbow', 'wrist'])
  })

  it('maps each segment to a valid JointAngles key', () => {
    const validKeys: (keyof JointAngles)[] = [
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
    for (const seg of FOREHAND_KINETIC_CHAIN) {
      expect(validKeys).toContain(seg.angleKey)
    }
  })
})
