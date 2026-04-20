import { describe, it, expect } from 'vitest'
import {
  averageVisibility,
  landmarkBboxArea,
  isFrameConfident,
  smoothFrames,
} from '@/lib/poseSmoothing'
import { LANDMARK_INDICES } from '@/lib/jointAngles'
import { makeLandmark, makeFrame, makeStandingPose } from '../helpers'
import type { PoseFrame, Landmark } from '@/lib/supabase'

// ---------------------------------------------------------------------------
// averageVisibility
// ---------------------------------------------------------------------------

describe('averageVisibility', () => {
  it('returns 0 for empty landmarks', () => {
    expect(averageVisibility([])).toBe(0)
  })

  it('returns the average visibility', () => {
    const landmarks = [
      makeLandmark(0, 0, 0, 0.8),
      makeLandmark(1, 0, 0, 0.6),
      makeLandmark(2, 0, 0, 1.0),
    ]
    expect(averageVisibility(landmarks)).toBeCloseTo(0.8, 5)
  })

  it('returns 1.0 when all landmarks have full visibility', () => {
    const landmarks = [
      makeLandmark(0, 0, 0, 1.0),
      makeLandmark(1, 0, 0, 1.0),
    ]
    expect(averageVisibility(landmarks)).toBe(1.0)
  })

  it('returns 0 when all landmarks have zero visibility', () => {
    const landmarks = [
      makeLandmark(0, 0, 0, 0),
      makeLandmark(1, 0, 0, 0),
    ]
    expect(averageVisibility(landmarks)).toBe(0)
  })
})

// ---------------------------------------------------------------------------
// landmarkBboxArea
// ---------------------------------------------------------------------------

describe('landmarkBboxArea', () => {
  it('returns 0 for empty landmarks', () => {
    expect(landmarkBboxArea([])).toBe(0)
  })

  it('returns correct area for known positions', () => {
    const landmarks = [
      makeLandmark(0, 0.2, 0.3),
      makeLandmark(1, 0.8, 0.9),
    ]
    // (0.8-0.2) * (0.9-0.3) = 0.6 * 0.6 = 0.36
    expect(landmarkBboxArea(landmarks)).toBeCloseTo(0.36, 5)
  })

  it('returns 0 for a single landmark (zero area)', () => {
    const landmarks = [makeLandmark(0, 0.5, 0.5)]
    expect(landmarkBboxArea(landmarks)).toBe(0)
  })

  it('returns 0 for collinear landmarks on a horizontal line', () => {
    const landmarks = [
      makeLandmark(0, 0.2, 0.5),
      makeLandmark(1, 0.8, 0.5),
    ]
    // height = 0
    expect(landmarkBboxArea(landmarks)).toBe(0)
  })
})

// ---------------------------------------------------------------------------
// isFrameConfident
// ---------------------------------------------------------------------------

describe('isFrameConfident', () => {
  it('returns true for well-detected landmarks', () => {
    const landmarks = makeStandingPose() // high visibility, spread out
    expect(isFrameConfident(landmarks)).toBe(true)
  })

  it('returns false when average visibility is below threshold', () => {
    const landmarks = [
      makeLandmark(0, 0.2, 0.3, 0.1),
      makeLandmark(1, 0.8, 0.9, 0.2),
    ]
    // avg visibility = 0.15, default threshold = 0.4
    expect(isFrameConfident(landmarks)).toBe(false)
  })

  it('returns false when bbox area is below threshold', () => {
    // Landmarks very close together
    const landmarks = [
      makeLandmark(0, 0.5, 0.5, 1.0),
      makeLandmark(1, 0.501, 0.501, 1.0),
    ]
    // area = 0.001 * 0.001 = 0.000001, default min = 0.01
    expect(isFrameConfident(landmarks)).toBe(false)
  })

  it('respects custom visibilityThreshold', () => {
    const landmarks = [
      makeLandmark(0, 0.2, 0.3, 0.3),
      makeLandmark(1, 0.8, 0.9, 0.3),
    ]
    // avg visibility = 0.3
    expect(isFrameConfident(landmarks, { visibilityThreshold: 0.2 })).toBe(true)
    expect(isFrameConfident(landmarks, { visibilityThreshold: 0.5 })).toBe(false)
  })

  it('respects custom minBboxArea', () => {
    const landmarks = [
      makeLandmark(0, 0.4, 0.4, 1.0),
      makeLandmark(1, 0.5, 0.5, 1.0),
    ]
    // area = 0.1 * 0.1 = 0.01
    expect(isFrameConfident(landmarks, { minBboxArea: 0.005 })).toBe(true)
    expect(isFrameConfident(landmarks, { minBboxArea: 0.02 })).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// smoothFrames
// ---------------------------------------------------------------------------

describe('smoothFrames', () => {
  it('returns empty array for empty input', () => {
    expect(smoothFrames([])).toEqual([])
  })

  it('discards warm-up frames when total > 10', () => {
    const frames = Array.from({ length: 20 }, (_, i) =>
      makeFrame(i, i * 33, makeStandingPose())
    )
    const result = smoothFrames(frames, { warmupDiscard: 5 })

    // 20 frames, discard first 5 = 15 frames (assuming all pass confidence)
    expect(result.length).toBeLessThanOrEqual(15)
    expect(result.length).toBeGreaterThan(0)
  })

  it('does not discard warm-up frames when total <= 10', () => {
    const frames = Array.from({ length: 8 }, (_, i) =>
      makeFrame(i, i * 33, makeStandingPose())
    )
    const result = smoothFrames(frames, { warmupDiscard: 3 })

    // 8 frames, below WARMUP_MIN_TOTAL_FRAMES (10), no discard
    expect(result.length).toBe(8)
  })

  it('filters out low-confidence frames', () => {
    const frames = [
      // Good frame
      makeFrame(0, 0, makeStandingPose()),
      // Bad frame (low visibility)
      makeFrame(
        1,
        33,
        makeStandingPose().map((l) => ({ ...l, visibility: 0.1 }))
      ),
      // Good frame
      makeFrame(2, 66, makeStandingPose()),
    ]
    const result = smoothFrames(frames)

    // Frame 1 should be filtered out
    expect(result.length).toBe(2)
  })

  it('attenuates high-frequency jitter (variance reduction invariant)', () => {
    // 30 frames at 30fps. Base sinusoid (slow, preserved) + per-frame jitter.
    const frames: PoseFrame[] = []
    for (let i = 0; i < 30; i++) {
      const base = 0.5 + 0.05 * Math.sin((i / 30) * Math.PI * 2)
      const jitter = (i % 2 === 0 ? 1 : -1) * 0.03 // sharp tick-tock
      frames.push(
        makeFrame(i, i * 33, [
          makeLandmark(LANDMARK_INDICES.NOSE, base + jitter, 0.5, 1.0),
          makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.4, 0.6, 1.0),
          makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.6, 0.6, 1.0),
        ])
      )
    }

    const smoothed = smoothFrames(frames, { warmupDiscard: 0 })

    const rawXs = frames
      .map((f) => f.landmarks.find((l) => l.id === LANDMARK_INDICES.NOSE)!.x)
    const smoothedXs = smoothed
      .map((f) => f.landmarks.find((l) => l.id === LANDMARK_INDICES.NOSE)!.x)

    // Frame-to-frame delta variance is the relevant proxy for jitter here.
    const rawDelta = rawXs.slice(1).map((v, i) => Math.abs(v - rawXs[i]))
    const smDelta = smoothedXs.slice(1).map((v, i) => Math.abs(v - smoothedXs[i]))
    const meanRaw = rawDelta.reduce((a, b) => a + b, 0) / rawDelta.length
    const meanSm = smDelta.reduce((a, b) => a + b, 0) / smDelta.length

    expect(meanSm).toBeLessThan(meanRaw)
  })

  it('preserves the contact-frame peak (high-speed events are not flattened)', () => {
    // Build a trajectory with a sharp peak at frame 15 (the "contact" frame).
    const frames: PoseFrame[] = []
    for (let i = 0; i < 30; i++) {
      // Triangle wave peaking at frame 15.
      const x = 0.3 + 0.4 * (1 - Math.abs(i - 15) / 15)
      frames.push(
        makeFrame(i, i * 33, [
          makeLandmark(LANDMARK_INDICES.NOSE, x, 0.5, 1.0),
          makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.4, 0.6, 1.0),
          makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.6, 0.6, 1.0),
        ])
      )
    }

    const smoothed = smoothFrames(frames, { warmupDiscard: 0 })
    const smoothedXs = smoothed.map(
      (f) => f.landmarks.find((l) => l.id === LANDMARK_INDICES.NOSE)!.x
    )

    // The peak should still land at or next to frame 15, and its amplitude
    // should stay within 30% of the raw peak (0.7).
    let peakIdx = 0
    let peakVal = -Infinity
    for (let i = 0; i < smoothedXs.length; i++) {
      if (smoothedXs[i] > peakVal) {
        peakVal = smoothedXs[i]
        peakIdx = i
      }
    }
    // One Euro has small phase lag at low cutoffs; a 3-frame shift is fine.
    expect(Math.abs(peakIdx - 15)).toBeLessThanOrEqual(3)
    expect(peakVal).toBeGreaterThan(0.7 * 0.7)
  })

  it('recomputes joint angles from smoothed landmarks', () => {
    const frames = Array.from({ length: 5 }, (_, i) =>
      makeFrame(i, i * 33, makeStandingPose(), {
        right_elbow: 999, // original value that should be replaced
      })
    )

    const result = smoothFrames(frames, { warmupDiscard: 0 })

    // Joint angles should be recomputed from the smoothed landmarks,
    // not carried over from the original
    for (const frame of result) {
      if (frame.joint_angles.right_elbow !== undefined) {
        expect(frame.joint_angles.right_elbow).not.toBe(999)
      }
    }
  })

  it('returns empty array when all frames are low confidence', () => {
    const frames = Array.from({ length: 5 }, (_, i) =>
      makeFrame(
        i,
        i * 33,
        makeStandingPose().map((l) => ({ ...l, visibility: 0.1 }))
      )
    )

    const result = smoothFrames(frames)
    expect(result).toEqual([])
  })

  it('does not mutate the input frames', () => {
    const origLandmarks = makeStandingPose()
    const origX = origLandmarks[0].x
    const frames = [makeFrame(0, 0, origLandmarks)]

    smoothFrames(frames, { warmupDiscard: 0 })

    // Original landmark should be unchanged
    expect(frames[0].landmarks[0].x).toBe(origX)
  })

  it('very high cutoffs approach passthrough behaviour', () => {
    const frames = Array.from({ length: 5 }, (_, i) => {
      const landmarks = makeStandingPose().map((l) => ({
        ...l,
        x: l.x + i * 0.05,
      }))
      return makeFrame(i, i * 33, landmarks)
    })

    // A very large minCutoff pushes α → 1 (minimal lag on linear trajectories).
    const result = smoothFrames(frames, {
      minCutoff: 10000,
      dcutoff: 10000,
      warmupDiscard: 0,
      lookaheadWindow: 1,
    })

    // With effectively no smoothing, output should track input to within
    // a small tolerance (not exact because the 1€ alpha can never hit 1.0).
    for (let i = 1; i < result.length; i++) {
      for (const lm of result[i].landmarks) {
        const orig = frames[i].landmarks.find((l) => l.id === lm.id)!
        expect(lm.x).toBeCloseTo(orig.x, 3)
      }
    }
  })
})
