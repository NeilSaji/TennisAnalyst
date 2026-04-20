import { describe, it, expect } from 'vitest'
import {
  averageVisibility,
  landmarkBboxArea,
  isFrameConfident,
  smoothFrames,
} from '@/lib/poseSmoothing'
import type { PoseFrame, Landmark } from '@/lib/supabase'

// ---------------------------------------------------------------------------
// Helpers to build test data
// ---------------------------------------------------------------------------

function makeLandmark(id: number, overrides: Partial<Landmark> = {}): Landmark {
  return {
    id,
    name: `landmark_${id}`,
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 1.0,
    ...overrides,
  }
}

function makeFrame(
  frameIndex: number,
  landmarkOverrides?: Partial<Landmark>[],
  numLandmarks = 33
): PoseFrame {
  const landmarks = Array.from({ length: numLandmarks }, (_, i) =>
    makeLandmark(i, landmarkOverrides?.[i] ?? {})
  )
  return {
    frame_index: frameIndex,
    timestamp_ms: frameIndex * 33.33,
    landmarks,
    joint_angles: {},
  }
}

// ---------------------------------------------------------------------------
// averageVisibility
// ---------------------------------------------------------------------------

describe('averageVisibility', () => {
  it('returns 0 for empty array', () => {
    expect(averageVisibility([])).toBe(0)
  })

  it('returns the mean visibility', () => {
    const lms = [
      makeLandmark(0, { visibility: 0.8 }),
      makeLandmark(1, { visibility: 0.6 }),
    ]
    expect(averageVisibility(lms)).toBeCloseTo(0.7)
  })
})

// ---------------------------------------------------------------------------
// landmarkBboxArea
// ---------------------------------------------------------------------------

describe('landmarkBboxArea', () => {
  it('returns 0 for empty array', () => {
    expect(landmarkBboxArea([])).toBe(0)
  })

  it('computes correct bbox area', () => {
    const lms = [
      makeLandmark(0, { x: 0.1, y: 0.2 }),
      makeLandmark(1, { x: 0.5, y: 0.8 }),
    ]
    // (0.5 - 0.1) * (0.8 - 0.2) = 0.4 * 0.6 = 0.24
    expect(landmarkBboxArea(lms)).toBeCloseTo(0.24)
  })
})

// ---------------------------------------------------------------------------
// isFrameConfident
// ---------------------------------------------------------------------------

describe('isFrameConfident', () => {
  it('rejects frame with low average visibility', () => {
    const lms = [
      makeLandmark(0, { visibility: 0.1 }),
      makeLandmark(1, { visibility: 0.2 }),
    ]
    expect(isFrameConfident(lms)).toBe(false)
  })

  it('rejects frame with tiny bounding box', () => {
    const lms = [
      makeLandmark(0, { x: 0.5, y: 0.5, visibility: 1 }),
      makeLandmark(1, { x: 0.501, y: 0.501, visibility: 1 }),
    ]
    // bbox area = 0.001 * 0.001 = 0.000001 < 0.01
    expect(isFrameConfident(lms)).toBe(false)
  })

  it('accepts frame with good visibility and bbox', () => {
    const lms = [
      makeLandmark(0, { x: 0.2, y: 0.1, visibility: 0.9 }),
      makeLandmark(1, { x: 0.8, y: 0.9, visibility: 0.8 }),
    ]
    expect(isFrameConfident(lms)).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// smoothFrames
// ---------------------------------------------------------------------------

describe('smoothFrames', () => {
  it('returns empty array for empty input', () => {
    expect(smoothFrames([])).toEqual([])
  })

  it('discards first 5 warmup frames when total > 10', () => {
    // Build 15 frames with spread-out landmarks so they pass confidence
    const frames = Array.from({ length: 15 }, (_, i) =>
      makeFrame(i, [
        { x: 0.2, y: 0.1 },
        { x: 0.8, y: 0.9 },
      ])
    )
    const result = smoothFrames(frames)
    // First 5 frames should be removed
    expect(result[0].frame_index).toBe(5)
  })

  it('does NOT discard warmup frames when total <= 10', () => {
    const frames = Array.from({ length: 8 }, (_, i) =>
      makeFrame(i, [
        { x: 0.2, y: 0.1 },
        { x: 0.8, y: 0.9 },
      ])
    )
    const result = smoothFrames(frames)
    expect(result[0].frame_index).toBe(0)
  })

  it('filters out low-confidence frames', () => {
    const frames = [
      // low visibility frame
      makeFrame(
        0,
        Array.from({ length: 33 }, () => ({
          x: 0.3,
          y: 0.3,
          visibility: 0.1,
        }))
      ),
      // good frame
      makeFrame(1, [
        { x: 0.2, y: 0.1, visibility: 0.9 },
        { x: 0.8, y: 0.9, visibility: 0.9 },
      ]),
    ]
    // Total <= 10 so no warmup discard
    const result = smoothFrames(frames)
    expect(result.length).toBe(1)
    expect(result[0].frame_index).toBe(1)
  })

  it('smoothed output moves toward new data each step (one euro)', () => {
    // With a low cutoff, smoothed output should sit between the previous
    // state and the fresh measurement.
    const frame0 = makeFrame(0, [
      { x: 0.2, y: 0.2, visibility: 1 },
      { x: 0.8, y: 0.8, visibility: 1 },
    ])
    const frame1 = makeFrame(1, [
      { x: 0.8, y: 0.8, visibility: 1 },
      { x: 0.2, y: 0.2, visibility: 1 },
    ])
    const result = smoothFrames([frame0, frame1], {
      warmupDiscard: 0,
      lookaheadWindow: 1,
      minCutoff: 1.0,
      beta: 0.007,
      dcutoff: 1.0,
    })
    expect(result.length).toBe(2)

    // First frame stays close to its raw value.
    expect(result[0].landmarks[0].x).toBeCloseTo(0.2, 2)

    // Second frame output lies between previous (0.2) and new (0.8).
    const x1 = result[1].landmarks[0].x
    expect(x1).toBeGreaterThan(0.2)
    expect(x1).toBeLessThan(0.8)
  })

  it('look-ahead seeding produces a stable first frame', () => {
    // With lookaheadWindow=2, seed = average of frame0 and frame1.
    // Seed for landmark 0 x = (0.2+0.8)/2 = 0.5. First-frame output is pulled
    // from the seed toward the raw 0.2 measurement.
    const frame0 = makeFrame(0, [
      { x: 0.2, y: 0.2, visibility: 1 },
      { x: 0.8, y: 0.8, visibility: 1 },
    ])
    const frame1 = makeFrame(1, [
      { x: 0.8, y: 0.8, visibility: 1 },
      { x: 0.2, y: 0.2, visibility: 1 },
    ])
    const result = smoothFrames([frame0, frame1], {
      warmupDiscard: 0,
      lookaheadWindow: 2,
    })
    expect(result.length).toBe(2)

    // The smoothed first-frame x for landmark 0 should land between the seed
    // (0.5) and the measurement (0.2).
    const x0 = result[0].landmarks[0].x
    expect(x0).toBeGreaterThan(0.2)
    expect(x0).toBeLessThan(0.5)
  })

  it('recomputes joint_angles from smoothed landmarks', () => {
    // Give the frame realistic shoulder/elbow/wrist landmarks so angles
    // can be computed. The exact values don't matter - we just verify
    // the angles object is freshly computed (not the original empty one).
    const frame = makeFrame(0, [
      // landmarks 0–10 default
      ...Array.from({ length: 11 }, () => ({ x: 0.3, y: 0.3, visibility: 0.9 })),
      // 11 LEFT_SHOULDER
      { x: 0.4, y: 0.3, visibility: 0.9 },
      // 12 RIGHT_SHOULDER
      { x: 0.6, y: 0.3, visibility: 0.9 },
      // 13 LEFT_ELBOW
      { x: 0.35, y: 0.5, visibility: 0.9 },
      // 14 RIGHT_ELBOW
      { x: 0.65, y: 0.5, visibility: 0.9 },
      // 15 LEFT_WRIST
      { x: 0.3, y: 0.7, visibility: 0.9 },
      // 16 RIGHT_WRIST
      { x: 0.7, y: 0.7, visibility: 0.9 },
      // 17–22 default
      ...Array.from({ length: 6 }, () => ({ x: 0.5, y: 0.5, visibility: 0.9 })),
      // 23 LEFT_HIP
      { x: 0.45, y: 0.6, visibility: 0.9 },
      // 24 RIGHT_HIP
      { x: 0.55, y: 0.6, visibility: 0.9 },
      // 25 LEFT_KNEE
      { x: 0.43, y: 0.8, visibility: 0.9 },
      // 26 RIGHT_KNEE
      { x: 0.57, y: 0.8, visibility: 0.9 },
      // 27 LEFT_ANKLE
      { x: 0.42, y: 0.95, visibility: 0.9 },
      // 28 RIGHT_ANKLE
      { x: 0.58, y: 0.95, visibility: 0.9 },
    ])

    // Zero out the angles to verify they get recomputed
    frame.joint_angles = {}

    const result = smoothFrames([frame], { warmupDiscard: 0 })
    expect(result.length).toBe(1)
    // At least some angles should now be computed
    const angles = result[0].joint_angles
    expect(angles.right_elbow).toBeDefined()
    expect(typeof angles.right_elbow).toBe('number')
  })

  it('higher minCutoff reduces lag (more responsive)', () => {
    const frame0 = makeFrame(0, [
      { x: 0.0, y: 0.0, visibility: 1 },
      { x: 1.0, y: 1.0, visibility: 1 },
    ])
    const frame1 = makeFrame(1, [
      { x: 1.0, y: 1.0, visibility: 1 },
      { x: 0.0, y: 0.0, visibility: 1 },
    ])

    const responsive = smoothFrames([frame0, frame1], {
      warmupDiscard: 0,
      lookaheadWindow: 1,
      minCutoff: 100,
      dcutoff: 100,
    })
    const sluggish = smoothFrames([frame0, frame1], {
      warmupDiscard: 0,
      lookaheadWindow: 1,
      minCutoff: 0.1,
      dcutoff: 0.1,
    })

    // The responsive filter should track the new measurement (1.0) closer
    // than the sluggish one.
    const responsiveDelta = Math.abs(responsive[1].landmarks[0].x - 1.0)
    const sluggishDelta = Math.abs(sluggish[1].landmarks[0].x - 1.0)
    expect(responsiveDelta).toBeLessThan(sluggishDelta)
  })
})
