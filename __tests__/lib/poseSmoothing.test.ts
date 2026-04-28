import { describe, it, expect } from 'vitest'
import {
  averageVisibility,
  landmarkBboxArea,
  isFrameConfident,
  isBodyVisible,
  smoothFrames,
  filterImplausibleArmJoints,
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
// isBodyVisible
// ---------------------------------------------------------------------------

describe('isBodyVisible', () => {
  // Full-body pose with realistic vertical spread for the body-visible
  // gate's 0.35 default. makeStandingPose() in helpers.ts is laid out
  // tighter (0.25 -> 0.55) for joint-angle tests and is intentionally
  // not used here.
  const makeFullBodyLandmarks = () => [
    makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.55, 0.25, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.45, 0.25, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_ELBOW, 0.6, 0.4, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.4, 0.4, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_WRIST, 0.62, 0.55, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.38, 0.55, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.53, 0.65, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.47, 0.65, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.54, 0.8, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.46, 0.8, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_ANKLE, 0.54, 0.95, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_ANKLE, 0.46, 0.95, 0.95),
  ]

  it('returns false for face-only landmarks (no shoulders/hips/wrists)', () => {
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1, 0.95),
    ]
    expect(isBodyVisible(landmarks)).toBe(false)
  })

  it('returns false for head + shoulders only (no hips)', () => {
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1, 0.95),
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.55, 0.25, 0.95),
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.45, 0.25, 0.95),
      // Wrist visible but no hips at all -- common "head-and-shoulders"
      // crop where the camera only sees the upper body.
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.4, 0.4, 0.95),
    ]
    expect(isBodyVisible(landmarks)).toBe(false)
  })

  it('returns true for full body with all required landmarks high-visibility', () => {
    // Vertical extent of the required landmarks (shoulders 0.25, wrists
    // 0.55, hips 0.65) is 0.40, comfortably above the 0.35 default.
    expect(isBodyVisible(makeFullBodyLandmarks())).toBe(true)
  })

  it('returns true when only one wrist is visible (single wrist required)', () => {
    // Take a full pose, then occlude the left wrist -- the right wrist is
    // still visible so the gate should pass.
    const landmarks = makeFullBodyLandmarks().map((lm) =>
      lm.id === LANDMARK_INDICES.LEFT_WRIST
        ? { ...lm, visibility: 0.1 }
        : lm,
    )
    expect(isBodyVisible(landmarks)).toBe(true)
  })

  it('returns false when vertical extent of required landmarks < 0.35', () => {
    // Subject is too small / too cropped: shoulders, hips, and wrist all
    // crammed into a ~0.2-tall band near the top of the frame. Visibility
    // is fine on every landmark, but the player isn't really in the frame.
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.55, 0.25, 0.95),
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.45, 0.25, 0.95),
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.4, 0.30, 0.95),
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.53, 0.45, 0.95),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.47, 0.45, 0.95),
    ]
    // vertical extent = 0.45 - 0.25 = 0.20, below the 0.35 default
    expect(isBodyVisible(landmarks)).toBe(false)
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

  it('filterImplausibleArmJoints: zeros elbow visibility when upper-arm length is wildly off', () => {
    // Build 10 frames where the right upper arm sits at a consistent
    // normalized length of 0.1, then inject one frame where the right
    // elbow is placed 0.5 units away from the shoulder -- clearly a
    // blurred/misplaced detection.
    const makeFrame = (
      i: number,
      rightElbow: { x: number; y: number },
    ) => ({
      frame_index: i,
      timestamp_ms: i * 33,
      joint_angles: {},
      landmarks: [
        // shoulders
        { id: 11, name: 'lshoulder', x: 0.5, y: 0.5, z: 0, visibility: 0.95 },
        { id: 12, name: 'rshoulder', x: 0.6, y: 0.5, z: 0, visibility: 0.95 },
        // elbows
        { id: 13, name: 'lelbow', x: 0.4, y: 0.5, z: 0, visibility: 0.95 },
        {
          id: 14,
          name: 'relbow',
          x: rightElbow.x,
          y: rightElbow.y,
          z: 0,
          visibility: 0.95,
        },
        // wrists
        { id: 15, name: 'lwrist', x: 0.3, y: 0.5, z: 0, visibility: 0.95 },
        { id: 16, name: 'rwrist', x: 0.8, y: 0.5, z: 0, visibility: 0.95 },
      ],
    })

    const frames = [
      ...Array.from({ length: 10 }, (_, i) =>
        makeFrame(i, { x: 0.7, y: 0.5 }), // upper arm length = 0.1
      ),
      makeFrame(10, { x: 1.1, y: 0.5 }), // upper arm length = 0.5 -> 5x median
    ]

    const result = filterImplausibleArmJoints(frames)

    // Frames 0-9 should be untouched (length == median).
    for (let i = 0; i < 10; i++) {
      const relbow = result[i].landmarks.find((l) => l.id === 14)!
      expect(relbow.visibility).toBe(0.95)
    }
    // Frame 10's right elbow is implausible -> visibility zeroed.
    const badElbow = result[10].landmarks.find((l) => l.id === 14)!
    expect(badElbow.visibility).toBe(0)
    // Shoulders untouched -- only the child end of a bad bone is penalized.
    const badShoulder = result[10].landmarks.find((l) => l.id === 12)!
    expect(badShoulder.visibility).toBe(0.95)
  })

  it('filterImplausibleArmJoints: no-ops when there are not enough confident samples', () => {
    // Only 3 confident frames -- below MIN_SAMPLES_FOR_MEDIAN=5.
    const frames = Array.from({ length: 3 }, (_, i) => ({
      frame_index: i,
      timestamp_ms: i * 33,
      joint_angles: {},
      landmarks: [
        { id: 12, name: 'rshoulder', x: 0.6, y: 0.5, z: 0, visibility: 0.95 },
        { id: 14, name: 'relbow', x: 0.9, y: 0.5, z: 0, visibility: 0.95 },
      ],
    }))

    const result = filterImplausibleArmJoints(frames)
    for (const f of result) {
      for (const lm of f.landmarks) {
        expect(lm.visibility).toBe(0.95)
      }
    }
  })

  it('filterImplausibleArmJoints: ignores low-visibility samples when computing median', () => {
    // Build 10 frames with good (0.95 vis) shoulder-elbow pairs at length
    // 0.1, plus one frame with an implausible length but below the 0.7
    // vis gate -- that bad sample must not pollute the median.
    const ok = (i: number) => ({
      frame_index: i,
      timestamp_ms: i * 33,
      joint_angles: {},
      landmarks: [
        { id: 12, name: 'rshoulder', x: 0.6, y: 0.5, z: 0, visibility: 0.95 },
        { id: 14, name: 'relbow', x: 0.7, y: 0.5, z: 0, visibility: 0.95 },
      ],
    })
    const hiddenBad = {
      frame_index: 99,
      timestamp_ms: 99 * 33,
      joint_angles: {},
      landmarks: [
        { id: 12, name: 'rshoulder', x: 0.6, y: 0.5, z: 0, visibility: 0.5 },
        { id: 14, name: 'relbow', x: 1.5, y: 0.5, z: 0, visibility: 0.5 },
      ],
    }
    const frames = [...Array.from({ length: 10 }, (_, i) => ok(i)), hiddenBad]
    const result = filterImplausibleArmJoints(frames)
    // The good frames remain untouched (median stayed at 0.1 because the
    // low-vis bad frame was excluded from the sample).
    for (let i = 0; i < 10; i++) {
      expect(result[i].landmarks.find((l) => l.id === 14)!.visibility).toBe(0.95)
    }
  })

  it('zero-phase mode shifts the peak less than causal mode', () => {
    // A causal filter phase-shifts peaks forward in time (baked-in lag).
    // filtfilt should cancel that, so the peak of the same triangle wave
    // lands closer to the true index.
    const frames: PoseFrame[] = []
    for (let i = 0; i < 30; i++) {
      const x = 0.3 + 0.4 * (1 - Math.abs(i - 15) / 15)
      frames.push(
        makeFrame(i, i * 33, [
          makeLandmark(LANDMARK_INDICES.NOSE, x, 0.5, 1.0),
          makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.4, 0.6, 1.0),
          makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.6, 0.6, 1.0),
        ]),
      )
    }

    const argmaxNose = (out: PoseFrame[]) => {
      let peakIdx = 0
      let peakVal = -Infinity
      for (let i = 0; i < out.length; i++) {
        const x = out[i].landmarks.find(
          (l) => l.id === LANDMARK_INDICES.NOSE,
        )!.x
        if (x > peakVal) {
          peakVal = x
          peakIdx = i
        }
      }
      return peakIdx
    }

    const causal = smoothFrames(frames, { warmupDiscard: 0, zeroPhase: false })
    const zeroPhase = smoothFrames(frames, { warmupDiscard: 0 })

    const causalShift = Math.abs(argmaxNose(causal) - 15)
    const zeroPhaseShift = Math.abs(argmaxNose(zeroPhase) - 15)

    expect(zeroPhaseShift).toBeLessThanOrEqual(causalShift)
    expect(zeroPhaseShift).toBeLessThanOrEqual(1)
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
