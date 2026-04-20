import type { PoseFrame, JointAngles } from './supabase'
import { getShotTypeConfig, type ShotType } from './shotTypeConfig'

/**
 * Kinetic chain analysis for tennis swings.
 *
 * The kinetic chain is the sequence of body segment rotations that transfer
 * energy from the ground up through the body to the racket. Proper sequence
 * is: hips -> trunk -> shoulder -> elbow -> wrist.
 *
 * This module detects the timing of peak angular velocity for each segment
 * and evaluates whether the sequence is correct (proximal-to-distal).
 */

/** A segment in the kinetic chain with its peak timing. */
export interface ChainSegment {
  name: string
  /** The JointAngles key used to track this segment */
  angleKey: keyof JointAngles
  /** Frame index where peak angular velocity occurred (-1 if not detected) */
  peakFrame: number
  /** Timestamp (ms) of peak angular velocity (-1 if not detected) */
  peakTimestampMs: number
  /** Peak angular velocity in degrees/second (0 if not detected) */
  peakVelocity: number
}

/** Result of kinetic chain analysis. */
export interface KineticChainResult {
  /** Ordered list of chain segments with their peak timings */
  segments: ChainSegment[]
  /** Whether the sequence follows proper proximal-to-distal ordering */
  isSequenceCorrect: boolean
  /** Human-readable description of the sequence quality */
  sequenceDescription: string
  /**
   * Segments that are out of order (fired too early or too late).
   * Empty if sequence is correct.
   */
  outOfOrderSegments: string[]
}

/** The ideal kinetic chain sequence for a forehand. */
const FOREHAND_CHAIN: { name: string; angleKey: keyof JointAngles }[] = [
  { name: 'hips', angleKey: 'hip_rotation' },
  { name: 'trunk', angleKey: 'trunk_rotation' },
  { name: 'shoulder', angleKey: 'right_shoulder' },
  { name: 'elbow', angleKey: 'right_elbow' },
  { name: 'wrist', angleKey: 'right_wrist' },
]

/**
 * Compute the frame index where the given angle key has its peak
 * angular velocity (largest frame-to-frame change per unit time).
 */
function findPeakAngularVelocity(
  frames: PoseFrame[],
  angleKey: keyof JointAngles
): { peakFrame: number; peakTimestampMs: number; peakVelocity: number } {
  let peakFrame = -1
  let peakTimestampMs = -1
  let peakVelocity = 0

  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1].joint_angles[angleKey]
    const curr = frames[i].joint_angles[angleKey]
    if (prev == null || curr == null) continue

    const dt = (frames[i].timestamp_ms - frames[i - 1].timestamp_ms) / 1000
    if (dt <= 0) continue

    const velocity = Math.abs((curr as number) - (prev as number)) / dt
    if (velocity > peakVelocity) {
      peakVelocity = velocity
      peakFrame = i
      peakTimestampMs = frames[i].timestamp_ms
    }
  }

  return { peakFrame, peakTimestampMs, peakVelocity }
}

/**
 * Analyze the kinetic chain sequencing for a forehand swing.
 *
 * Checks whether body segments fire in the correct proximal-to-distal
 * order: hips -> trunk -> shoulder -> elbow.
 *
 * Returns meaningful results even with partial data (some segments may
 * not be detected if angle data is missing).
 */
export function analyzeKineticChain(
  frames: PoseFrame[],
  chain: { name: string; angleKey: keyof JointAngles }[] = FOREHAND_CHAIN
): KineticChainResult {
  if (frames.length < 2) {
    return {
      segments: chain.map((c) => ({
        name: c.name,
        angleKey: c.angleKey,
        peakFrame: -1,
        peakTimestampMs: -1,
        peakVelocity: 0,
      })),
      isSequenceCorrect: false,
      sequenceDescription:
        'Insufficient frames to analyze kinetic chain sequence.',
      outOfOrderSegments: [],
    }
  }

  const segments: ChainSegment[] = chain.map((c) => {
    const peak = findPeakAngularVelocity(frames, c.angleKey)
    return {
      name: c.name,
      angleKey: c.angleKey,
      ...peak,
    }
  })

  // Filter to only segments that were detected
  const detected = segments.filter((s) => s.peakFrame >= 0)

  if (detected.length < 2) {
    return {
      segments,
      isSequenceCorrect: false,
      sequenceDescription:
        detected.length === 0
          ? 'Could not detect peak angular velocity for any segment.'
          : `Only detected peak timing for ${detected[0].name}. Need at least 2 segments to evaluate sequencing.`,
      outOfOrderSegments: [],
    }
  }

  // Check if the detected segments are in the correct order
  // (peak timestamps should be increasing following the chain order)
  const outOfOrder: string[] = []
  let isCorrect = true

  for (let i = 1; i < detected.length; i++) {
    // Find the expected order indices
    const prevIdx = chain.findIndex((c) => c.name === detected[i - 1].name)
    const currIdx = chain.findIndex((c) => c.name === detected[i].name)

    if (prevIdx < currIdx) {
      // Correct chain order -now check timing
      if (detected[i].peakTimestampMs < detected[i - 1].peakTimestampMs) {
        isCorrect = false
        outOfOrder.push(detected[i].name)
      }
    } else if (prevIdx > currIdx) {
      // Segments arrived in wrong chain order
      isCorrect = false
      outOfOrder.push(detected[i - 1].name)
    }
  }

  // Also check that within the detected segments, timestamps follow chain order
  // Build a map of chain position -> detected segment timestamp
  const chainOrder = new Map<number, ChainSegment>()
  for (const seg of detected) {
    const idx = chain.findIndex((c) => c.name === seg.name)
    if (idx >= 0) chainOrder.set(idx, seg)
  }

  const orderedEntries = [...chainOrder.entries()].sort((a, b) => a[0] - b[0])
  for (let i = 1; i < orderedEntries.length; i++) {
    const prev = orderedEntries[i - 1][1]
    const curr = orderedEntries[i][1]
    if (curr.peakTimestampMs < prev.peakTimestampMs) {
      if (!outOfOrder.includes(curr.name)) {
        outOfOrder.push(curr.name)
      }
      isCorrect = false
    }
  }

  let description: string
  if (isCorrect) {
    description =
      'Kinetic chain sequence is correct: ' +
      detected.map((s) => s.name).join(' -> ') +
      '.'
  } else {
    description =
      'Kinetic chain sequence is out of order. ' +
      (outOfOrder.length > 0
        ? `${outOfOrder.join(', ')} fired out of sequence.`
        : 'Timing does not follow the expected proximal-to-distal pattern.')
  }

  return {
    segments,
    isSequenceCorrect: isCorrect,
    sequenceDescription: description,
    outOfOrderSegments: outOfOrder,
  }
}

/** The default forehand chain definition, exported for testing and reuse. */
export const FOREHAND_KINETIC_CHAIN = FOREHAND_CHAIN

/**
 * Per-chain-link peak timestamps for the timing-bar UI. The returned array
 * mirrors the shot's `kineticChainOrder` so the UI can walk through it
 * without re-deriving the chain. Links whose peak could not be detected are
 * omitted — the UI renders them as absent ticks.
 */
export function getChainTimings(
  frames: PoseFrame[],
  shotType: ShotType
): { joint: string; peakMs: number }[] {
  const order = getShotTypeConfig(shotType).kineticChainOrder
  const out: { joint: string; peakMs: number }[] = []
  for (const angleKey of order) {
    const peak = findPeakAngularVelocity(frames, angleKey)
    if (peak.peakFrame >= 0) {
      out.push({ joint: String(angleKey), peakMs: peak.peakTimestampMs })
    }
  }
  return out
}
