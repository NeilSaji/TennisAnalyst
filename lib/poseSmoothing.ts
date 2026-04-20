import { computeJointAngles } from './jointAngles'
import type { PoseFrame, Landmark } from './supabase'

/**
 * Minimum average landmark visibility for a frame to be considered valid.
 * Frames below this threshold are likely bad detections from MediaPipe's
 * warm-up period and are discarded.
 */
const DEFAULT_VISIBILITY_THRESHOLD = 0.4

/**
 * Minimum bounding-box area (as fraction of frame area [0,1]x[0,1]) for a
 * detection to be considered valid. A tiny bbox means MediaPipe only found
 * a sliver of the person — usually an artifact.
 */
const DEFAULT_MIN_BBOX_AREA = 0.01

/**
 * Number of warm-up frames to discard from the start of a detection run.
 * Only applied when total frame count exceeds WARMUP_MIN_TOTAL_FRAMES.
 */
const DEFAULT_WARMUP_DISCARD = 5
const WARMUP_MIN_TOTAL_FRAMES = 10

/**
 * Number of initial frames averaged to seed the filter state. Avoids
 * letting a single noisy first frame bias the first several outputs.
 */
const DEFAULT_LOOKAHEAD_WINDOW = 3

// One Euro Filter defaults (Casiez 2012). `mincutoff` sets the floor cutoff
// when the signal is still; `beta` controls how aggressively cutoff rises
// with speed; `dcutoff` is the fixed cutoff for the derivative smoother.
const DEFAULT_MIN_CUTOFF = 1.0
const DEFAULT_BETA = 0.007
const DEFAULT_DCUTOFF = 1.0

// Fallback dt (seconds) when timestamps are identical or go backwards.
const FALLBACK_DT = 1 / 30

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Average visibility across all landmarks in a frame. */
export function averageVisibility(landmarks: Landmark[]): number {
  if (landmarks.length === 0) return 0
  let sum = 0
  for (const lm of landmarks) {
    sum += lm.visibility
  }
  return sum / landmarks.length
}

/** Bounding-box area of landmarks in normalized [0,1] coordinate space. */
export function landmarkBboxArea(landmarks: Landmark[]): number {
  if (landmarks.length === 0) return 0
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x
    if (lm.y < minY) minY = lm.y
    if (lm.x > maxX) maxX = lm.x
    if (lm.y > maxY) maxY = lm.y
  }
  return (maxX - minX) * (maxY - minY)
}

// ---------------------------------------------------------------------------
// Confidence gating
// ---------------------------------------------------------------------------

export interface GateOptions {
  visibilityThreshold?: number
  minBboxArea?: number
}

export function isFrameConfident(
  landmarks: Landmark[],
  opts: GateOptions = {}
): boolean {
  const { visibilityThreshold = DEFAULT_VISIBILITY_THRESHOLD, minBboxArea = DEFAULT_MIN_BBOX_AREA } = opts
  if (averageVisibility(landmarks) < visibilityThreshold) return false
  if (landmarkBboxArea(landmarks) < minBboxArea) return false
  return true
}

// ---------------------------------------------------------------------------
// One Euro Filter
// ---------------------------------------------------------------------------

/**
 * Per-axis state for the One Euro Filter. Tracks the previous smoothed
 * value and the previous smoothed derivative so the next step can compute
 * an adaptive cutoff.
 */
interface OneEuroAxisState {
  initialized: boolean
  prevValue: number
  prevDeriv: number
  prevTsSec: number
}

interface OneEuroLandmarkState {
  x: OneEuroAxisState
  y: OneEuroAxisState
  z: OneEuroAxisState
}

function makeAxisState(seed?: number): OneEuroAxisState {
  return {
    initialized: seed !== undefined,
    prevValue: seed ?? 0,
    prevDeriv: 0,
    prevTsSec: 0,
  }
}

// Cutoff-frequency → exponential smoothing coefficient, per the 1€ paper.
function smoothingAlpha(cutoff: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cutoff)
  return 1 / (1 + tau / dt)
}

function exponentialSmooth(alpha: number, x: number, prev: number): number {
  return alpha * x + (1 - alpha) * prev
}

interface OneEuroParams {
  minCutoff: number
  beta: number
  dcutoff: number
}

function oneEuroStep(
  state: OneEuroAxisState,
  value: number,
  tsSec: number,
  params: OneEuroParams
): number {
  if (!state.initialized) {
    state.initialized = true
    state.prevValue = value
    state.prevDeriv = 0
    state.prevTsSec = tsSec
    return value
  }

  let dt = tsSec - state.prevTsSec
  if (!Number.isFinite(dt) || dt <= 0) dt = FALLBACK_DT

  const rawDeriv = (value - state.prevValue) / dt
  const dAlpha = smoothingAlpha(params.dcutoff, dt)
  const smoothedDeriv = exponentialSmooth(dAlpha, rawDeriv, state.prevDeriv)

  const cutoff = params.minCutoff + params.beta * Math.abs(smoothedDeriv)
  const alpha = smoothingAlpha(cutoff, dt)
  const smoothedValue = exponentialSmooth(alpha, value, state.prevValue)

  state.prevValue = smoothedValue
  state.prevDeriv = smoothedDeriv
  state.prevTsSec = tsSec
  return smoothedValue
}

/**
 * Average the first `windowSize` frames per landmark to produce a stable
 * seed for the filter. This prevents a single noisy first frame from
 * biasing several subsequent outputs.
 */
function computeLookaheadSeed(
  frames: PoseFrame[],
  windowSize: number
): Map<number, { x: number; y: number; z: number }> {
  const sums = new Map<number, { x: number; y: number; z: number; n: number }>()
  const n = Math.min(windowSize, frames.length)

  for (let i = 0; i < n; i++) {
    for (const lm of frames[i].landmarks) {
      const prev = sums.get(lm.id)
      if (!prev) {
        sums.set(lm.id, { x: lm.x, y: lm.y, z: lm.z, n: 1 })
      } else {
        prev.x += lm.x
        prev.y += lm.y
        prev.z += lm.z
        prev.n += 1
      }
    }
  }

  const seed = new Map<number, { x: number; y: number; z: number }>()
  for (const [id, v] of sums) {
    seed.set(id, { x: v.x / v.n, y: v.y / v.n, z: v.z / v.n })
  }
  return seed
}

function oneEuroSmooth(
  frames: PoseFrame[],
  params: OneEuroParams,
  lookaheadWindow: number
): PoseFrame[] {
  if (frames.length === 0) return []

  const seed = computeLookaheadSeed(frames, lookaheadWindow)
  const states = new Map<number, OneEuroLandmarkState>()

  // Seed timestamp ~= one frame step BEFORE the first frame so the first
  // output is the first measurement blended with the averaged seed (the
  // warm-up behavior the existing tests rely on).
  const tsSec0 = frames[0].timestamp_ms / 1000
  const tsSec1 = frames.length > 1 ? frames[1].timestamp_ms / 1000 : tsSec0 + FALLBACK_DT
  const seedDt = Math.max(FALLBACK_DT, tsSec1 - tsSec0)
  const seedTs = tsSec0 - seedDt

  for (const [id, pos] of seed) {
    states.set(id, {
      x: { initialized: true, prevValue: pos.x, prevDeriv: 0, prevTsSec: seedTs },
      y: { initialized: true, prevValue: pos.y, prevDeriv: 0, prevTsSec: seedTs },
      z: { initialized: true, prevValue: pos.z, prevDeriv: 0, prevTsSec: seedTs },
    })
  }

  const result: PoseFrame[] = []
  for (const frame of frames) {
    const tsSec = frame.timestamp_ms / 1000
    const smoothedLandmarks: Landmark[] = frame.landmarks.map((lm) => {
      let state = states.get(lm.id)
      if (!state) {
        state = {
          x: makeAxisState(lm.x),
          y: makeAxisState(lm.y),
          z: makeAxisState(lm.z),
        }
        // Initialize with the raw frame so prevTsSec is set for next step.
        state.x.prevTsSec = tsSec
        state.y.prevTsSec = tsSec
        state.z.prevTsSec = tsSec
        states.set(lm.id, state)
        return { ...lm }
      }
      const sx = oneEuroStep(state.x, lm.x, tsSec, params)
      const sy = oneEuroStep(state.y, lm.y, tsSec, params)
      const sz = oneEuroStep(state.z, lm.z, tsSec, params)
      return { ...lm, x: sx, y: sy, z: sz }
    })

    result.push({
      ...frame,
      landmarks: smoothedLandmarks,
      joint_angles: computeJointAngles(smoothedLandmarks),
    })
  }

  return result
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface SmoothOptions {
  /** One Euro Filter minimum cutoff frequency (Hz). Default 1.0. */
  minCutoff?: number
  /** One Euro Filter speed coefficient. Default 0.007. */
  beta?: number
  /** One Euro Filter derivative cutoff (Hz). Default 1.0. */
  dcutoff?: number
  /** Min average visibility to keep a frame. Default 0.4. */
  visibilityThreshold?: number
  /** Min bbox area fraction to keep a frame. Default 0.01. */
  minBboxArea?: number
  /** Number of initial warm-up frames to discard. Default 5. */
  warmupDiscard?: number
  /** Number of initial frames to average for filter seed. Default 3. */
  lookaheadWindow?: number
}

/**
 * Post-process raw PoseFrames extracted from MediaPipe:
 *
 * 1. Discard early warm-up frames (if total > 10).
 * 2. Filter out low-confidence frames.
 * 3. Apply a One Euro Filter per landmark axis.
 * 4. Recompute joint angles from smoothed landmarks.
 *
 * One Euro adapts its cutoff to motion speed: it smooths aggressively while
 * the landmark is slow (kills jitter at rest) and relaxes at high speed
 * (preserves the contact-frame peak). Input is not mutated.
 */
export function smoothFrames(
  frames: PoseFrame[],
  opts: SmoothOptions = {}
): PoseFrame[] {
  const {
    minCutoff = DEFAULT_MIN_CUTOFF,
    beta = DEFAULT_BETA,
    dcutoff = DEFAULT_DCUTOFF,
    visibilityThreshold = DEFAULT_VISIBILITY_THRESHOLD,
    minBboxArea = DEFAULT_MIN_BBOX_AREA,
    warmupDiscard = DEFAULT_WARMUP_DISCARD,
    lookaheadWindow = DEFAULT_LOOKAHEAD_WINDOW,
  } = opts

  if (frames.length === 0) return []

  let filtered = frames
  if (frames.length > WARMUP_MIN_TOTAL_FRAMES && warmupDiscard > 0) {
    filtered = frames.slice(warmupDiscard)
  }

  filtered = filtered.filter((f) =>
    isFrameConfident(f.landmarks, { visibilityThreshold, minBboxArea })
  )

  if (filtered.length === 0) return []

  return oneEuroSmooth(filtered, { minCutoff, beta, dcutoff }, lookaheadWindow)
}
