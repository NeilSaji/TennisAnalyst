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
// Body-presence gate
// ---------------------------------------------------------------------------

// MediaPipe Pose landmark indices used by the body-presence gate. We
// duplicate them here rather than importing LANDMARK_INDICES to keep
// poseSmoothing.ts free of cross-module deps that would create cycles
// (jointAngles.ts already pulls from this file's neighbors).
const BODY_LEFT_SHOULDER = 11
const BODY_RIGHT_SHOULDER = 12
const BODY_LEFT_WRIST = 15
const BODY_RIGHT_WRIST = 16
const BODY_LEFT_HIP = 23
const BODY_RIGHT_HIP = 24

/** Default per-landmark visibility floor for the strict body-presence gate. */
const DEFAULT_BODY_VISIBILITY = 0.5
/**
 * Default minimum vertical extent (max-y minus min-y, normalized [0,1])
 * across the required landmarks. 0.35 rejects "head + shoulders only"
 * crops where the player's torso/legs aren't actually in frame.
 */
const DEFAULT_MIN_VERTICAL_EXTENT = 0.35

export interface BodyVisibilityOptions {
  /** Per-landmark visibility floor. Default 0.5. */
  visibilityFloor?: number
  /** Min vertical extent of the required landmarks. Default 0.35. */
  minVerticalExtent?: number
}

/**
 * Strict body-presence gate. Returns true only when both shoulders, both
 * hips, AND at least one wrist are visible at >= visibilityFloor AND the
 * vertical extent across those required landmarks is at least
 * minVerticalExtent.
 *
 * Sits next to (not in place of) `isFrameConfident`, which gates on the
 * average visibility of all 33 landmarks. The face has 11 stable points
 * that drag the average above 0.4 even when hips/legs are at 0.1, so a
 * face-only frame can pass `isFrameConfident`. We use this stricter gate
 * to decide whether a frame is allowed to drive the live swing detector.
 */
export function isBodyVisible(
  landmarks: Landmark[],
  opts: BodyVisibilityOptions = {},
): boolean {
  const {
    visibilityFloor = DEFAULT_BODY_VISIBILITY,
    minVerticalExtent = DEFAULT_MIN_VERTICAL_EXTENT,
  } = opts

  if (landmarks.length === 0) return false

  const byId = new Map<number, Landmark>()
  for (const lm of landmarks) byId.set(lm.id, lm)

  const lShoulder = byId.get(BODY_LEFT_SHOULDER)
  const rShoulder = byId.get(BODY_RIGHT_SHOULDER)
  const lHip = byId.get(BODY_LEFT_HIP)
  const rHip = byId.get(BODY_RIGHT_HIP)
  const lWrist = byId.get(BODY_LEFT_WRIST)
  const rWrist = byId.get(BODY_RIGHT_WRIST)

  // Both shoulders + both hips required.
  if (!lShoulder || lShoulder.visibility < visibilityFloor) return false
  if (!rShoulder || rShoulder.visibility < visibilityFloor) return false
  if (!lHip || lHip.visibility < visibilityFloor) return false
  if (!rHip || rHip.visibility < visibilityFloor) return false

  // At least one wrist required.
  const lWristOk = !!lWrist && lWrist.visibility >= visibilityFloor
  const rWristOk = !!rWrist && rWrist.visibility >= visibilityFloor
  if (!lWristOk && !rWristOk) return false

  // Vertical extent over the required landmarks. Only count the wrist(s)
  // that actually cleared the visibility floor — a hidden wrist contributes
  // no spatial information.
  const ys: number[] = [lShoulder.y, rShoulder.y, lHip.y, rHip.y]
  if (lWristOk) ys.push(lWrist!.y)
  if (rWristOk) ys.push(rWrist!.y)

  let minY = Infinity
  let maxY = -Infinity
  for (const y of ys) {
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  if (maxY - minY < minVerticalExtent) return false

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
  /**
   * Zero-phase filtering: apply the One Euro filter forward, then on the
   * time-reversed sequence, then re-reverse. Cancels the ~50-80ms phase
   * lag that a causal IIR produces on fast motion, which was baking a
   * visible "skeleton lags the body" offset into cached keypoints.
   * Default true. Set false to keep single-pass causal behavior (used by
   * tests that assert the step-by-step causal response shape).
   */
  zeroPhase?: boolean
}

// Reverse a frame sequence in time so oneEuroSmooth can run it as the
// backward half of a filtfilt pair. Order is flipped and each timestamp
// is remapped to `lastTs - timestamp_ms` so deltas stay positive (the
// filter would otherwise fall back to FALLBACK_DT every step).
// Applying this function twice restores the original order and timestamps.
function reverseFramesForFiltfilt(frames: PoseFrame[]): PoseFrame[] {
  if (frames.length === 0) return []
  const lastTs = frames[frames.length - 1].timestamp_ms
  const out: PoseFrame[] = new Array(frames.length)
  for (let i = frames.length - 1, j = 0; i >= 0; i--, j++) {
    const f = frames[i]
    out[j] = { ...f, timestamp_ms: lastTs - f.timestamp_ms }
  }
  return out
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
    zeroPhase = true,
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

  const params = { minCutoff, beta, dcutoff }
  const forward = oneEuroSmooth(filtered, params, lookaheadWindow)
  if (!zeroPhase) return forward

  // Forward pass lags by +τ. Running the same causal filter over the
  // time-reversed sequence applies a +τ shift in the reversed timeline,
  // which is -τ in the original timeline. Combined: zero net phase.
  // Edge transients from both passes are ≤ lookaheadWindow frames, which
  // matches the existing warmup discard behavior at the start of a clip.
  const reversed = reverseFramesForFiltfilt(forward)
  const reverseSmoothed = oneEuroSmooth(reversed, params, lookaheadWindow)
  return reverseFramesForFiltfilt(reverseSmoothed)
}

// ---------------------------------------------------------------------------
// Bone-length plausibility
// ---------------------------------------------------------------------------

// MediaPipe landmark indices we care about for arm-bone checks. Duplicated
// here to avoid a circular dep with lib/jointAngles.ts (which imports this
// module via poseExtraction).
const LM_LEFT_SHOULDER = 11
const LM_RIGHT_SHOULDER = 12
const LM_LEFT_ELBOW = 13
const LM_RIGHT_ELBOW = 14
const LM_LEFT_WRIST = 15
const LM_RIGHT_WRIST = 16

// Only use landmarks above this visibility when measuring bone length, and
// only measure when BOTH endpoints clear the bar. Anything dimmer is a
// guess and would pollute the median.
const BONE_MEDIAN_VIS_GATE = 0.7
// Shortest/longest acceptable ratio of per-frame bone length to the clip's
// median for that bone. Tuned generous: normal within-clip length variation
// from perspective foreshortening is small (the camera doesn't move), so
// anything outside this range is almost certainly a blurred/misplaced joint.
// Widened from 0.65/1.5 -> 0.55/1.75: the tighter band was zeroing elbow/
// wrist visibility on frames with legitimate racket-arm extension, making
// the skeleton drop joints during the exact moment (contact / follow-through)
// the user most wants to see. Still catches gross misplacements (2x+ off).
const BONE_LOW_TOL = 0.55
const BONE_HIGH_TOL = 1.75
// Need at least this many frames with a clean measurement to trust the
// median. Below that, the filter no-ops (nothing zeroed) rather than
// zeroing everything against a noisy sample.
const MIN_SAMPLES_FOR_MEDIAN = 5

function distance2D(
  a: { x: number; y: number },
  b: { x: number; y: number },
): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return Math.sqrt(dx * dx + dy * dy)
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0
  const sorted = [...arr].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid]
}

/**
 * Zero the visibility of elbow/wrist landmarks whose shoulder→elbow or
 * elbow→wrist distance is implausible compared to the clip's median for
 * that bone on that side.
 *
 * Rationale: MediaPipe's elbow/wrist detections go to hell on fast-moving
 * blurred arms, while shoulders stay solid. A tennis player's upper-arm
 * and forearm lengths don't meaningfully change within a clip (camera is
 * fixed, limb doesn't grow), so a frame where the upper arm is suddenly
 * 2x or 0.4x the usual length is almost certainly a misplaced landmark.
 * We zero its visibility so the render-side 0.6 cutoff drops it — the
 * overlay shows partial skeletons through contact rather than wrong ones,
 * which is the "don't predict, show accuracy" posture the user wants.
 *
 * Caveats: on very short clips (<MIN_SAMPLES_FOR_MEDIAN solid frames per
 * bone) the median is untrustworthy and the filter no-ops. That's
 * deliberate — better to keep the raw detection than to filter against
 * noise.
 */
export function filterImplausibleArmJoints(frames: PoseFrame[]): PoseFrame[] {
  if (frames.length === 0) return frames

  // Collect per-bone length samples from frames where both endpoints are
  // confidently detected. Separate arrays per side since a player's left
  // and right upper arms can differ slightly on a side-view clip due to
  // foreshortening of the far arm.
  const leftUpper: number[] = []
  const rightUpper: number[] = []
  const leftFore: number[] = []
  const rightFore: number[] = []

  const lmOf = (f: PoseFrame, id: number) =>
    f.landmarks.find((lm) => lm.id === id)

  for (const f of frames) {
    const ls = lmOf(f, LM_LEFT_SHOULDER)
    const le = lmOf(f, LM_LEFT_ELBOW)
    const lw = lmOf(f, LM_LEFT_WRIST)
    const rs = lmOf(f, LM_RIGHT_SHOULDER)
    const re = lmOf(f, LM_RIGHT_ELBOW)
    const rw = lmOf(f, LM_RIGHT_WRIST)

    if (
      ls && le &&
      ls.visibility >= BONE_MEDIAN_VIS_GATE &&
      le.visibility >= BONE_MEDIAN_VIS_GATE
    ) {
      leftUpper.push(distance2D(ls, le))
    }
    if (
      le && lw &&
      le.visibility >= BONE_MEDIAN_VIS_GATE &&
      lw.visibility >= BONE_MEDIAN_VIS_GATE
    ) {
      leftFore.push(distance2D(le, lw))
    }
    if (
      rs && re &&
      rs.visibility >= BONE_MEDIAN_VIS_GATE &&
      re.visibility >= BONE_MEDIAN_VIS_GATE
    ) {
      rightUpper.push(distance2D(rs, re))
    }
    if (
      re && rw &&
      re.visibility >= BONE_MEDIAN_VIS_GATE &&
      rw.visibility >= BONE_MEDIAN_VIS_GATE
    ) {
      rightFore.push(distance2D(re, rw))
    }
  }

  const leftUpperMed = leftUpper.length >= MIN_SAMPLES_FOR_MEDIAN ? median(leftUpper) : null
  const leftForeMed = leftFore.length >= MIN_SAMPLES_FOR_MEDIAN ? median(leftFore) : null
  const rightUpperMed = rightUpper.length >= MIN_SAMPLES_FOR_MEDIAN ? median(rightUpper) : null
  const rightForeMed = rightFore.length >= MIN_SAMPLES_FOR_MEDIAN ? median(rightFore) : null

  // If we don't have enough clean samples for any bone, skip the pass
  // entirely rather than filter against noise.
  if (
    leftUpperMed === null &&
    leftForeMed === null &&
    rightUpperMed === null &&
    rightForeMed === null
  ) {
    return frames
  }

  const outOfRange = (d: number, med: number) =>
    d < med * BONE_LOW_TOL || d > med * BONE_HIGH_TOL

  return frames.map((f) => {
    // Shallow-copy landmarks before mutating so input isn't touched.
    const landmarks = f.landmarks.map((lm) => ({ ...lm }))
    const byId = new Map<number, (typeof landmarks)[number]>()
    for (const lm of landmarks) byId.set(lm.id, lm)

    const checkBone = (
      parentId: number,
      childId: number,
      med: number | null,
    ) => {
      if (med === null) return
      const p = byId.get(parentId)
      const c = byId.get(childId)
      if (!p || !c) return
      // Only penalize the child end. The parent (shoulder/elbow) may be
      // fine on its own — we only know its pairing looked weird. Zeroing
      // both ends would take out a good shoulder because of a bad elbow.
      if (outOfRange(distance2D(p, c), med)) {
        c.visibility = 0
      }
    }

    checkBone(LM_LEFT_SHOULDER, LM_LEFT_ELBOW, leftUpperMed)
    checkBone(LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW, rightUpperMed)
    checkBone(LM_LEFT_ELBOW, LM_LEFT_WRIST, leftForeMed)
    checkBone(LM_RIGHT_ELBOW, LM_RIGHT_WRIST, rightForeMed)

    return { ...f, landmarks }
  })
}
