import type { PoseFrame, JointAngles } from './supabase'

// The same six joints used by detectSwings() in lib/jointAngles.ts so live
// detection and post-hoc segmentation agree on what "activity" means.
const ACTIVITY_KEYS: (keyof JointAngles)[] = [
  'right_elbow',
  'left_elbow',
  'right_shoulder',
  'left_shoulder',
  'hip_rotation',
  'trunk_rotation',
]

// MediaPipe wrist landmark indices. Used by the wrist-displacement gate
// in finalizeSwing to reject candidates where the wrist barely moved.
const LM_LEFT_WRIST = 15
const LM_RIGHT_WRIST = 16
// Visibility cutoff for sampling a wrist position. Matches PoseRenderer's
// 0.5 cutoff so live detection and the on-screen overlay agree on which
// frames they trust.
const WRIST_VISIBILITY_GATE = 0.5

export type StreamedSwing = {
  swingIndex: number
  startFrameIndex: number
  endFrameIndex: number
  peakFrameIndex: number
  startMs: number
  endMs: number
  frames: PoseFrame[]
}

export interface LiveSwingDetectorOptions {
  windowFrames?: number
  warmupFrames?: number
  triggerK?: number
  smoothingWindow?: number
  minCandidateFrames?: number
  mergeGapFrames?: number
  refractoryFrames?: number
  minSwingFrames?: number
  maxSwingFrames?: number
  minTriggerAbs?: number
  minPeakRatio?: number
  /**
   * Minimum total wrist displacement (max-min spread of wrist position over
   * the swing body) for at least one wrist, normalized to frame coordinates.
   * Default 0.12 (12% of the frame). Falls back to the angle-only gate when
   * both wrists are occluded throughout — we don't want to punish brief
   * tracking dips during a real swing.
   */
  minWristDisplacement?: number
  /**
   * Minimum single frame-to-frame wrist jump (peak step) for at least one
   * wrist, normalized to frame coordinates. Default 0.012 (1.2% of the
   * frame). Catches the "slow pendulum" failure mode that can otherwise
   * hit minWristDisplacement through accumulated drift.
   */
  minWristPeakStep?: number
}

type State = 'warmup' | 'idle' | 'candidate' | 'active' | 'refractory'

// Streaming swing detector. Fed one PoseFrame at a time from the capture loop;
// returns a StreamedSwing on the frame that closes out a detected swing (the
// player has stopped for `mergeGapFrames`), else null.
//
// Can't reuse detectSwings() because that function computes median/max over
// the whole buffer — a luxury we don't have at frame time. The per-frame
// activity formula is identical (sum of |Δ| across six angle keys) so live
// and post-hoc segmentation stay consistent.
export class LiveSwingDetector {
  private readonly opts: Required<LiveSwingDetectorOptions>
  private activityWindow: number[] = []
  private smoothingBuf: number[] = []
  private prevAngles: JointAngles | null = null

  private state: State = 'warmup'
  private candidateCount = 0
  private refractoryCount = 0
  private swingCount = 0
  // Counts only frames whose ingest produced a real angle delta — i.e.
  // frames where prevAngles was non-null after the previous step. Body-less
  // or empty frames don't pad warmup; otherwise the detector exits warmup
  // with no real baseline and the next motion reads as a swing.
  private validWarmupCount = 0

  private peakActivity = 0
  private peakFrameIdx = -1
  private belowCount = 0
  private pendingFrames: PoseFrame[] = []

  // EMA of the per-frame median activity, used to clamp the dynamic
  // threshold so a long idle window can't collapse it. Computed at ~15fps
  // over ~60s (~900 frames). alpha = 1 - exp(-1/900) ~= 0.001110, which
  // gives the EMA ~63% weight from the most recent ~900 samples.
  //
  // We start at 0 and let the EMA crawl up — seeding on first sample lets
  // a transient spike (e.g. the discontinuity when angles jump at the
  // start of a session) dominate the floor for hundreds of frames. The
  // dynamic threshold + the absolute minTriggerAbs are enough cover until
  // the EMA settles.
  private medianEMA = 0
  private static readonly MEDIAN_EMA_ALPHA = 1 - Math.exp(-1 / 900)
  // How aggressively the EMA floors the dynamic threshold. Ratio chosen
  // so the threshold can't drop more than ~1.5x below the running median
  // baseline, which is what stops the "long idle then sneeze = swing"
  // failure mode without making a real swing's threshold gate too stiff.
  private static readonly MEDIAN_EMA_FLOOR_K = 1.5

  constructor(opts: LiveSwingDetectorOptions = {}) {
    this.opts = {
      windowFrames: opts.windowFrames ?? 90,
      warmupFrames: opts.warmupFrames ?? 30,
      triggerK: opts.triggerK ?? 0.6,
      smoothingWindow: opts.smoothingWindow ?? 5,
      minCandidateFrames: opts.minCandidateFrames ?? 3,
      mergeGapFrames: opts.mergeGapFrames ?? 10,
      refractoryFrames: opts.refractoryFrames ?? 30,
      minSwingFrames: opts.minSwingFrames ?? 12,
      maxSwingFrames: opts.maxSwingFrames ?? 150,
      minTriggerAbs: opts.minTriggerAbs ?? 2.0,
      minPeakRatio: opts.minPeakRatio ?? 2.0,
      minWristDisplacement: opts.minWristDisplacement ?? 0.12,
      minWristPeakStep: opts.minWristPeakStep ?? 0.012,
    }
  }

  reset(): void {
    this.activityWindow = []
    this.smoothingBuf = []
    this.prevAngles = null
    this.state = 'warmup'
    this.candidateCount = 0
    this.refractoryCount = 0
    this.swingCount = 0
    this.validWarmupCount = 0
    this.peakActivity = 0
    this.peakFrameIdx = -1
    this.belowCount = 0
    this.pendingFrames = []
    this.medianEMA = 0
  }

  feed(frame: PoseFrame): StreamedSwing | null {
    const { smoothed, valid } = this.ingest(frame)

    if (this.state === 'warmup') {
      // Only frames where ingest produced a real angle-delta count toward
      // warmup. A run of body-less or empty-angle frames otherwise exits
      // warmup with no baseline, and the first real motion looks like a
      // swing.
      if (valid) this.validWarmupCount++
      if (this.validWarmupCount >= this.opts.warmupFrames) {
        this.state = 'idle'
      }
      return null
    }

    const { median, threshold } = this.thresholdStats()
    const isAbove = smoothed >= threshold

    switch (this.state) {
      case 'idle':
        if (isAbove) {
          this.state = 'candidate'
          this.candidateCount = 1
          this.peakActivity = smoothed
          this.peakFrameIdx = frame.frame_index
          this.pendingFrames = [frame]
        }
        return null

      case 'candidate':
        this.pendingFrames.push(frame)
        if (isAbove) {
          this.candidateCount++
          if (smoothed > this.peakActivity) {
            this.peakActivity = smoothed
            this.peakFrameIdx = frame.frame_index
          }
          if (this.candidateCount >= this.opts.minCandidateFrames) {
            this.state = 'active'
            this.belowCount = 0
          }
        } else {
          this.state = 'idle'
          this.pendingFrames = []
          this.candidateCount = 0
        }
        return null

      case 'active':
        this.pendingFrames.push(frame)
        if (isAbove) {
          this.belowCount = 0
          if (smoothed > this.peakActivity) {
            this.peakActivity = smoothed
            this.peakFrameIdx = frame.frame_index
          }
        } else {
          this.belowCount++
          const hitGap = this.belowCount >= this.opts.mergeGapFrames
          const hitMax = this.pendingFrames.length >= this.opts.maxSwingFrames
          if (hitGap || hitMax) {
            return this.finalizeSwing(median)
          }
        }
        return null

      case 'refractory':
        this.refractoryCount++
        if (this.refractoryCount >= this.opts.refractoryFrames) {
          this.state = 'idle'
          this.refractoryCount = 0
        }
        return null
    }
  }

  // Advance the smoothing and activity-window buffers with a new frame.
  // Returns the smoothed activity value for this frame and a `valid` flag
  // that tells `feed` whether this frame contributed a real angle delta.
  // A frame is valid iff prevAngles was non-null AND at least one
  // ACTIVITY_KEY had a numeric value in BOTH the previous and current
  // frame -- empty joint_angles ({}) on a face-only frame would otherwise
  // satisfy `prevAngles != null` and pad warmup with garbage.
  private ingest(frame: PoseFrame): { smoothed: number; valid: boolean } {
    let rawDelta = 0
    let valid = false
    if (this.prevAngles) {
      for (const k of ACTIVITY_KEYS) {
        const p = this.prevAngles[k]
        const c = frame.joint_angles?.[k]
        if (p != null && c != null) {
          rawDelta += Math.abs(c - p)
          valid = true
        }
      }
    }
    this.prevAngles = frame.joint_angles ?? null

    this.smoothingBuf.push(rawDelta)
    if (this.smoothingBuf.length > this.opts.smoothingWindow) this.smoothingBuf.shift()
    const smoothed =
      this.smoothingBuf.reduce((a, b) => a + b, 0) / this.smoothingBuf.length

    this.activityWindow.push(smoothed)
    if (this.activityWindow.length > this.opts.windowFrames) this.activityWindow.shift()
    return { smoothed, valid }
  }

  // Rolling window stats. Sort is O(n log n) per frame; n <= windowFrames (90
  // by default), called at ≤15 fps, so ~1350 comparisons/s — negligible.
  //
  // The `threshold` returned here is clamped against an EMA of the per-frame
  // median over the last ~60s. Without that floor, a long idle window
  // saturates the rolling buffer with low-activity samples; both `median`
  // and `p90` collapse, the dynamic threshold falls to its minTriggerAbs
  // floor, and the next mild fidget reads as a swing. The EMA is the slow
  // baseline of "what's quiet supposed to look like" and we won't let the
  // threshold fall meaningfully below it.
  private thresholdStats(): { median: number; p90: number; threshold: number } {
    const sorted = [...this.activityWindow].sort((a, b) => a - b)
    const medianIdx = Math.floor(sorted.length / 2)
    const p90Idx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))
    const median = sorted[medianIdx]
    const p90 = sorted[p90Idx]

    // Update the median EMA. Starts at 0 and crawls toward the true
    // baseline — early-frame transients (like a discontinuity at session
    // start) get only their natural weight in the average rather than
    // being seeded as the floor.
    const a = LiveSwingDetector.MEDIAN_EMA_ALPHA
    this.medianEMA = a * median + (1 - a) * this.medianEMA

    const dynamicThreshold = median + this.opts.triggerK * (p90 - median)
    const emaFloor =
      LiveSwingDetector.MEDIAN_EMA_FLOOR_K * this.medianEMA +
      this.opts.minTriggerAbs / 2
    const threshold = Math.max(
      dynamicThreshold,
      this.opts.minTriggerAbs,
      emaFloor,
    )
    return { median, p90, threshold }
  }

  private finalizeSwing(median: number): StreamedSwing | null {
    const body = this.pendingFrames.slice(
      0,
      Math.max(0, this.pendingFrames.length - this.belowCount),
    )

    const keepRefractory = () => {
      this.state = 'refractory'
      this.refractoryCount = 0
      this.pendingFrames = []
      this.candidateCount = 0
      this.belowCount = 0
    }

    // Too-short burst — drop without incrementing swingCount
    if (body.length < this.opts.minSwingFrames) {
      keepRefractory()
      return null
    }
    // Weak burst that barely crept above baseline — drop as noise
    const peakRatio = median > 0 ? this.peakActivity / median : this.opts.minPeakRatio
    if (peakRatio < this.opts.minPeakRatio) {
      keepRefractory()
      return null
    }

    // Wrist-displacement gate. Standing micro-motion / breathing / stance
    // shifts can pass the angle gate. Real swings move the wrist through
    // significant frame space. We gate on BOTH the total wrist spread AND
    // a single-frame peak step; the spread alone could be hit by slow
    // pendulum sway accumulating over a long body, and the peak-step alone
    // could be hit by a single tracking blip.
    //
    // Visibility-gated samples only — a wrist landmark below 0.5 vis is a
    // guess (PoseRenderer uses the same cutoff). If both wrists are
    // occluded throughout the body window, fall through (don't punish brief
    // tracking dips during a real swing — the angle gate already filtered).
    const wristMetrics = this.computeWristMetrics(body)
    if (wristMetrics.anyWristSampled) {
      const maxDisplacement = Math.max(
        wristMetrics.leftDisplacement,
        wristMetrics.rightDisplacement,
      )
      const maxPeakStep = Math.max(
        wristMetrics.leftPeakStep,
        wristMetrics.rightPeakStep,
      )
      if (
        maxDisplacement < this.opts.minWristDisplacement ||
        maxPeakStep < this.opts.minWristPeakStep
      ) {
        keepRefractory()
        return null
      }
    }

    this.swingCount++
    const startFrame = body[0]
    const endFrame = body[body.length - 1]
    const swing: StreamedSwing = {
      swingIndex: this.swingCount,
      startFrameIndex: startFrame.frame_index,
      endFrameIndex: endFrame.frame_index,
      peakFrameIndex: this.peakFrameIdx,
      startMs: startFrame.timestamp_ms,
      endMs: endFrame.timestamp_ms,
      frames: body,
    }

    keepRefractory()
    return swing
  }

  /**
   * Compute per-wrist motion metrics across the swing body. For each side,
   * we look at the bounding-box diagonal of all visibility-gated wrist
   * samples (total displacement) and the largest frame-to-frame jump
   * between consecutive visibility-gated samples (peak step).
   *
   * `anyWristSampled` is false only when both wrists were occluded for
   * every body frame — finalizeSwing falls through to the angle-only gate
   * in that case.
   */
  private computeWristMetrics(body: PoseFrame[]): {
    leftDisplacement: number
    rightDisplacement: number
    leftPeakStep: number
    rightPeakStep: number
    anyWristSampled: boolean
  } {
    const left: { x: number; y: number }[] = []
    const right: { x: number; y: number }[] = []

    for (const f of body) {
      for (const lm of f.landmarks) {
        if (lm.visibility < WRIST_VISIBILITY_GATE) continue
        if (lm.id === LM_LEFT_WRIST) left.push({ x: lm.x, y: lm.y })
        else if (lm.id === LM_RIGHT_WRIST) right.push({ x: lm.x, y: lm.y })
      }
    }

    const summary = (samples: { x: number; y: number }[]) => {
      if (samples.length < 2) return { displacement: 0, peakStep: 0 }
      let minX = Infinity
      let minY = Infinity
      let maxX = -Infinity
      let maxY = -Infinity
      let peakStep = 0
      for (let i = 0; i < samples.length; i++) {
        const s = samples[i]
        if (s.x < minX) minX = s.x
        if (s.y < minY) minY = s.y
        if (s.x > maxX) maxX = s.x
        if (s.y > maxY) maxY = s.y
        if (i > 0) {
          const dx = s.x - samples[i - 1].x
          const dy = s.y - samples[i - 1].y
          const step = Math.sqrt(dx * dx + dy * dy)
          if (step > peakStep) peakStep = step
        }
      }
      const dx = maxX - minX
      const dy = maxY - minY
      return { displacement: Math.sqrt(dx * dx + dy * dy), peakStep }
    }

    const ls = summary(left)
    const rs = summary(right)
    return {
      leftDisplacement: ls.displacement,
      rightDisplacement: rs.displacement,
      leftPeakStep: ls.peakStep,
      rightPeakStep: rs.peakStep,
      anyWristSampled: left.length > 0 || right.length > 0,
    }
  }
}
