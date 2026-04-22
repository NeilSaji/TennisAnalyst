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

  private peakActivity = 0
  private peakFrameIdx = -1
  private belowCount = 0
  private pendingFrames: PoseFrame[] = []

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
    this.peakActivity = 0
    this.peakFrameIdx = -1
    this.belowCount = 0
    this.pendingFrames = []
  }

  feed(frame: PoseFrame): StreamedSwing | null {
    const smoothed = this.ingest(frame)

    if (this.state === 'warmup') {
      if (this.activityWindow.length >= this.opts.warmupFrames) {
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

  // Advance the smoothing and activity-window buffers with a new frame. Returns
  // the smoothed activity value for this frame.
  private ingest(frame: PoseFrame): number {
    let rawDelta = 0
    if (this.prevAngles) {
      for (const k of ACTIVITY_KEYS) {
        const p = this.prevAngles[k]
        const c = frame.joint_angles?.[k]
        if (p != null && c != null) rawDelta += Math.abs(c - p)
      }
    }
    this.prevAngles = frame.joint_angles ?? null

    this.smoothingBuf.push(rawDelta)
    if (this.smoothingBuf.length > this.opts.smoothingWindow) this.smoothingBuf.shift()
    const smoothed =
      this.smoothingBuf.reduce((a, b) => a + b, 0) / this.smoothingBuf.length

    this.activityWindow.push(smoothed)
    if (this.activityWindow.length > this.opts.windowFrames) this.activityWindow.shift()
    return smoothed
  }

  // Rolling window stats. Sort is O(n log n) per frame; n <= windowFrames (90
  // by default), called at ≤15 fps, so ~1350 comparisons/s — negligible.
  private thresholdStats(): { median: number; p90: number; threshold: number } {
    const sorted = [...this.activityWindow].sort((a, b) => a - b)
    const medianIdx = Math.floor(sorted.length / 2)
    const p90Idx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))
    const median = sorted[medianIdx]
    const p90 = sorted[p90Idx]
    const threshold = Math.max(
      median + this.opts.triggerK * (p90 - median),
      this.opts.minTriggerAbs,
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
}
