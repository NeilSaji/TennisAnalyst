import { describe, it, expect } from 'vitest'
import { LiveSwingDetector, type StreamedSwing } from '@/lib/liveSwingDetector'
import type { PoseFrame, JointAngles } from '@/lib/supabase'

function makeFrame(index: number, angles: Partial<JointAngles>, fps = 15): PoseFrame {
  return {
    frame_index: index,
    timestamp_ms: Math.round((index * 1000) / fps),
    landmarks: [],
    joint_angles: angles as JointAngles,
  }
}

// Drive a detector with a sequence of right_elbow angles and collect emissions.
function run(
  detector: LiveSwingDetector,
  elbowSeries: number[],
  startIndex = 0,
): { emissions: StreamedSwing[]; frames: PoseFrame[] } {
  const emissions: StreamedSwing[] = []
  const frames: PoseFrame[] = []
  elbowSeries.forEach((elbow, i) => {
    const frame = makeFrame(startIndex + i, { right_elbow: elbow })
    frames.push(frame)
    const out = detector.feed(frame)
    if (out) emissions.push(out)
  })
  return { emissions, frames }
}

// Build a sequence: N frames at baselineAngle, M frames ramping from
// baselineAngle at +degPerFrame, N frames flat at the final angle.
function pulse(
  restBefore: number,
  pulseLen: number,
  degPerFrame: number,
  restAfter: number,
  baselineAngle = 90,
): number[] {
  const out: number[] = []
  for (let i = 0; i < restBefore; i++) out.push(baselineAngle)
  let cur = baselineAngle
  for (let i = 0; i < pulseLen; i++) {
    cur += degPerFrame
    out.push(cur)
  }
  for (let i = 0; i < restAfter; i++) out.push(cur)
  return out
}

describe('LiveSwingDetector', () => {
  it('emits nothing during warmup', () => {
    const det = new LiveSwingDetector({ warmupFrames: 30 })
    const { emissions } = run(det, pulse(5, 20, 10, 10))
    // Pulse happens within the 30-frame warmup so no swing can be detected
    expect(emissions.length).toBe(0)
  })

  it('emits exactly one swing for a single activity pulse after warmup', () => {
    const det = new LiveSwingDetector()
    const series = [
      ...Array(40).fill(90),
      ...Array.from({ length: 20 }, (_, i) => 90 + (i + 1) * 10),
      ...Array(30).fill(290),
    ]
    const { emissions } = run(det, series)

    expect(emissions.length).toBe(1)
    const swing = emissions[0]
    expect(swing.swingIndex).toBe(1)
    expect(swing.startFrameIndex).toBeGreaterThanOrEqual(40)
    expect(swing.endFrameIndex).toBeLessThan(80)
    expect(swing.frames.length).toBeGreaterThanOrEqual(12)
    expect(swing.startMs).toBeLessThan(swing.endMs)
    expect(swing.peakFrameIndex).toBeGreaterThanOrEqual(swing.startFrameIndex)
    expect(swing.peakFrameIndex).toBeLessThanOrEqual(swing.endFrameIndex)
  })

  it('filters out pulses shorter than minSwingFrames', () => {
    const det = new LiveSwingDetector({ minSwingFrames: 20 })
    const series = [
      ...Array(40).fill(90),
      // Only 6 high-activity frames — below the 20-frame floor
      ...Array.from({ length: 6 }, (_, i) => 90 + (i + 1) * 10),
      ...Array(30).fill(150),
    ]
    const { emissions } = run(det, series)
    expect(emissions.length).toBe(0)
  })

  it('merges two pulses separated by a short gap into one swing', () => {
    const det = new LiveSwingDetector({ mergeGapFrames: 12 })
    // Two 15-frame bursts with a 6-frame rest between them — rest is below
    // mergeGap so the detector should treat them as one swing body
    const series = [
      ...Array(40).fill(90),
      ...Array.from({ length: 15 }, (_, i) => 90 + (i + 1) * 8),
      ...Array(6).fill(210),
      ...Array.from({ length: 15 }, (_, i) => 210 + (i + 1) * 8),
      ...Array(30).fill(330),
    ]
    const { emissions } = run(det, series)
    expect(emissions.length).toBe(1)
  })

  it('emits two swings when separated by a long enough refractory gap', () => {
    const det = new LiveSwingDetector({
      mergeGapFrames: 8,
      refractoryFrames: 10,
    })
    const series = [
      ...Array(40).fill(90),
      ...Array.from({ length: 20 }, (_, i) => 90 + (i + 1) * 10),
      // Long rest — longer than mergeGap + refractory + warmup-back-to-idle
      ...Array(60).fill(290),
      ...Array.from({ length: 20 }, (_, i) => 290 + (i + 1) * 10),
      ...Array(30).fill(490),
    ]
    const { emissions } = run(det, series)
    expect(emissions.length).toBe(2)
    expect(emissions[0].swingIndex).toBe(1)
    expect(emissions[1].swingIndex).toBe(2)
    expect(emissions[1].startFrameIndex).toBeGreaterThan(emissions[0].endFrameIndex)
  })

  it('emits nothing for low-activity noise below minTriggerAbs', () => {
    const det = new LiveSwingDetector({ minTriggerAbs: 2.0 })
    // Tiny alternating jitter — raw delta is ~0.2 per frame, smoothed still well below 2
    const series: number[] = []
    for (let i = 0; i < 200; i++) series.push(90 + (i % 2 === 0 ? 0 : 0.2))
    const { emissions } = run(det, series)
    expect(emissions.length).toBe(0)
  })

  it('resets cleanly between sessions', () => {
    const det = new LiveSwingDetector()
    const series1 = [
      ...Array(40).fill(90),
      ...Array.from({ length: 20 }, (_, i) => 90 + (i + 1) * 10),
      ...Array(30).fill(290),
    ]
    const first = run(det, series1, 0)
    expect(first.emissions.length).toBe(1)

    det.reset()

    // After reset the detector must re-warmup before emitting
    const earlyPulse = run(det, pulse(5, 20, 10, 5), 1000)
    expect(earlyPulse.emissions.length).toBe(0)

    // And a proper pulse after warmup still emits with swingIndex = 1
    const second = run(det, series1, 2000)
    expect(second.emissions.length).toBe(1)
    expect(second.emissions[0].swingIndex).toBe(1)
  })
})
