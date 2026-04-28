import { describe, it, expect } from 'vitest'
import { LiveSwingDetector, type StreamedSwing } from '@/lib/liveSwingDetector'
import type { PoseFrame, JointAngles, Landmark } from '@/lib/supabase'

// MediaPipe right-wrist landmark index, duplicated here to keep this
// test file self-contained (matches the value in lib/liveSwingDetector.ts).
// We only drive the right wrist in these fixtures.
const LM_RIGHT_WRIST = 16

function makeFrame(
  index: number,
  angles: Partial<JointAngles>,
  fps = 15,
  landmarks: Landmark[] = [],
): PoseFrame {
  return {
    frame_index: index,
    timestamp_ms: Math.round((index * 1000) / fps),
    landmarks,
    joint_angles: angles as JointAngles,
  }
}

function makeWrist(
  id: number,
  x: number,
  y: number,
  visibility = 0.95,
): Landmark {
  return { id, name: `landmark_${id}`, x, y, z: 0, visibility }
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

// Drive the detector with frames where both joint angles AND right-wrist
// landmark positions vary frame-to-frame. The wrist follows wristXSeries
// (and a fixed y), with `wristVis` as the visibility — pass 0.0 to
// simulate a completely-occluded wrist throughout. Set wristVis < 0 to
// omit the wrist landmark entirely.
function runWithWrist(
  detector: LiveSwingDetector,
  series: { elbow: number; wristX: number }[],
  opts: { startIndex?: number; wristVis?: number; wristY?: number } = {},
): { emissions: StreamedSwing[]; frames: PoseFrame[] } {
  const { startIndex = 0, wristVis = 0.95, wristY = 0.5 } = opts
  const emissions: StreamedSwing[] = []
  const frames: PoseFrame[] = []
  series.forEach(({ elbow, wristX }, i) => {
    const landmarks: Landmark[] =
      wristVis < 0 ? [] : [makeWrist(LM_RIGHT_WRIST, wristX, wristY, wristVis)]
    const frame = makeFrame(
      startIndex + i,
      { right_elbow: elbow },
      15,
      landmarks,
    )
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

  it('rejects an angle-only "swing" when the wrist barely moved (full body in frame)', () => {
    // Angle stream looks just like a real swing: 40 baseline frames, a
    // strong 20-frame ramp, and a long settle. But the wrist is locked
    // around x=0.5 with only ±0.005 jitter — well below the 0.12 default
    // displacement floor and the 0.012 peak-step floor. The wrist gate
    // must reject this candidate.
    const det = new LiveSwingDetector()
    const elbowSeries = [
      ...Array(40).fill(90),
      ...Array.from({ length: 20 }, (_, i) => 90 + (i + 1) * 10),
      ...Array(30).fill(290),
    ]
    const series = elbowSeries.map((elbow, i) => ({
      elbow,
      // Tiny tick-tock jitter; total spread is ~0.01.
      wristX: 0.5 + (i % 2 === 0 ? 0.005 : -0.005),
    }))
    const { emissions } = runWithWrist(det, series)
    expect(emissions.length).toBe(0)
  })

  it('emits a swing when both angles AND wrist position move clearly', () => {
    // Same angle ramp as the standing-still test, but this time the wrist
    // sweeps from x=0.2 to x=0.8 during the active body — total spread
    // 0.6, well above the 0.12 floor, with healthy per-frame steps.
    const det = new LiveSwingDetector()
    const elbowSeries = [
      ...Array(40).fill(90),
      ...Array.from({ length: 20 }, (_, i) => 90 + (i + 1) * 10),
      ...Array(30).fill(290),
    ]
    const series = elbowSeries.map((elbow, i) => {
      let wristX: number
      if (i < 40) wristX = 0.2
      else if (i < 60) wristX = 0.2 + ((i - 40) / 19) * 0.6 // 0.2 -> 0.8
      else wristX = 0.8
      return { elbow, wristX }
    })
    const { emissions } = runWithWrist(det, series)
    expect(emissions.length).toBe(1)
    expect(emissions[0].swingIndex).toBe(1)
  })

  it('still emits a swing when both wrists are occluded throughout (visibility fallback)', () => {
    // Angles ramp like a real swing AND the right-wrist landmark is
    // present on every frame BUT below the 0.5 visibility cutoff. The
    // wrist gate sees zero confident samples, so it must fall through
    // and let the angle gate decide.
    const det = new LiveSwingDetector()
    const elbowSeries = [
      ...Array(40).fill(90),
      ...Array.from({ length: 20 }, (_, i) => 90 + (i + 1) * 10),
      ...Array(30).fill(290),
    ]
    const series = elbowSeries.map((elbow, i) => ({
      elbow,
      wristX: 0.5 + (i % 2 === 0 ? 0.005 : -0.005),
    }))
    const { emissions } = runWithWrist(det, series, { wristVis: 0.2 })
    expect(emissions.length).toBe(1)
  })

  it('long-idle then small fidget does not emit (EMA-floored threshold)', () => {
    // Without an idle floor: 200+ flat frames make the activity-window
    // median collapse to ~0, the dynamic threshold falls to minTriggerAbs,
    // and a tiny fidget ramps just above that floor and reads as a swing.
    // With the EMA floor in place, the threshold stays anchored to the
    // long-running baseline so the small fidget can't trigger.
    const det = new LiveSwingDetector()
    const elbowSeries = [
      // 200 frames at exactly 90 — long enough that the activity window
      // saturates with zeros.
      ...Array(200).fill(90),
      // A small fidget: 20-frame ramp at +0.5°/frame -> peaks at 100.
      // Per-frame raw delta is 0.5°, smoothed delta plateaus at 0.5°,
      // sum across keys still tiny — but the *only* angle we drive is
      // right_elbow, so the per-frame activity is 0.5°. minTriggerAbs is
      // 2.0° so even without the EMA floor this would gate. To prove
      // the EMA-specific behavior we use an angle stream that *would*
      // sneak above the collapsed dynamic threshold but stays under the
      // EMA-floored one. Bumping the fidget to ~3°/frame -> 60° peak.
    ]
    const fidget = Array.from({ length: 20 }, (_, i) => 90 + (i + 1) * 3)
    const tail = Array(40).fill(fidget[fidget.length - 1])
    const full = [...elbowSeries, ...fidget, ...tail]
    // We use a generous wrist sweep so any rejection comes from the angle
    // floor, not the wrist gate — this test is about the EMA floor only.
    const series = full.map((elbow, i) => ({
      elbow,
      wristX: 0.2 + (i / Math.max(1, full.length - 1)) * 0.6,
    }))
    const { emissions } = runWithWrist(det, series)
    expect(emissions.length).toBe(0)
  })

  it('warmup with body-less frames (no angles) does not progress to idle', () => {
    // Simulate face-only frames at the start: empty joint_angles. ingest
    // sees no prevAngles -> rawDelta=0 every frame, but more importantly
    // each frame is "invalid" for warmup-counting purposes. The detector
    // must stay in warmup until valid full-body frames arrive.
    const det = new LiveSwingDetector({ warmupFrames: 30 })

    // 50 face-only frames first.
    for (let i = 0; i < 50; i++) {
      const frame = makeFrame(i, {}, 15)
      const out = det.feed(frame)
      expect(out).toBeNull()
    }

    // Now feed a strong activity pulse. If warmup had wrongly exited on
    // the empty frames, this pulse would emit. With the validity check
    // it should not, because we're still warming up.
    const pulseSeries = [
      ...Array(10).fill(90),
      ...Array.from({ length: 25 }, (_, i) => 90 + (i + 1) * 10),
      ...Array(20).fill(340),
    ]
    let firstEmission: StreamedSwing | null = null
    pulseSeries.forEach((elbow, i) => {
      const frame = makeFrame(50 + i, { right_elbow: elbow })
      const out = det.feed(frame)
      if (out && !firstEmission) firstEmission = out
    })
    // With proper validity gating the detector spends most of this stretch
    // collecting its first real warmup frames; the activity pulse arrives
    // before warmup completes, so no swing fires.
    expect(firstEmission).toBeNull()
  })
})
