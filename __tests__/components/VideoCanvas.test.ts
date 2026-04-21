import { describe, it, expect } from 'vitest'
import { getFrameAtTime } from '@/components/VideoCanvas'
import { makeLandmark, makeFrame } from '../helpers'

function frameAt(indexAndMs: number, x: number, angle: number) {
  return makeFrame(
    indexAndMs,
    indexAndMs,
    [makeLandmark(0, x, 0.5, 1.0, 0)],
    { right_elbow: angle },
  )
}

describe('getFrameAtTime', () => {
  it('returns null for an empty framesData array', () => {
    expect(getFrameAtTime([], 1.0)).toBeNull()
  })

  it('clamps to the first frame when time is before the first sample', () => {
    const frames = [frameAt(100, 0.1, 90), frameAt(200, 0.2, 100)]
    expect(getFrameAtTime(frames, 0.05)).toBe(frames[0])
  })

  it('clamps to the last frame when time is past the last sample', () => {
    const frames = [frameAt(100, 0.1, 90), frameAt(200, 0.2, 100)]
    expect(getFrameAtTime(frames, 5.0)).toBe(frames[1])
  })

  it('returns the nearest sample when time falls between two frames', () => {
    // samples at 100ms and 200ms; 140ms is closer to 100ms than 200ms.
    const frames = [frameAt(100, 0.10, 90), frameAt(200, 0.20, 110)]
    expect(getFrameAtTime(frames, 0.14)).toBe(frames[0])
    // 170ms is closer to 200ms.
    expect(getFrameAtTime(frames, 0.17)).toBe(frames[1])
  })

  it('breaks ties toward the previous frame at exact midpoint', () => {
    const frames = [frameAt(100, 0.10, 90), frameAt(200, 0.20, 110)]
    // midpoint 150ms: dPrev=50, dCur=50; implementation picks prev (dCur<dPrev is false)
    expect(getFrameAtTime(frames, 0.15)).toBe(frames[0])
  })

  it('returns the frame whose timestamp exactly matches', () => {
    const frames = [frameAt(100, 0.10, 90), frameAt(200, 0.20, 110)]
    // 200ms exactly -> the second frame via the post-loop clamp
    expect(getFrameAtTime(frames, 0.2)).toBe(frames[1])
  })

  it('does not synthesize a position between samples', () => {
    // This is the regression guard: getFrameAtTime must not return an
    // interpolated frame whose landmarks differ from both bracketing
    // samples. That would reintroduce the "predict where joints are when
    // they're not in view" behavior we deliberately removed.
    const frames = [frameAt(100, 0.1, 90), frameAt(200, 0.9, 110)]
    const got = getFrameAtTime(frames, 0.15)
    expect(got).not.toBeNull()
    const x = got!.landmarks[0].x
    expect([0.1, 0.9]).toContain(x)
  })
})
