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

  it('returns the nearer-in-time bracketing sample', () => {
    // Samples at 100ms and 200ms. Nearest-neighbor: 140ms is closer to
    // 100ms (40ms vs 60ms), 170ms is closer to 200ms (30ms vs 70ms).
    // Switching from strict-previous halves the worst-case visible drift
    // between the player and the skeleton overlay during fast motion.
    const frames = [frameAt(100, 0.10, 90), frameAt(200, 0.20, 110)]
    expect(getFrameAtTime(frames, 0.14)).toBe(frames[0])
    expect(getFrameAtTime(frames, 0.17)).toBe(frames[1])
    // 200ms exactly resolves to the frame at 200 (the post-loop clamp).
    expect(getFrameAtTime(frames, 0.2)).toBe(frames[1])
  })

  it('breaks ties by returning the earlier sample', () => {
    // 150ms is exactly between 100ms and 200ms (50ms each way). Tie
    // broken in favor of the earlier sample (dPrev <= dNext).
    const frames = [frameAt(100, 0.10, 90), frameAt(200, 0.20, 110)]
    expect(getFrameAtTime(frames, 0.15)).toBe(frames[0])
  })

  it('returns the frame whose timestamp exactly matches', () => {
    const frames = [frameAt(100, 0.10, 90), frameAt(200, 0.20, 110)]
    // 200ms exactly -> the second frame via the post-loop clamp
    expect(getFrameAtTime(frames, 0.2)).toBe(frames[1])
  })

  it('does not synthesize a position between samples', () => {
    // Regression guard: even with nearest-neighbor we never interpolate
    // — the returned frame's landmarks must equal one of the two
    // bracketing samples exactly. (Synthesizing positions reintroduces
    // the predictive-racket bug class from earlier.)
    const frames = [frameAt(100, 0.1, 90), frameAt(200, 0.9, 110)]
    const got = getFrameAtTime(frames, 0.15)
    expect(got).not.toBeNull()
    const x = got!.landmarks[0].x
    expect([0.1, 0.9]).toContain(x)
  })
})
