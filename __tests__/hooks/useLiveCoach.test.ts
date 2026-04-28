/**
 * Phase 3 — Visibility worker tests for useLiveCoach.isRequestInFlight().
 *
 * Worker 3 owns the batching logic; Worker 5 owns awaitInFlight. This
 * file covers ONLY the `isRequestInFlight()` getter, which Worker 4
 * adds to drive the transcript-header pulse.
 *
 * Contract guarded:
 *   1. Returns false at idle (no pending swings, no batch fired).
 *   2. Returns true the moment a batch fires (i.e. while the fetch is
 *      pending).
 *   3. Returns false the moment the batch settles, regardless of
 *      whether it succeeded or failed.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'

// Avoid pulling in a real speech-synthesis layer.
vi.mock('@/lib/liveTts', () => ({
  createBrowserSynth: () => null,
  LiveTtsQueue: class {
    isAvailable() {
      return false
    }
    enqueue() {}
    mute() {}
    unmute() {}
    prime() {}
    reset() {}
  },
}))

// jointAngles.buildAngleSummary is pure — keep it real but stub the
// payload so the test isn't dependent on its formatting.
vi.mock('@/lib/jointAngles', async (orig) => {
  const actual = (await orig()) as Record<string, unknown>
  return {
    ...actual,
    buildAngleSummary: () => 'fake-angle-summary',
  }
})

import { useLiveCoach } from '@/hooks/useLiveCoach'
import type { StreamedSwing } from '@/lib/liveSwingDetector'

function fakeSwing(index: number): StreamedSwing {
  return {
    swingIndex: index,
    startFrameIndex: index * 10,
    endFrameIndex: index * 10 + 5,
    peakFrameIndex: index * 10 + 3,
    startMs: index * 1000,
    endMs: index * 1000 + 500,
    frames: [],
  }
}

// Drives the fetch mock with control over when the response resolves
// or rejects. Lets us assert isRequestInFlight() at three distinct
// points: before the batch fires, during the in-flight window, and
// after settle.
function controllableFetch() {
  const calls: { resolve: (v: Response) => void; reject: (e: unknown) => void }[] = []
  const fn = vi.fn(() => {
    return new Promise<Response>((resolve, reject) => {
      calls.push({ resolve, reject })
    })
  })
  return { fn, calls }
}

describe('useLiveCoach.isRequestInFlight()', () => {
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    globalThis.fetch = originalFetch
  })

  it('returns false when no batch has fired and no swings are pending', () => {
    const { result } = renderHook(() => useLiveCoach())
    expect(result.current.isRequestInFlight()).toBe(false)
  })

  it('flips to true the moment a batch fires, and back to false when the response settles', async () => {
    const { fn, calls } = controllableFetch()
    globalThis.fetch = fn as unknown as typeof globalThis.fetch

    const { result } = renderHook(() =>
      useLiveCoach({ maxSwingsPerBatch: 2, minBatchIntervalMs: 0 }),
    )

    // Idle.
    expect(result.current.isRequestInFlight()).toBe(false)

    // Push two swings — that's the batch threshold, so a batch fires
    // immediately.
    act(() => {
      result.current.pushSwing(fakeSwing(0))
      result.current.pushSwing(fakeSwing(1))
    })

    // Allow the microtask queue (the fireBatch IIFE pre-await checks)
    // to run so the fetch is actually issued.
    await act(async () => {
      await Promise.resolve()
    })

    // The fetch promise is held open by our controllable mock. While
    // it's pending, isRequestInFlight() must be true.
    expect(fn).toHaveBeenCalledTimes(1)
    expect(result.current.isRequestInFlight()).toBe(true)

    // Now settle the response. Use a successful Response so the
    // success-path code (transcript append + TTS enqueue) runs.
    await act(async () => {
      calls[0].resolve(
        new Response('Stay loose.', {
          status: 200,
          headers: { 'X-Analysis-Event-Id': 'evt-1' },
        }),
      )
      // Drain microtasks so the await chain completes.
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(result.current.isRequestInFlight()).toBe(false)
  })

  it('returns false after the batch errors out (network failure path)', async () => {
    // The hook retries once with a 2s backoff after a fetch failure
    // before giving up. We need to drive the timer through that.
    const fn = vi.fn(() => Promise.reject(new Error('network down')))
    globalThis.fetch = fn as unknown as typeof globalThis.fetch

    const { result } = renderHook(() =>
      useLiveCoach({ maxSwingsPerBatch: 2, minBatchIntervalMs: 0 }),
    )

    act(() => {
      result.current.pushSwing(fakeSwing(0))
      result.current.pushSwing(fakeSwing(1))
    })

    // Let the first fetch reject, then advance through the 2s backoff
    // so the retry runs and also rejects.
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
      vi.advanceTimersByTime(2_000)
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    // After both attempts have rejected, the hook must clear the
    // in-flight flag so the UI doesn't stay stuck on a stale pulse.
    expect(fn).toHaveBeenCalledTimes(2)
    expect(result.current.isRequestInFlight()).toBe(false)
  })
})
