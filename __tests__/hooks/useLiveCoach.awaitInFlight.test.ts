import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import type { StreamedSwing } from '@/lib/liveSwingDetector'

// We don't need TTS to fire in this suite — stub the synth / queue so the
// hook can mount in jsdom without a real speechSynthesis surface.
vi.mock('@/lib/liveTts', () => {
  return {
    createBrowserSynth: () => null,
    LiveTtsQueue: class {
      isAvailable() { return false }
      enqueue() {}
      mute() {}
      unmute() {}
      reset() {}
      prime() {}
    },
  }
})

import { useLiveCoach } from '@/hooks/useLiveCoach'

function makeSwing(i: number): StreamedSwing {
  return {
    swingIndex: i,
    startFrameIndex: i * 10,
    endFrameIndex: i * 10 + 9,
    peakFrameIndex: i * 10 + 5,
    startMs: i * 1000,
    endMs: i * 1000 + 900,
    frames: [],
  }
}

// Push four swings — the default maxSwingsPerBatch — to force a batch fire.
function pushBatch(push: (s: StreamedSwing) => void) {
  for (let i = 0; i < 4; i++) push(makeSwing(i))
}

describe('useLiveCoach.awaitInFlight', () => {
  let originalFetch: typeof globalThis.fetch
  beforeEach(() => {
    originalFetch = globalThis.fetch
  })
  afterEach(() => {
    globalThis.fetch = originalFetch
    vi.restoreAllMocks()
  })

  it('resolves immediately when no request is in flight', async () => {
    // No fetch call expected — the hook hasn't fired a batch.
    const fetchSpy = vi.fn()
    globalThis.fetch = fetchSpy as unknown as typeof globalThis.fetch

    const { result } = renderHook(() => useLiveCoach())
    expect(result.current.isRequestInFlight()).toBe(false)

    const before = Date.now()
    await act(async () => {
      await result.current.awaitInFlight(5_000)
    })
    const elapsed = Date.now() - before
    // Fast-path: must not block on the timeout.
    expect(elapsed).toBeLessThan(200)
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('resolves when the in-flight batch settles', async () => {
    let resolveFetch: (value: Response) => void = () => {}
    const fetchPromise = new Promise<Response>((resolve) => {
      resolveFetch = resolve
    })
    globalThis.fetch = vi.fn(() => fetchPromise) as unknown as typeof globalThis.fetch

    const { result } = renderHook(() => useLiveCoach())

    // Fire a batch by pushing 4 swings — the hook fetches synchronously
    // inside fireBatch but the request hangs on our promise.
    await act(async () => {
      pushBatch(result.current.pushSwing)
    })
    expect(result.current.isRequestInFlight()).toBe(true)

    // Start awaiting before the request settles.
    let resolved = false
    const awaitPromise = act(async () => {
      await result.current.awaitInFlight(5_000)
      resolved = true
    })

    // Sanity: not resolved yet.
    await Promise.resolve()
    expect(resolved).toBe(false)

    // Now settle the in-flight request — awaitInFlight should resolve.
    await act(async () => {
      resolveFetch(new Response('coaching cue text', { status: 200 }))
    })
    await awaitPromise

    expect(resolved).toBe(true)
    expect(result.current.isRequestInFlight()).toBe(false)
  })

  it('resolves on timeout if a batch hangs longer than timeoutMs', async () => {
    // A fetch that never settles within the test window.
    globalThis.fetch = vi.fn(
      () => new Promise<Response>(() => { /* never resolves */ }),
    ) as unknown as typeof globalThis.fetch

    vi.useFakeTimers()
    try {
      const { result } = renderHook(() => useLiveCoach())

      await act(async () => {
        pushBatch(result.current.pushSwing)
      })
      expect(result.current.isRequestInFlight()).toBe(true)

      let resolved = false
      const awaitPromise = result.current.awaitInFlight(1_000).then(() => {
        resolved = true
      })

      // Advance past the timeout — must resolve via the cap, not the batch.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1_500)
      })
      await awaitPromise

      expect(resolved).toBe(true)
      // The batch is still in flight (it never settled) — awaitInFlight just
      // capped its wait. The flag should still read true.
      expect(result.current.isRequestInFlight()).toBe(true)
    } finally {
      vi.useRealTimers()
    }
  })

  it('handles a batch that errors out (still treated as settled)', async () => {
    const fetchSpy = vi.fn(() => Promise.reject(new Error('boom')))
    globalThis.fetch = fetchSpy as unknown as typeof globalThis.fetch

    const { result } = renderHook(() =>
      // Tiny minBatchIntervalMs so the retry path doesn't itself block.
      useLiveCoach({ minBatchIntervalMs: 1 }),
    )

    let awaitPromise: Promise<void> | null = null
    await act(async () => {
      pushBatch(result.current.pushSwing)
      // Begin awaiting after the batch has fired but before it settles.
      awaitPromise = result.current.awaitInFlight(5_000)
    })

    // The fetch path retries once after a 2s sleep before declaring failure;
    // settle the awaiter by waiting for the inFlight ref to flip.
    await act(async () => {
      await awaitPromise
    })

    expect(result.current.isRequestInFlight()).toBe(false)
  })
})
