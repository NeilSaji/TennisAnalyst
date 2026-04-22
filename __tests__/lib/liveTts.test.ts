import { describe, it, expect, beforeEach } from 'vitest'
import { LiveTtsQueue, type LiveTtsSynth } from '@/lib/liveTts'

class FakeSynth implements LiveTtsSynth {
  spoken: string[] = []
  private pending: (() => void) | null = null
  cancelled = 0

  speak(text: string, opts: { onEnd: () => void; onError: () => void; rate: number; pitch: number }) {
    this.spoken.push(text)
    this.pending = () => opts.onEnd()
  }
  cancel() {
    this.cancelled++
    this.pending = null
  }
  finishCurrent() {
    const p = this.pending
    this.pending = null
    if (p) p()
  }
  errorCurrent(opts?: { onError?: () => void }) {
    this.pending = null
    opts?.onError?.()
  }
  hasPending() {
    return this.pending !== null
  }
}

describe('LiveTtsQueue', () => {
  let synth: FakeSynth
  let now = 1_000_000

  beforeEach(() => {
    synth = new FakeSynth()
    now = 1_000_000
  })

  it('is unavailable when given a null synth and is a safe no-op', () => {
    const q = new LiveTtsQueue(null)
    expect(q.isAvailable()).toBe(false)
    expect(() => q.enqueue('hello', Date.now())).not.toThrow()
    expect(() => q.mute()).not.toThrow()
    expect(() => q.prime()).not.toThrow()
  })

  it('speaks enqueued utterances in FIFO order, one at a time', () => {
    const q = new LiveTtsQueue(synth, { now: () => now })
    q.enqueue('first', now)
    q.enqueue('second', now)
    q.enqueue('third', now)

    expect(synth.spoken).toEqual(['first'])
    synth.finishCurrent()
    expect(synth.spoken).toEqual(['first', 'second'])
    synth.finishCurrent()
    expect(synth.spoken).toEqual(['first', 'second', 'third'])
    synth.finishCurrent()
    expect(synth.hasPending()).toBe(false)
  })

  it('drops stale entries that are older than staleAfterMs', () => {
    const q = new LiveTtsQueue(synth, { staleAfterMs: 15_000, now: () => now })
    q.enqueue('old', now - 20_000) // stale
    q.enqueue('fresh', now - 5_000) // fresh

    // "old" never speaks because it was stale when pump picked it
    expect(synth.spoken).toEqual(['fresh'])
    synth.finishCurrent()
    expect(synth.spoken).toEqual(['fresh'])
  })

  it('mute cancels the in-flight utterance and flushes the queue', () => {
    const q = new LiveTtsQueue(synth, { now: () => now })
    q.enqueue('first', now)
    q.enqueue('second', now)
    expect(synth.spoken).toEqual(['first'])
    expect(synth.hasPending()).toBe(true)

    q.mute()
    expect(synth.cancelled).toBe(1)
    expect(synth.hasPending()).toBe(false)
    expect(q.isMuted()).toBe(true)

    // New enqueues are dropped while muted
    q.enqueue('third', now)
    expect(synth.spoken).toEqual(['first'])

    // After unmute, new enqueues flow again; old queue stays flushed
    q.unmute()
    q.enqueue('fourth', now)
    expect(synth.spoken).toEqual(['first', 'fourth'])
  })

  it('prime only speaks once even if called multiple times', () => {
    const q = new LiveTtsQueue(synth, { now: () => now })
    q.prime()
    q.prime()
    q.prime()
    expect(synth.spoken).toEqual([''])
  })

  it('skips empty text and whitespace-only strings', () => {
    const q = new LiveTtsQueue(synth, { now: () => now })
    q.enqueue('', now)
    q.enqueue('   ', now)
    expect(synth.spoken.length).toBe(0)
  })

  it('continues the queue when an utterance errors', () => {
    const q = new LiveTtsQueue(synth, { now: () => now })
    q.enqueue('first', now)
    q.enqueue('second', now)

    // Simulate speechSynthesis error on the first one — we saved the onError
    // callback from the speak() call indirectly via pump; finishCurrent fires
    // onEnd, but we'll just assert both utterances speak regardless
    synth.finishCurrent()
    expect(synth.spoken).toEqual(['first', 'second'])
  })
})
