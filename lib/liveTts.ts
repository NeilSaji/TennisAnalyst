// Browser TTS queue for live coaching. Isolates speechSynthesis quirks (iOS
// requires a user-gesture prime before the first audible utterance; voices
// load async via `voiceschanged`; stale queue entries would arrive after the
// player has moved on to the next coaching moment).
//
// The queue drops utterances produced more than `staleAfterMs` ago so we
// never coach a player about a batch from 30 seconds earlier.
//
// Synth access goes through a small dependency-injectable interface so the
// queue itself is testable under vitest/jsdom without a real SpeechSynthesis.

export interface LiveTtsSynth {
  speak(
    text: string,
    options: {
      rate: number
      pitch: number
      onEnd: () => void
      onError: () => void
    },
  ): void
  cancel(): void
}

export interface LiveTtsOptions {
  staleAfterMs?: number
  rate?: number
  pitch?: number
  now?: () => number
}

type QueueEntry = { text: string; producedAt: number }

export function createBrowserSynth(): LiveTtsSynth | null {
  if (typeof window === 'undefined') return null
  if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) {
    return null
  }
  return {
    speak(text, { rate, pitch, onEnd, onError }) {
      const u = new window.SpeechSynthesisUtterance(text)
      u.rate = rate
      u.pitch = pitch
      u.onend = () => onEnd()
      u.onerror = () => onError()
      window.speechSynthesis.speak(u)
    },
    cancel() {
      window.speechSynthesis.cancel()
    },
  }
}

export class LiveTtsQueue {
  private readonly staleAfterMs: number
  private readonly rate: number
  private readonly pitch: number
  private readonly now: () => number
  private queue: QueueEntry[] = []
  private speaking = false
  private muted = false
  private primed = false

  constructor(
    private readonly synth: LiveTtsSynth | null,
    opts: LiveTtsOptions = {},
  ) {
    this.staleAfterMs = opts.staleAfterMs ?? 15_000
    this.rate = opts.rate ?? 1.05
    this.pitch = opts.pitch ?? 1.0
    this.now = opts.now ?? (() => Date.now())
  }

  isAvailable(): boolean {
    return this.synth !== null
  }

  isMuted(): boolean {
    return this.muted
  }

  // Must be called from a user-gesture handler (e.g. Start button onClick) so
  // iOS Safari unlocks speechSynthesis. An empty utterance is enough — it
  // silently flushes the "no gesture yet" gate without the user hearing anything.
  prime(): void {
    if (!this.synth || this.primed) return
    this.synth.speak('', {
      rate: this.rate,
      pitch: this.pitch,
      onEnd: () => {},
      onError: () => {},
    })
    this.primed = true
  }

  enqueue(text: string, producedAt: number): void {
    if (!text || !text.trim()) return
    if (!this.synth || this.muted) return
    this.queue.push({ text: text.trim(), producedAt })
    this.pump()
  }

  mute(): void {
    this.muted = true
    this.queue = []
    this.speaking = false
    if (this.synth) this.synth.cancel()
  }

  unmute(): void {
    this.muted = false
  }

  reset(): void {
    this.queue = []
    this.speaking = false
    if (this.synth) this.synth.cancel()
  }

  private pump(): void {
    if (this.speaking || this.muted || !this.synth) return

    let entry: QueueEntry | undefined
    const threshold = this.now() - this.staleAfterMs
    while ((entry = this.queue.shift())) {
      if (entry.producedAt >= threshold) break
      entry = undefined
    }
    if (!entry) return

    this.speaking = true
    this.synth.speak(entry.text, {
      rate: this.rate,
      pitch: this.pitch,
      onEnd: () => {
        this.speaking = false
        this.pump()
      },
      onError: () => {
        this.speaking = false
        this.pump()
      },
    })
  }
}
