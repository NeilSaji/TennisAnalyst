'use client'

import { useEffect, useRef } from 'react'
import { useLiveStore } from '@/store/live'

function formatSessionMs(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000))
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

interface LiveCoachingTranscriptProps {
  // True while a coach request is awaiting a response. Drives the header
  // "Listening to your last few swings…" pulse. When omitted, falls back
  // to the live store's coachRequestInFlight flag — that's the wiring
  // path used by the live page where LiveCapturePanel mirrors its
  // useLiveCoach instance into the store.
  isRequestInFlight?: boolean
}

export default function LiveCoachingTranscript({
  isRequestInFlight,
}: LiveCoachingTranscriptProps = {}) {
  const transcript = useLiveStore((s) => s.transcript)
  const ttsEnabled = useLiveStore((s) => s.ttsEnabled)
  const setTtsEnabled = useLiveStore((s) => s.setTtsEnabled)
  const ttsAvailable = useLiveStore((s) => s.ttsAvailable)
  // Inline error banner. Set by useLiveCoach when /api/live-coach actually
  // failed (network / 5xx). Cleared on the next successful batch and on
  // deliberate silence — so a clean drill that gets silence-as-cue won't
  // hold the banner up indefinitely.
  const coachingError = useLiveStore((s) => s.coachingError)
  const coachRequestInFlightStore = useLiveStore((s) => s.coachRequestInFlight)
  const inFlight = isRequestInFlight ?? coachRequestInFlightStore
  const listRef = useRef<HTMLUListElement | null>(null)

  useEffect(() => {
    const el = listRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [transcript])

  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden">
      <header className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        <div>
          <h2 className="text-white font-medium text-sm">Live coaching</h2>
          {inFlight ? (
            <p
              data-testid="coach-thinking-indicator"
              role="status"
              aria-live="polite"
              className="text-emerald-300/80 text-xs flex items-center gap-1.5 animate-pulse"
            >
              <span
                aria-hidden
                className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400"
              />
              Listening to your last few swings…
            </p>
          ) : (
            <p className="text-white/40 text-xs">
              {ttsAvailable ? 'Spoken through your earbuds, batched every few swings.' : 'Audio coaching is not available in this browser — transcript only.'}
            </p>
          )}
        </div>
        {ttsAvailable ? (
          <button
            onClick={() => setTtsEnabled(!ttsEnabled)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
              ttsEnabled
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                : 'bg-white/5 text-white/60 border border-white/10'
            }`}
            aria-pressed={ttsEnabled}
          >
            {ttsEnabled ? 'Audio on' : 'Audio muted'}
          </button>
        ) : null}
      </header>

      {coachingError ? (
        <div
          role="status"
          aria-live="polite"
          className="px-4 py-2 text-xs text-amber-200 bg-amber-500/10 border-b border-amber-500/20 flex items-center gap-2"
        >
          <span aria-hidden="true" className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400" />
          <span>{coachingError}</span>
        </div>
      ) : null}

      <ul ref={listRef} className="max-h-72 overflow-y-auto divide-y divide-white/5">
        {transcript.length === 0 ? (
          <li className="px-4 py-6 text-white/40 text-sm text-center">
            Coaching will appear here after a few swings.
          </li>
        ) : (
          transcript.map((entry) => (
            <li key={entry.id} className="px-4 py-3 space-y-1">
              <div className="flex items-center gap-2 text-white/40 text-xs">
                <span>{formatSessionMs(entry.sessionMs)}</span>
                <span>·</span>
                <span>{entry.swingCount} {entry.swingCount === 1 ? 'swing' : 'swings'}</span>
              </div>
              <p className="text-white text-sm leading-relaxed">{entry.text}</p>
            </li>
          ))
        )}
      </ul>
    </section>
  )
}
