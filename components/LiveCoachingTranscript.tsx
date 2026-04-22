'use client'

import { useEffect, useRef } from 'react'
import { useLiveStore } from '@/store/live'

function formatSessionMs(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000))
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

export default function LiveCoachingTranscript() {
  const transcript = useLiveStore((s) => s.transcript)
  const ttsEnabled = useLiveStore((s) => s.ttsEnabled)
  const setTtsEnabled = useLiveStore((s) => s.setTtsEnabled)
  const ttsAvailable = useLiveStore((s) => s.ttsAvailable)
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
          <p className="text-white/40 text-xs">
            {ttsAvailable ? 'Spoken through your earbuds, batched every few swings.' : 'Audio coaching is not available in this browser — transcript only.'}
          </p>
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
