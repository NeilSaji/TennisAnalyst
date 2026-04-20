'use client'

import { useEffect, useState } from 'react'
import { useAnalysisStore, usePoseStore } from '@/store'
import { useUser } from '@/hooks/useUser'
import type { PoseFrame } from '@/lib/supabase'

interface LLMCoachingPanelProps {
  compareMode?: 'solo' | 'custom' | 'baseline'
  frames?: PoseFrame[]
  // Second-take frames. When present in 'custom' or 'baseline' mode,
  // analyze runs as a user-vs-user comparison (consistency / best-day drift).
  compareFrames?: PoseFrame[]
  // Shown when compareMode === 'baseline'. Wired to the prompt so coaching
  // references the specific baseline by name.
  baselineLabel?: string
}

type Correction = 'correct' | 'too_easy' | 'too_hard'
type FeedbackState = 'idle' | 'sending' | 'sent' | 'error'

export default function LLMCoachingPanel({ compareMode = 'solo', frames, compareFrames, baselineLabel }: LLMCoachingPanelProps) {
  const [open, setOpen] = useState(false)
  const { feedback, loading, setFeedback, appendFeedback, setLoading, reset } =
    useAnalysisStore()
  const { framesData, sessionId } = usePoseStore()
  const { user } = useUser()

  const [userFocus, setUserFocus] = useState('')
  const [analysisEventId, setAnalysisEventId] = useState<string | null>(null)
  const [feedbackState, setFeedbackState] = useState<FeedbackState>('idle')

  const canAnalyze = framesData.length > 0

  // If the feedback text clears (e.g. reset/re-analyze), drop any lingering
  // rating state so the strip doesn't carry across runs.
  useEffect(() => {
    if (!feedback) {
      setFeedbackState('idle')
      setAnalysisEventId(null)
    }
  }, [feedback])

  const runAnalysis = async () => {
    if (!canAnalyze || loading) return
    reset()
    setOpen(true)
    setLoading(true)
    setAnalysisEventId(null)
    setFeedbackState('idle')

    const effectiveFrames = frames ?? framesData
    const keypointsJson = {
      fps_sampled: 30,
      frame_count: effectiveFrames.length,
      frames: effectiveFrames,
    }

    // Pass the second take only in user-vs-user modes (custom or baseline).
    const compareKeypointsJson =
      (compareMode === 'custom' || compareMode === 'baseline') &&
      compareFrames &&
      compareFrames.length > 0
        ? {
            fps_sampled: 30,
            frame_count: compareFrames.length,
            frames: compareFrames,
          }
        : undefined

    const trimmedFocus = userFocus.trim()

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          keypointsJson,
          ...(compareKeypointsJson ? { compareKeypointsJson } : {}),
          ...(trimmedFocus ? { userFocus: trimmedFocus } : {}),
          ...(compareMode === 'baseline' ? { compareMode: 'baseline', baselineLabel } : {}),
        }),
      })

      if (!res.ok || !res.body) throw new Error('Analysis failed')

      // Backend emits this header at the start of the stream so we can
      // round-trip a thumbs rating back to the same analysis_events row.
      const eventId = res.headers.get('X-Analysis-Event-Id')
      if (eventId) setAnalysisEventId(eventId)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        appendFeedback(decoder.decode(value, { stream: true }))
      }
      const remaining = decoder.decode()
      if (remaining) appendFeedback(remaining)

      const currentFeedback = useAnalysisStore.getState().feedback
      if (currentFeedback.includes('\n\n[ERROR] ')) {
        const cleaned = currentFeedback.split('\n\n[ERROR] ')[0]
        setFeedback(cleaned || 'Analysis failed. Please try again.')
      }
    } catch {
      setFeedback('Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const submitFeedback = async (correction: Correction) => {
    if (!analysisEventId || feedbackState === 'sending' || feedbackState === 'sent') return
    setFeedbackState('sending')
    try {
      const res = await fetch(`/api/analysis-events/${analysisEventId}/feedback`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ correction }),
      })
      if (!res.ok) throw new Error('Feedback failed')
      setFeedbackState('sent')
    } catch {
      setFeedbackState('error')
    }
  }

  const showFeedbackStrip = Boolean(feedback) && !loading && Boolean(user) && Boolean(analysisEventId)

  return (
    <div className="rounded-xl border border-white/10 overflow-hidden">
      <div className="bg-white/5 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">🤖</span>
          <h3 className="text-sm font-semibold text-white">AI Coach</h3>
          {compareMode === 'baseline' ? (
            <span className="text-xs text-white/40">
              vs {baselineLabel ?? 'your best day'}
            </span>
          ) : compareMode === 'custom' ? (
            <span className="text-xs text-white/40">Form Analysis</span>
          ) : null}
        </div>

        <div className="flex items-center gap-2">
          {feedback && (
            <button
              onClick={() => setOpen((o) => !o)}
              className="text-xs text-white/50 hover:text-white"
            >
              {open ? 'Collapse' : 'Expand'}
            </button>
          )}
          <button
            onClick={runAnalysis}
            disabled={!canAnalyze || loading}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              canAnalyze && !loading
                ? 'bg-emerald-500 hover:bg-emerald-400 text-white'
                : 'bg-white/10 text-white/30 cursor-not-allowed'
            }`}
          >
            {loading ? (
              <span className="flex items-center gap-1">
                <span className="animate-spin">⚙️</span> Analyzing...
              </span>
            ) : feedback ? (
              'Re-analyze'
            ) : (
              'Analyze Swing'
            )}
          </button>
        </div>
      </div>

      {canAnalyze && !loading && !feedback && (
        <div className="px-4 py-3 bg-white/[0.02] border-t border-white/5">
          <label className="block text-xs text-white/50 mb-1.5">
            Anything specific you want feedback on?{' '}
            <span className="text-white/30">(optional)</span>
          </label>
          <textarea
            value={userFocus}
            onChange={(e) => setUserFocus(e.target.value)}
            placeholder={
              compareMode === 'baseline' && compareFrames?.length
                ? `e.g. "did my hip rotation hold up?" or "focus on follow-through"`
                : compareMode === 'custom' && compareFrames?.length
                  ? `e.g. "why is my second shot flatter?" or "focus on hip rotation"`
                  : `e.g. "working on topspin" or "am I rotating my hips enough?"`
            }
            rows={2}
            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder:text-white/25 focus:outline-none focus:border-emerald-500/40 resize-none"
          />
        </div>
      )}

      {open && (
        <div className="bg-black/20 p-4 max-h-[500px] overflow-y-auto">
          {!canAnalyze && !loading && !feedback && (
            <p className="text-white/40 text-sm text-center py-4">
              Upload a video first to get coaching feedback.
            </p>
          )}
          {(loading || feedback) && (
            <div className="prose prose-invert prose-sm max-w-none">
              <MarkdownText text={feedback} />
              {loading && (
                <span className="inline-block w-1.5 h-4 bg-emerald-400 animate-pulse ml-0.5 rounded-sm" />
              )}
            </div>
          )}
        </div>
      )}

      {showFeedbackStrip && (
        <FeedbackStrip state={feedbackState} onSubmit={submitFeedback} />
      )}
    </div>
  )
}

function FeedbackStrip({
  state,
  onSubmit,
}: {
  state: FeedbackState
  onSubmit: (correction: Correction) => void
}) {
  if (state === 'sent') {
    return (
      <div className="bg-white/[0.02] border-t border-white/5 px-4 py-3 text-xs text-white/50">
        Thanks — logged.
      </div>
    )
  }

  const disabled = state === 'sending'

  return (
    <div className="bg-white/[0.02] border-t border-white/5 px-4 py-3 flex items-center gap-3 flex-wrap">
      <span className="text-xs text-white/50">Was this coaching right for you?</span>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onSubmit('correct')}
          disabled={disabled}
          className="px-2.5 py-1 rounded-lg text-xs font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/25 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          👍 Spot on
        </button>
        <button
          type="button"
          onClick={() => onSubmit('too_hard')}
          disabled={disabled}
          className="px-2.5 py-1 rounded-lg text-xs font-medium bg-white/10 text-white/70 border border-white/10 hover:bg-white/15 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ⬇️ Too advanced
        </button>
        <button
          type="button"
          onClick={() => onSubmit('too_easy')}
          disabled={disabled}
          className="px-2.5 py-1 rounded-lg text-xs font-medium bg-white/10 text-white/70 border border-white/10 hover:bg-white/15 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ⬆️ Too simple
        </button>
      </div>
      {state === 'error' && (
        <span className="text-xs text-rose-300">Couldn&apos;t save — try again?</span>
      )}
    </div>
  )
}

// Minimal markdown renderer for bold + headers
function MarkdownText({ text }: { text: string }) {
  const lines = text.split('\n')
  return (
    <div className="space-y-1 text-white/80 text-sm leading-relaxed">
      {lines.map((line, i) => {
        if (line.startsWith('## ')) {
          return (
            <h3 key={i} className="text-white font-bold text-base mt-3 mb-1">
              {line.replace('## ', '')}
            </h3>
          )
        }
        const parts = line.split(/(\*\*[^*]+\*\*)/)
        return (
          <p key={i} className={line.startsWith('-') ? 'pl-3' : ''}>
            {parts.map((part, j) =>
              part.startsWith('**') && part.endsWith('**') ? (
                <strong key={j} className="text-white">
                  {part.replace(/\*\*/g, '')}
                </strong>
              ) : (
                part
              )
            )}
          </p>
        )
      })}
    </div>
  )
}
