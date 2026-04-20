'use client'

import { useState, useRef, useEffect } from 'react'
import { useAnalysisStore, usePoseStore } from '@/store'
import type { ProSwing, PoseFrame } from '@/lib/supabase'

interface LLMCoachingPanelProps {
  proSwing: ProSwing | null
  compareMode?: 'pro' | 'custom' | 'baseline'
  frames?: PoseFrame[]
  // Second-take frames. When present in 'custom' or 'baseline' mode,
  // analyze runs as a user-vs-user comparison (consistency / best-day drift).
  compareFrames?: PoseFrame[]
  // Shown when compareMode === 'baseline'. Wired to the prompt so coaching
  // references the specific baseline by name.
  baselineLabel?: string
}

export default function LLMCoachingPanel({ proSwing, compareMode = 'pro', frames, compareFrames, baselineLabel }: LLMCoachingPanelProps) {
  const [open, setOpen] = useState(false)
  const { feedback, loading, setFeedback, appendFeedback, setLoading, reset } =
    useAnalysisStore()
  const { framesData, sessionId } = usePoseStore()

  const [chatMessages, setChatMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [userFocus, setUserFocus] = useState('')
  const chatEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (chatMessages.length > 0) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [chatMessages, chatLoading])

  const canAnalyze =
    framesData.length > 0 &&
    (compareMode === 'custom' || compareMode === 'baseline' || proSwing !== null)

  const sendChatMessage = async () => {
    const text = chatInput.trim()
    if (!text || chatLoading) return
    setChatInput('')

    const userMsg = { role: 'user' as const, content: text }
    const allMessages = [...chatMessages, userMsg]
    setChatMessages(allMessages)
    setChatLoading(true)

    try {
      const res = await fetch('/api/pro-coach', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          proSwingId: proSwing?.id,
          messages: allMessages.map((m) => ({ role: m.role, content: m.content })),
          context: feedback ? `Previous analysis:\n${feedback}` : undefined,
        }),
      })

      if (!res.ok || !res.body) throw new Error('Chat failed')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let assistantContent = ''

      setChatMessages((prev) => [...prev, { role: 'assistant', content: '' }])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        assistantContent += decoder.decode(value, { stream: true })
        const content = assistantContent
        setChatMessages((prev) => [
          ...prev.slice(0, -1),
          { role: 'assistant', content },
        ])
      }
    } catch {
      setChatMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, something went wrong. Please try again.' },
      ])
    } finally {
      setChatLoading(false)
    }
  }

  const runAnalysis = async () => {
    if (!canAnalyze || loading) return
    reset()
    setOpen(true)
    setLoading(true)

    const effectiveFrames = frames ?? framesData
    const keypointsJson = {
      fps_sampled: 30,
      frame_count: effectiveFrames.length,
      frames: effectiveFrames,
    }

    // Pass the second take only in user-vs-user modes (custom or baseline).
    // In pro mode the pro reference wins even if compareFrames is set.
    const compareKeypointsJson =
      (compareMode === 'custom' || compareMode === 'baseline') &&
      !proSwing &&
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
          ...(proSwing ? { proSwingId: proSwing.id } : {}),
          keypointsJson,
          ...(compareKeypointsJson ? { compareKeypointsJson } : {}),
          ...(trimmedFocus ? { userFocus: trimmedFocus } : {}),
          ...(compareMode === 'baseline' ? { compareMode: 'baseline', baselineLabel } : {}),
        }),
      })

      if (!res.ok || !res.body) throw new Error('Analysis failed')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        appendFeedback(decoder.decode(value, { stream: true }))
      }
      // Flush any remaining bytes held by the streaming decoder (UTF-8 boundary safety)
      const remaining = decoder.decode()
      if (remaining) appendFeedback(remaining)

      // Check if the stream ended with an error sentinel from the server
      const currentFeedback = useAnalysisStore.getState().feedback
      if (currentFeedback.includes('\n\n[ERROR] ')) {
        const cleaned = currentFeedback.split('\n\n[ERROR] ')[0]
        setFeedback(cleaned || 'Analysis failed. Please try again.')
      }
    } catch (err) {
      setFeedback('Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-xl border border-white/10 overflow-hidden">
      {/* Header / trigger */}
      <div className="bg-white/5 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">🤖</span>
          <h3 className="text-sm font-semibold text-white">AI Coach</h3>
          {proSwing ? (
            <span className="text-xs text-white/40">
              vs {proSwing.pros?.name ?? 'Pro'} · {proSwing.shot_type}
            </span>
          ) : compareMode === 'baseline' ? (
            <span className="text-xs text-white/40">
              vs {baselineLabel ?? 'your best day'}
            </span>
          ) : compareMode === 'custom' ? (
            <span className="text-xs text-white/40">
              Form Analysis
            </span>
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

      {/* Hint when disabled due to no pro selected */}
      {!canAnalyze && !loading && !feedback && framesData.length > 0 && compareMode !== 'custom' && compareMode !== 'baseline' && !proSwing && (
        <div className="px-4 py-2 bg-white/[0.02]">
          <p className="text-white/40 text-xs">Select a pro player above to compare against</p>
        </div>
      )}

      {/* Focus input — optional user prompt before Analyze. Hidden once analysis
          has run or is running; user can always re-open with Re-analyze. */}
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
              proSwing
                ? `e.g. "working on topspin" or "why does my follow-through look different from ${proSwing.pros?.name ?? 'the pro'}?"`
                : compareMode === 'baseline' && compareFrames?.length
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

      {/* Content */}
      {open && (
        <div className="bg-black/20 p-4 max-h-[500px] overflow-y-auto">
          {!canAnalyze && !loading && !feedback && (
            <p className="text-white/40 text-sm text-center py-4">
              {framesData.length === 0
                ? 'Upload a video first to get coaching feedback.'
                : compareMode === 'custom' || compareMode === 'baseline'
                  ? 'Click Analyze Swing for AI coaching on your form.'
                  : 'Select a pro player to compare against.'}
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

          {/* Chat messages */}
          {chatMessages.length > 0 && (
            <div className="mt-4 border-t border-white/10 pt-4 space-y-3">
              {chatMessages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] rounded-xl px-3 py-2 text-sm ${
                    msg.role === 'user'
                      ? 'bg-emerald-500/20 text-emerald-100'
                      : 'bg-white/5 text-white/80'
                  }`}>
                    <MarkdownText text={msg.content} />
                    {chatLoading && i === chatMessages.length - 1 && msg.role === 'assistant' && (
                      <span className="inline-block w-1.5 h-4 bg-emerald-400 animate-pulse ml-0.5 rounded-sm" />
                    )}
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>
      )}

      {/* Chat input — visible after analysis is done */}
      {open && feedback && !loading && (
        <div className="border-t border-white/10 p-3 flex gap-2">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendChatMessage()}
            placeholder="Ask a follow-up question..."
            className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-emerald-500/50"
            disabled={chatLoading}
          />
          <button
            onClick={sendChatMessage}
            disabled={!chatInput.trim() || chatLoading}
            className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 disabled:bg-white/10 disabled:text-white/30 text-white text-sm font-medium rounded-lg transition-colors"
          >
            Send
          </button>
        </div>
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
        // Inline bold - split handles all cases including mixed bold/normal
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
