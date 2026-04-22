'use client'

import { useCallback, useEffect, useMemo, useRef } from 'react'
import { buildAngleSummary } from '@/lib/jointAngles'
import { createBrowserSynth, LiveTtsQueue } from '@/lib/liveTts'
import type { StreamedSwing } from '@/lib/liveSwingDetector'
import { useLiveStore } from '@/store/live'

interface UseLiveCoachOptions {
  // Hard cap on swings per batch. Triggers a batch fire once reached.
  maxSwingsPerBatch?: number
  // After this many ms since the last batch, fire even if we only have a
  // partial batch (minimum minSwingsForTimeout).
  idleTimeoutMs?: number
  minSwingsForTimeout?: number
  // Minimum delay between batches, regardless of swing count.
  minBatchIntervalMs?: number
}

export interface UseLiveCoachReturn {
  // Push a newly detected swing into the coach's pending queue.
  pushSwing: (swing: StreamedSwing) => void
  // Give the coach the user-visible shot type so batches know what the player
  // is drilling.
  setShotType: (shotType: 'forehand' | 'backhand' | 'serve' | 'volley') => void
  // Hand it the wall-clock start time of the session so per-batch timestamps
  // and session_duration_ms are correct.
  markSessionStart: (ms: number) => void
  // User gesture tied to the Start button — unlocks speechSynthesis on iOS.
  primeTts: () => void
  // Clear pending state. Call on Stop or when switching sessions.
  reset: () => void
  // Convenience flag for the UI: true while a coach request is in flight.
  isRequestInFlight: () => boolean
}

type InternalSwing = {
  angleSummary: string
  startMs: number
  endMs: number
}

/**
 * Orchestrates the "every 3-5 swings, ask the coach a question" cadence.
 *
 * Swings come in via pushSwing() from useLiveCapture's onSwing callback.
 * We accumulate them into an internal queue and fire a batch when either:
 *   - the queue reaches maxSwingsPerBatch, OR
 *   - idleTimeoutMs has passed since the last batch and we have at least
 *     minSwingsForTimeout swings
 * Batches are serialized: a second batch is never in flight while the first
 * one is still running. New swings during a flight just queue up.
 */
export function useLiveCoach(options: UseLiveCoachOptions = {}): UseLiveCoachReturn {
  const {
    maxSwingsPerBatch = 4,
    idleTimeoutMs = 10_000,
    minSwingsForTimeout = 2,
    minBatchIntervalMs = 4_000,
  } = options

  const appendTranscriptEntry = useLiveStore((s) => s.appendTranscriptEntry)
  const setTtsAvailable = useLiveStore((s) => s.setTtsAvailable)
  const ttsEnabled = useLiveStore((s) => s.ttsEnabled)
  const setLastBatchAtMs = useLiveStore((s) => s.setLastBatchAtMs)

  const pendingRef = useRef<InternalSwing[]>([])
  const batchIndexRef = useRef(0)
  const inFlightRef = useRef(false)
  const lastBatchWallMsRef = useRef(0)
  const sessionStartMsRef = useRef(0)
  const shotTypeRef = useRef<'forehand' | 'backhand' | 'serve' | 'volley'>('forehand')
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const fireBatchRef = useRef<() => Promise<void>>(() => Promise.resolve())

  const ttsQueue = useMemo(() => new LiveTtsQueue(createBrowserSynth()), [])

  useEffect(() => {
    setTtsAvailable(ttsQueue.isAvailable())
  }, [setTtsAvailable, ttsQueue])

  useEffect(() => {
    if (ttsEnabled) ttsQueue.unmute()
    else ttsQueue.mute()
  }, [ttsEnabled, ttsQueue])

  const clearIdleTimer = useCallback(() => {
    if (idleTimerRef.current != null) {
      clearTimeout(idleTimerRef.current)
      idleTimerRef.current = null
    }
  }, [])

  const scheduleIdleTimer = useCallback(() => {
    if (idleTimerRef.current != null) {
      clearTimeout(idleTimerRef.current)
      idleTimerRef.current = null
    }
    idleTimerRef.current = setTimeout(() => {
      idleTimerRef.current = null
      if (pendingRef.current.length >= minSwingsForTimeout) {
        void fireBatchRef.current()
      }
    }, idleTimeoutMs)
  }, [idleTimeoutMs, minSwingsForTimeout])

  const fireBatch = useCallback(async () => {
    if (inFlightRef.current) return
    if (pendingRef.current.length === 0) return
    // Respect a minimum spacing so we don't fire two batches back-to-back
    // when swings arrive clustered.
    const now = Date.now()
    if (lastBatchWallMsRef.current > 0 && now - lastBatchWallMsRef.current < minBatchIntervalMs) {
      return
    }

    const swings = pendingRef.current.slice(0, maxSwingsPerBatch)
    pendingRef.current = pendingRef.current.slice(swings.length)
    clearIdleTimer()
    inFlightRef.current = true
    const batchIndex = batchIndexRef.current++
    const producedAt = Date.now()
    const sessionDurationMs = sessionStartMsRef.current
      ? producedAt - sessionStartMsRef.current
      : 0

    const tryPost = async (): Promise<{ text: string; eventId: string | null } | null> => {
      try {
        const res = await fetch('/api/live-coach', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            shotType: shotTypeRef.current,
            recentSwings: swings,
            batchIndex,
            sessionDurationMs,
          }),
        })
        if (!res.ok) return null
        const eventId = res.headers.get('X-Analysis-Event-Id')
        const text = (await res.text()).trim()
        if (!text) return null
        return { text, eventId }
      } catch {
        return null
      }
    }

    let result = await tryPost()
    if (!result) {
      // One retry with short backoff. After the second failure we drop the
      // batch rather than block the player with retries.
      await new Promise((r) => setTimeout(r, 2_000))
      result = await tryPost()
    }

    lastBatchWallMsRef.current = Date.now()
    setLastBatchAtMs(lastBatchWallMsRef.current)
    inFlightRef.current = false

    if (result) {
      appendTranscriptEntry({
        id: `batch-${batchIndex}-${producedAt}`,
        text: result.text,
        sessionMs: sessionDurationMs,
        eventId: result.eventId,
        swingCount: swings.length,
      })
      ttsQueue.enqueue(result.text, producedAt)
    } else {
      appendTranscriptEntry({
        id: `batch-${batchIndex}-skip-${producedAt}`,
        text: '(coaching skipped — network hiccup)',
        sessionMs: sessionDurationMs,
        eventId: null,
        swingCount: swings.length,
      })
    }

    // If more swings arrived while we were in flight, evaluate again.
    if (pendingRef.current.length >= maxSwingsPerBatch) {
      void fireBatchRef.current()
    } else if (pendingRef.current.length >= minSwingsForTimeout) {
      scheduleIdleTimer()
    }
  }, [appendTranscriptEntry, clearIdleTimer, maxSwingsPerBatch, minBatchIntervalMs, minSwingsForTimeout, scheduleIdleTimer, setLastBatchAtMs, ttsQueue])

  useEffect(() => {
    fireBatchRef.current = fireBatch
  }, [fireBatch])

  const pushSwing = useCallback(
    (swing: StreamedSwing) => {
      const angleSummary = buildAngleSummary(swing.frames)
      pendingRef.current.push({
        angleSummary,
        startMs: swing.startMs,
        endMs: swing.endMs,
      })
      if (pendingRef.current.length >= maxSwingsPerBatch) {
        void fireBatch()
      } else if (pendingRef.current.length >= minSwingsForTimeout) {
        scheduleIdleTimer()
      }
    },
    [fireBatch, maxSwingsPerBatch, minSwingsForTimeout, scheduleIdleTimer],
  )

  const setShotType = useCallback((shotType: 'forehand' | 'backhand' | 'serve' | 'volley') => {
    shotTypeRef.current = shotType
  }, [])

  const markSessionStart = useCallback((ms: number) => {
    sessionStartMsRef.current = ms
    batchIndexRef.current = 0
    lastBatchWallMsRef.current = 0
  }, [])

  const primeTts = useCallback(() => {
    ttsQueue.prime()
  }, [ttsQueue])

  const reset = useCallback(() => {
    pendingRef.current = []
    batchIndexRef.current = 0
    inFlightRef.current = false
    lastBatchWallMsRef.current = 0
    sessionStartMsRef.current = 0
    clearIdleTimer()
    ttsQueue.reset()
  }, [clearIdleTimer, ttsQueue])

  const isRequestInFlight = useCallback(() => inFlightRef.current, [])

  return { pushSwing, setShotType, markSessionStart, primeTts, reset, isRequestInFlight }
}
