'use client'

import { useCallback, useEffect, useMemo, useRef } from 'react'
import { buildAngleSummary } from '@/lib/jointAngles'
import { createBrowserSynth, LiveTtsQueue } from '@/lib/liveTts'
import type { StreamedSwing } from '@/lib/liveSwingDetector'
import { useLiveStore } from '@/store/live'

interface UseLiveCoachOptions {
  // Hard cap on swings per batch. Triggers a batch fire once reached, but
  // only after the post-swing grace period has elapsed (see postSwingGraceMs).
  maxSwingsPerBatch?: number
  // After this many ms since the LAST SWING (not the last batch), fire even
  // if we only have a partial batch (minimum minSwingsForTimeout). Naming
  // this off the last swing rather than the last batch means we don't fire
  // mid-rally just because the model is overdue for a turn.
  idleTimeoutMs?: number
  minSwingsForTimeout?: number
  // Minimum delay between batches, regardless of swing count.
  minBatchIntervalMs?: number
  // Hold even a full-size batch until at least this many ms have elapsed
  // since the most recent swing. Stops cues from firing into the player's
  // ear during a hot rally where 4 swings can land in 3 seconds.
  postSwingGraceMs?: number
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
  // Wait until any outstanding coach request settles, or the timeout elapses
  // (whichever comes first). Used by the Stop flow so analysis_events rows
  // are written before we backfill session_id.
  awaitInFlight: (timeoutMs: number) => Promise<void>
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
 *   - the queue reaches maxSwingsPerBatch AND the post-swing grace period
 *     has elapsed (no batch fires while the player is mid-rally), OR
 *   - idleTimeoutMs has passed since the LAST SWING (not the last batch)
 *     and we have at least minSwingsForTimeout swings.
 * Batches are serialized: a second batch is never in flight while the first
 * one is still running. New swings during a flight just queue up.
 *
 * Each batch ships the last 3 transcript cues as `recentCues` so the model
 * can pivot, acknowledge persistence, or stay silent rather than repeating
 * itself. A deliberate silence (X-Live-Coach-Silence header or empty body)
 * is treated as success — the error banner stays clear. Real failures (5xx
 * or network exceptions, after one retry) surface as a coachingError on the
 * live store, which the transcript header renders as an inline banner.
 */
export function useLiveCoach(options: UseLiveCoachOptions = {}): UseLiveCoachReturn {
  const {
    maxSwingsPerBatch = 4,
    idleTimeoutMs = 10_000,
    minSwingsForTimeout = 2,
    minBatchIntervalMs = 4_000,
    postSwingGraceMs = 3_000,
  } = options

  const appendTranscriptEntry = useLiveStore((s) => s.appendTranscriptEntry)
  const setTtsAvailable = useLiveStore((s) => s.setTtsAvailable)
  const ttsEnabled = useLiveStore((s) => s.ttsEnabled)
  const setLastBatchAtMs = useLiveStore((s) => s.setLastBatchAtMs)
  const setCoachingError = useLiveStore((s) => s.setCoachingError)
  // Read transcript via getState() inside fire-time so the coach sees the
  // most recent cues without re-creating the fireBatch callback on every
  // transcript update (which would churn idle timers).
  const liveStoreRef = useRef(useLiveStore)
  liveStoreRef.current = useLiveStore

  const pendingRef = useRef<InternalSwing[]>([])
  const batchIndexRef = useRef(0)
  const inFlightRef = useRef(false)
  // Resolvers for pending awaitInFlight() callers. Flushed when inFlightRef
  // flips back to false (batch settled) or when reset() is called.
  const inFlightWaitersRef = useRef<Array<() => void>>([])
  const lastBatchWallMsRef = useRef(0)
  // Wall-clock timestamp (Date.now()) of the most recent swing pushed via
  // pushSwing(). Drives both the idle timer (idleTimeoutMs since last swing)
  // and the post-swing grace period (don't fire mid-rally).
  const lastSwingWallMsRef = useRef(0)
  const sessionStartMsRef = useRef(0)
  const shotTypeRef = useRef<'forehand' | 'backhand' | 'serve' | 'volley'>('forehand')
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const graceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
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

  const clearGraceTimer = useCallback(() => {
    if (graceTimerRef.current != null) {
      clearTimeout(graceTimerRef.current)
      graceTimerRef.current = null
    }
  }, [])

  // Reschedules the idle-fire timer relative to the most recent swing.
  // Replaces the old "since last batch" semantic — that one would fire mid-
  // rally if a previous batch had been quiet for 10s, regardless of whether
  // the player was actively hitting right now.
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

  // Schedules a follow-up fire attempt once the post-swing grace period has
  // expired. Used when a full-size batch is queued but the most recent swing
  // is still inside the grace window. The timer fires fireBatch() again,
  // which will either succeed (grace cleared) or no-op (grace re-armed).
  const scheduleGraceTimer = useCallback((delayMs: number) => {
    if (graceTimerRef.current != null) {
      clearTimeout(graceTimerRef.current)
      graceTimerRef.current = null
    }
    const wait = Math.max(0, delayMs)
    graceTimerRef.current = setTimeout(() => {
      graceTimerRef.current = null
      void fireBatchRef.current()
    }, wait)
  }, [])

  const fireBatch = useCallback(async () => {
    if (inFlightRef.current) return
    if (pendingRef.current.length === 0) return
    // Respect a minimum spacing so we don't fire two batches back-to-back
    // when swings arrive clustered.
    const now = Date.now()
    if (lastBatchWallMsRef.current > 0 && now - lastBatchWallMsRef.current < minBatchIntervalMs) {
      return
    }
    // Post-swing grace: if the player is still mid-rally, hold the batch
    // even when the queue is full. We only enforce this when we actually
    // know when the last swing landed (lastSwingWallMsRef > 0).
    if (lastSwingWallMsRef.current > 0) {
      const sinceLastSwing = now - lastSwingWallMsRef.current
      if (sinceLastSwing < postSwingGraceMs) {
        scheduleGraceTimer(postSwingGraceMs - sinceLastSwing)
        return
      }
    }

    const swings = pendingRef.current.slice(0, maxSwingsPerBatch)
    pendingRef.current = pendingRef.current.slice(swings.length)
    clearIdleTimer()
    clearGraceTimer()
    inFlightRef.current = true
    const batchIndex = batchIndexRef.current++
    const producedAt = Date.now()
    const sessionDurationMs = sessionStartMsRef.current
      ? producedAt - sessionStartMsRef.current
      : 0

    // Pull the last 3 transcript lines as session memory for the model. Read
    // via getState() so we don't bake transcript identity into fireBatch's
    // dependencies — that would re-create the timer-driven callback every
    // time a cue lands.
    const transcript = liveStoreRef.current.getState().transcript
    const recentCues = transcript
      .slice(-3)
      .map((entry) => entry.text)
      .filter((t) => typeof t === 'string' && t.length > 0)

    type PostOk = { kind: 'ok'; text: string; eventId: string | null }
    type PostSilence = { kind: 'silence'; eventId: string | null }
    type PostFail = { kind: 'fail' }
    type PostResult = PostOk | PostSilence | PostFail

    const tryPost = async (): Promise<PostResult> => {
      try {
        const res = await fetch('/api/live-coach', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            shotType: shotTypeRef.current,
            recentSwings: swings,
            recentCues,
            batchIndex,
            sessionDurationMs,
          }),
        })
        if (!res.ok) return { kind: 'fail' }
        const eventId = res.headers.get('X-Analysis-Event-Id')
        const text = (await res.text()).trim()
        const isSilence = res.headers.get('X-Live-Coach-Silence') === '1' || text.length === 0
        if (isSilence) return { kind: 'silence', eventId }
        return { kind: 'ok', text, eventId }
      } catch {
        return { kind: 'fail' }
      }
    }

    let result = await tryPost()
    if (result.kind === 'fail') {
      // One retry with short backoff. After the second failure we drop the
      // batch rather than block the player with retries.
      await new Promise((r) => setTimeout(r, 2_000))
      result = await tryPost()
    }

    lastBatchWallMsRef.current = Date.now()
    setLastBatchAtMs(lastBatchWallMsRef.current)
    inFlightRef.current = false
    // Notify any awaitInFlight() callers that the batch has settled.
    if (inFlightWaitersRef.current.length > 0) {
      const waiters = inFlightWaitersRef.current
      inFlightWaitersRef.current = []
      for (const w of waiters) {
        try { w() } catch { /* ignore */ }
      }
    }

    if (result.kind === 'ok') {
      // Successful cue: clear any prior error banner and append the line.
      setCoachingError(null)
      appendTranscriptEntry({
        id: `batch-${batchIndex}-${producedAt}`,
        text: result.text,
        sessionMs: sessionDurationMs,
        eventId: result.eventId,
        swingCount: swings.length,
      })
      ttsQueue.enqueue(result.text, producedAt)
    } else if (result.kind === 'silence') {
      // Model deliberately stayed silent (silence-as-cue is a first-class
      // response now). Don't surface an error and don't pollute the
      // transcript with a no-op entry — just clear any prior error so the
      // banner doesn't hang around forever.
      setCoachingError(null)
    } else {
      // Real failure: surface it visibly in the transcript header instead
      // of dropping a silent "(network hiccup)" entry into the transcript.
      setCoachingError('Coaching paused — connection issue. Retrying.')
    }

    // If more swings arrived while we were in flight, evaluate again.
    if (pendingRef.current.length >= maxSwingsPerBatch) {
      void fireBatchRef.current()
    } else if (pendingRef.current.length >= minSwingsForTimeout) {
      scheduleIdleTimer()
    }
  }, [
    appendTranscriptEntry,
    clearGraceTimer,
    clearIdleTimer,
    maxSwingsPerBatch,
    minBatchIntervalMs,
    minSwingsForTimeout,
    postSwingGraceMs,
    scheduleGraceTimer,
    scheduleIdleTimer,
    setCoachingError,
    setLastBatchAtMs,
    ttsQueue,
  ])

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
      // Mark the wall-clock arrival time so fireBatch's grace check has
      // something to compare against. The detector's startMs/endMs are
      // session-relative, not wall-clock.
      lastSwingWallMsRef.current = Date.now()
      if (pendingRef.current.length >= maxSwingsPerBatch) {
        // fireBatch enforces the post-swing grace itself; if we're still in
        // the grace window it will reschedule via scheduleGraceTimer.
        void fireBatch()
      } else if (pendingRef.current.length >= minSwingsForTimeout) {
        // idle timer is now "since last swing" — re-arm on each new swing
        // so the timer always points at the most recent activity.
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
    lastSwingWallMsRef.current = 0
  }, [])

  const primeTts = useCallback(() => {
    ttsQueue.prime()
  }, [ttsQueue])

  const reset = useCallback(() => {
    pendingRef.current = []
    batchIndexRef.current = 0
    inFlightRef.current = false
    lastBatchWallMsRef.current = 0
    lastSwingWallMsRef.current = 0
    sessionStartMsRef.current = 0
    clearIdleTimer()
    clearGraceTimer()
    setCoachingError(null)
    ttsQueue.reset()
    // Resolve any pending awaitInFlight() callers — the session is over.
    if (inFlightWaitersRef.current.length > 0) {
      const waiters = inFlightWaitersRef.current
      inFlightWaitersRef.current = []
      for (const w of waiters) {
        try { w() } catch { /* ignore */ }
      }
    }
  }, [clearGraceTimer, clearIdleTimer, setCoachingError, ttsQueue])

  const isRequestInFlight = useCallback(() => inFlightRef.current, [])

  const awaitInFlight = useCallback((timeoutMs: number) => {
    return new Promise<void>((resolve) => {
      // Fast path: nothing in flight, resolve immediately.
      if (!inFlightRef.current) {
        resolve()
        return
      }
      let settled = false
      let timeoutHandle: ReturnType<typeof setTimeout> | null = null
      const finish = () => {
        if (settled) return
        settled = true
        if (timeoutHandle != null) {
          clearTimeout(timeoutHandle)
          timeoutHandle = null
        }
        // Drop ourselves from the waiters list if still present so a later
        // batch settle doesn't double-resolve.
        const idx = inFlightWaitersRef.current.indexOf(finish)
        if (idx >= 0) inFlightWaitersRef.current.splice(idx, 1)
        resolve()
      }
      // Register a one-shot waiter; the batch-settle path flushes the list.
      inFlightWaitersRef.current.push(finish)
      // Cap the wait so a wedged request can never block Stop forever.
      const cap = Math.max(0, timeoutMs)
      if (cap > 0) {
        timeoutHandle = setTimeout(finish, cap)
      } else {
        // Zero/negative timeout still yields a microtask so the caller
        // gets a real Promise tick before resolving.
        queueMicrotask(finish)
      }
    })
  }, [])

  return { pushSwing, setShotType, markSessionStart, primeTts, reset, isRequestInFlight, awaitInFlight }
}
