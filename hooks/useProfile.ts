'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createClient } from '@/lib/supabase/client'
import { parseProfile, wasSkipped, type UserProfile } from '@/lib/profile'
import type { User } from '@supabase/supabase-js'

// Client-side profile state. Wraps supabase.auth.getUser() and
// re-reads on auth state changes so the UI reflects the signed-in
// user's onboarding metadata without a manual refresh.
//
// `profile` is null both when signed out and when metadata is missing
// or malformed — callers differentiate using `isOnboarded` (which is
// just "has a complete profile while signed in"). `updateProfile`
// merges partial updates over existing metadata so callers don't have
// to re-send every field.
export interface UseProfileResult {
  profile: UserProfile | null
  loading: boolean
  error: string | null
  isOnboarded: boolean
  skipped: boolean
  refresh: () => Promise<void>
  updateProfile: (data: Partial<UserProfile>) => Promise<void>
  skipOnboarding: () => Promise<void>
}

export function useProfile(): UseProfileResult {
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [skipped, setSkipped] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Stable supabase client across renders so useEffect/useCallback deps
  // don't re-subscribe on every render.
  const supabase = useMemo(() => createClient(), [])

  // Gate flag so onAuthStateChange's synthetic INITIAL_SESSION can't race
  // the explicit getUser() on mount. Only applied once mount's getUser
  // resolves.
  const initializedRef = useRef(false)

  // Monotonic counter so concurrent refresh() calls discard stale responses.
  const latestRequestIdRef = useRef(0)

  const applyUser = useCallback((next: User | null) => {
    const metadata = next?.user_metadata ?? null
    setProfile(next ? parseProfile(metadata) : null)
    setSkipped(next ? wasSkipped(metadata) : false)
  }, [])

  const refresh = useCallback(async () => {
    setError(null)
    const requestId = ++latestRequestIdRef.current
    const { data, error: err } = await supabase.auth.getUser()
    // Discard late-arriving responses from superseded calls.
    if (requestId !== latestRequestIdRef.current) return
    if (err) {
      setError(err.message)
      applyUser(null)
      return
    }
    applyUser(data.user ?? null)
  }, [supabase, applyUser])

  useEffect(() => {
    let active = true

    supabase.auth
      .getUser()
      .then(({ data, error: err }) => {
        if (!active) return
        if (err) setError(err.message)
        applyUser(data.user ?? null)
        setLoading(false)
      })
      .finally(() => {
        initializedRef.current = true
      })

    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!active) return
      // Gate: ignore the synthetic INITIAL_SESSION that fires before mount's
      // getUser() has resolved.
      if (!initializedRef.current) return
      applyUser(session?.user ?? null)
      setLoading(false)
    })

    return () => {
      active = false
      sub.subscription.unsubscribe()
    }
  }, [supabase, applyUser])

  const updateProfile = useCallback(
    async (data: Partial<UserProfile>) => {
      setError(null)
      // Re-fetch the freshest user metadata before merging so two rapid
      // updateProfile calls don't clobber fields written by the prior call.
      const { data: fresh } = await supabase.auth.getUser()
      const currentMetadata = fresh.user?.user_metadata ?? {}
      // Merge over existing metadata so partial updates don't wipe
      // untouched fields. Supabase's updateUser already shallow-merges
      // `data`, but being explicit keeps the intent readable.
      const merged: Record<string, unknown> = { ...currentMetadata, ...data }
      // Hygiene: if the caller switches primary_goal to anything other
      // than 'other', clear any prior note so the raw row doesn't carry
      // stale text that bypasses parseProfile's normalization.
      if (data.primary_goal !== undefined && data.primary_goal !== 'other') {
        merged.primary_goal_note = null
      }
      const { data: result, error: err } = await supabase.auth.updateUser({ data: merged })
      if (err) {
        setError(err.message)
        throw err
      }
      applyUser(result.user ?? null)
    },
    [supabase, applyUser],
  )

  const skipOnboarding = useCallback(async () => {
    setError(null)
    // Re-fetch freshest metadata before merging so we don't clobber a
    // concurrent updateProfile write.
    const { data: fresh } = await supabase.auth.getUser()
    const currentMetadata = fresh.user?.user_metadata ?? {}
    // No-op guard: if the user already has a complete profile, don't stamp
    // a skip timestamp over them — we'd demote an onboarded user.
    if (parseProfile(currentMetadata) !== null) return
    const merged: Record<string, unknown> = {
      ...currentMetadata,
      skipped_onboarding_at: new Date().toISOString(),
    }
    const { data: result, error: err } = await supabase.auth.updateUser({ data: merged })
    if (err) {
      setError(err.message)
      throw err
    }
    applyUser(result.user ?? null)
  }, [supabase, applyUser])

  return {
    profile,
    loading,
    error,
    isOnboarded: profile !== null,
    skipped,
    refresh,
    updateProfile,
    skipOnboarding,
  }
}
