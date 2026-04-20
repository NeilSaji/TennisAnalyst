import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'

// Shared handles the test mutates between cases.
const mockGetUser = vi.fn()
const mockUpdateUser = vi.fn()
const mockOnAuthStateChange = vi.fn()
const mockUnsubscribe = vi.fn()

vi.mock('@/lib/supabase/client', () => ({
  createClient: () => ({
    auth: {
      getUser: mockGetUser,
      updateUser: mockUpdateUser,
      onAuthStateChange: mockOnAuthStateChange,
    },
  }),
}))

import { useProfile } from '@/hooks/useProfile'

const COMPLETE_METADATA = {
  skill_tier: 'intermediate' as const,
  dominant_hand: 'right' as const,
  backhand_style: 'two_handed' as const,
  primary_goal: 'consistency' as const,
  primary_goal_note: null,
  onboarded_at: '2026-04-17T00:00:00.000Z',
}

const userWith = (metadata: Record<string, unknown> | null) => ({
  id: 'u1',
  email: 'test@example.com',
  user_metadata: metadata ?? {},
  app_metadata: {},
  aud: 'authenticated',
  created_at: '2026-01-01T00:00:00.000Z',
})

beforeEach(() => {
  vi.clearAllMocks()
  mockOnAuthStateChange.mockReturnValue({
    data: { subscription: { unsubscribe: mockUnsubscribe } },
  })
})

describe('useProfile', () => {
  it('returns a parsed profile when metadata is complete', async () => {
    mockGetUser.mockResolvedValue({ data: { user: userWith(COMPLETE_METADATA) }, error: null })

    const { result } = renderHook(() => useProfile())

    expect(result.current.loading).toBe(true)

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.profile).toEqual(COMPLETE_METADATA)
    expect(result.current.isOnboarded).toBe(true)
    expect(result.current.error).toBeNull()
  })

  it('returns null profile when metadata is missing', async () => {
    mockGetUser.mockResolvedValue({ data: { user: userWith({}) }, error: null })

    const { result } = renderHook(() => useProfile())

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.profile).toBeNull()
    expect(result.current.isOnboarded).toBe(false)
  })

  it('returns null profile when metadata is malformed', async () => {
    mockGetUser.mockResolvedValue({
      data: { user: userWith({ skill_tier: 'legendary' }) },
      error: null,
    })

    const { result } = renderHook(() => useProfile())

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.profile).toBeNull()
    expect(result.current.isOnboarded).toBe(false)
  })

  it('returns null profile when signed out', async () => {
    mockGetUser.mockResolvedValue({ data: { user: null }, error: null })

    const { result } = renderHook(() => useProfile())

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.profile).toBeNull()
    expect(result.current.isOnboarded).toBe(false)
  })

  it('toggles loading from true to false on first resolve', async () => {
    let resolveUser: (v: { data: { user: unknown }; error: null }) => void = () => {}
    mockGetUser.mockReturnValue(
      new Promise((resolve) => {
        resolveUser = resolve as typeof resolveUser
      }),
    )

    const { result } = renderHook(() => useProfile())
    expect(result.current.loading).toBe(true)

    await act(async () => {
      resolveUser({ data: { user: userWith(COMPLETE_METADATA) }, error: null })
    })

    await waitFor(() => expect(result.current.loading).toBe(false))
  })

  it('updateProfile calls updateUser with metadata merged over existing fields', async () => {
    const existing = { ...COMPLETE_METADATA, custom_field: 'keep-me' }
    mockGetUser.mockResolvedValue({ data: { user: userWith(existing) }, error: null })
    mockUpdateUser.mockResolvedValue({
      data: { user: userWith({ ...existing, skill_tier: 'advanced' }) },
      error: null,
    })

    const { result } = renderHook(() => useProfile())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.updateProfile({ skill_tier: 'advanced' })
    })

    expect(mockUpdateUser).toHaveBeenCalledOnce()
    const call = mockUpdateUser.mock.calls[0][0]
    expect(call.data.skill_tier).toBe('advanced')
    // merged, not replaced
    expect(call.data.custom_field).toBe('keep-me')
    expect(call.data.dominant_hand).toBe('right')
    // and the local profile reflects the updated metadata
    expect(result.current.profile?.skill_tier).toBe('advanced')
  })

  it('updateProfile clears primary_goal_note when switching goal away from "other"', async () => {
    const existing = {
      ...COMPLETE_METADATA,
      primary_goal: 'other' as const,
      primary_goal_note: 'stale note',
    }
    mockGetUser.mockResolvedValue({ data: { user: userWith(existing) }, error: null })
    mockUpdateUser.mockResolvedValue({
      data: { user: userWith({ ...existing, primary_goal: 'power', primary_goal_note: null }) },
      error: null,
    })

    const { result } = renderHook(() => useProfile())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.updateProfile({ primary_goal: 'power' })
    })

    expect(mockUpdateUser).toHaveBeenCalledOnce()
    const call = mockUpdateUser.mock.calls[0][0]
    expect(call.data.primary_goal).toBe('power')
    expect(call.data.primary_goal_note).toBeNull()
  })

  it('updateProfile surfaces errors via throw + error state', async () => {
    mockGetUser.mockResolvedValue({ data: { user: userWith(COMPLETE_METADATA) }, error: null })
    mockUpdateUser.mockResolvedValue({ data: { user: null }, error: { message: 'nope' } })

    const { result } = renderHook(() => useProfile())
    await waitFor(() => expect(result.current.loading).toBe(false))

    let threw: unknown = null
    await act(async () => {
      try {
        await result.current.updateProfile({ skill_tier: 'advanced' })
      } catch (e) {
        threw = e
      }
    })
    expect(threw).toBeTruthy()

    await waitFor(() => expect(result.current.error).toBe('nope'))
  })

  it('onAuthStateChange updates profile when a user signs in mid-session', async () => {
    // Start signed out.
    mockGetUser.mockResolvedValue({ data: { user: null }, error: null })

    // Capture the subscription callback so we can drive it from the test.
    let authCb: ((event: string, session: { user: unknown } | null) => void) | null = null
    mockOnAuthStateChange.mockImplementation((cb) => {
      authCb = cb
      return { data: { subscription: { unsubscribe: mockUnsubscribe } } }
    })

    const { result } = renderHook(() => useProfile())
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.profile).toBeNull()

    // Now simulate a mid-session sign-in event.
    await act(async () => {
      authCb?.('SIGNED_IN', { user: userWith(COMPLETE_METADATA) })
    })

    await waitFor(() => expect(result.current.profile).toEqual(COMPLETE_METADATA))
    expect(result.current.isOnboarded).toBe(true)
  })

  it('unsubscribes from onAuthStateChange when unmounted', async () => {
    mockGetUser.mockResolvedValue({ data: { user: userWith(COMPLETE_METADATA) }, error: null })

    const { unmount, result } = renderHook(() => useProfile())
    await waitFor(() => expect(result.current.loading).toBe(false))

    expect(mockUnsubscribe).not.toHaveBeenCalled()
    unmount()
    expect(mockUnsubscribe).toHaveBeenCalledTimes(1)
  })

  it('exposes skipped=true when skipped_onboarding_at is set', async () => {
    mockGetUser.mockResolvedValue({
      data: { user: userWith({ skipped_onboarding_at: '2026-04-17T12:00:00.000Z' }) },
      error: null,
    })

    const { result } = renderHook(() => useProfile())

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.skipped).toBe(true)
    // Skipped users still don't have a complete profile.
    expect(result.current.profile).toBeNull()
    expect(result.current.isOnboarded).toBe(false)
  })

  it('exposes skipped=false when metadata lacks skipped_onboarding_at', async () => {
    mockGetUser.mockResolvedValue({ data: { user: userWith({}) }, error: null })

    const { result } = renderHook(() => useProfile())

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.skipped).toBe(false)
  })

  it('skipOnboarding writes an ISO timestamp and merges with existing metadata', async () => {
    const existing = { custom_field: 'keep-me', other: 1 }
    mockGetUser.mockResolvedValue({ data: { user: userWith(existing) }, error: null })
    mockUpdateUser.mockImplementation(({ data }) =>
      Promise.resolve({ data: { user: userWith(data as Record<string, unknown>) }, error: null }),
    )

    const { result } = renderHook(() => useProfile())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.skipOnboarding()
    })

    expect(mockUpdateUser).toHaveBeenCalledOnce()
    const call = mockUpdateUser.mock.calls[0][0]
    expect(typeof call.data.skipped_onboarding_at).toBe('string')
    // ISO 8601 sanity check — must parse as a valid date.
    expect(Number.isNaN(Date.parse(call.data.skipped_onboarding_at))).toBe(false)
    // Merge-not-replace: prior metadata survives.
    expect(call.data.custom_field).toBe('keep-me')
    expect(call.data.other).toBe(1)
    // Local state picks up the new flag.
    expect(result.current.skipped).toBe(true)
  })

  it('skipOnboarding is a no-op when the user already has a complete profile', async () => {
    // Already-onboarded user shouldn't be demoted to "skipped" if skipOnboarding
    // is called by mistake.
    mockGetUser.mockResolvedValue({ data: { user: userWith(COMPLETE_METADATA) }, error: null })

    const { result } = renderHook(() => useProfile())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.skipOnboarding()
    })

    expect(mockUpdateUser).not.toHaveBeenCalled()
    expect(result.current.skipped).toBe(false)
  })

  it('discards stale responses from concurrent refresh() calls', async () => {
    // Initial load with a complete profile.
    mockGetUser.mockResolvedValueOnce({ data: { user: userWith(COMPLETE_METADATA) }, error: null })

    const { result } = renderHook(() => useProfile())
    await waitFor(() => expect(result.current.loading).toBe(false))

    // Two refresh()es. The first is slow (advanced), the second is fast (beginner).
    // The late-arriving slow response must NOT overwrite the fast one.
    let resolveSlow: (v: { data: { user: unknown }; error: null }) => void = () => {}
    const slowPromise = new Promise((resolve) => {
      resolveSlow = resolve as typeof resolveSlow
    })
    const slowUser = userWith({ ...COMPLETE_METADATA, skill_tier: 'advanced' })
    const fastUser = userWith({ ...COMPLETE_METADATA, skill_tier: 'beginner' })

    mockGetUser.mockReturnValueOnce(slowPromise)
    mockGetUser.mockResolvedValueOnce({ data: { user: fastUser }, error: null })

    await act(async () => {
      const first = result.current.refresh()
      const second = result.current.refresh()
      // Let the fast one settle first.
      await second
      // Then release the slow one.
      resolveSlow({ data: { user: slowUser }, error: null })
      await first
    })

    // Final state must reflect the fast (most-recent) call, not the stale slow one.
    expect(result.current.profile?.skill_tier).toBe('beginner')
  })
})
