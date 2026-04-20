// @vitest-environment node
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { NextRequest } from 'next/server'

// Mock the Supabase SSR client. Each test stubs the user payload through
// mockGetUser so we can exercise anonymous / pre-onboarding / completed
// states without hitting a real Supabase.
const mockGetUser = vi.fn()

vi.mock('@supabase/ssr', () => ({
  createServerClient: () => ({
    auth: { getUser: mockGetUser },
  }),
}))

// Anything-truthy values keep the createServerClient call from throwing on
// the `!` assertions in middleware.ts.
beforeEach(() => {
  process.env.NEXT_PUBLIC_SUPABASE_URL = 'http://localhost:54321'
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY = 'anon-key-test'
  mockGetUser.mockReset()
})

const VALID_METADATA = {
  skill_tier: 'intermediate',
  dominant_hand: 'right',
  backhand_style: 'two_handed',
  primary_goal: 'consistency',
  onboarded_at: '2024-01-01T00:00:00.000Z',
}

function makeRequest(pathname: string): NextRequest {
  // Pass an explicit Headers instance so NextResponse.next({ request }) is
  // happy — jsdom's fetch does not always attach one that passes Next's
  // instanceof check.
  return new NextRequest(new URL(`http://localhost${pathname}`), {
    headers: new Headers(),
  })
}

describe('middleware profile gate', () => {
  it('redirects an authed user without a profile to /onboarding with next', async () => {
    mockGetUser.mockResolvedValue({
      data: { user: { user_metadata: {} } },
    })

    const { middleware } = await import('@/middleware')
    const res = await middleware(makeRequest('/analyze'))

    expect(res.status).toBe(307) // NextResponse.redirect defaults to 307
    const location = res.headers.get('location') ?? ''
    expect(location).toContain('/onboarding')
    expect(location).toContain('next=%2Fanalyze')
  })

  it('lets an authed user with a valid profile pass through /analyze', async () => {
    mockGetUser.mockResolvedValue({
      data: { user: { user_metadata: VALID_METADATA } },
    })

    const { middleware } = await import('@/middleware')
    const res = await middleware(makeRequest('/analyze'))

    // No redirect means the response was built by NextResponse.next(), which
    // has no `location` header and status 200.
    expect(res.headers.get('location')).toBeNull()
    expect(res.status).toBe(200)
  })

  it('does not redirect an authed user without a profile when they are already on /onboarding', async () => {
    mockGetUser.mockResolvedValue({
      data: { user: { user_metadata: {} } },
    })

    const { middleware } = await import('@/middleware')
    const res = await middleware(makeRequest('/onboarding'))

    expect(res.headers.get('location')).toBeNull()
    expect(res.status).toBe(200)
  })

  it('redirects an anonymous user hitting /analyze to /login with next', async () => {
    mockGetUser.mockResolvedValue({
      data: { user: null },
    })

    const { middleware } = await import('@/middleware')
    const res = await middleware(makeRequest('/analyze'))

    expect(res.status).toBe(307)
    const location = res.headers.get('location') ?? ''
    expect(location).toContain('/login')
    expect(location).toContain('next=%2Fanalyze')
  })

  it('preserves the query string on the next param when redirecting to /onboarding', async () => {
    mockGetUser.mockResolvedValue({
      data: { user: { user_metadata: {} } },
    })

    const { middleware } = await import('@/middleware')
    const res = await middleware(makeRequest('/analyze?shot=forehand'))

    const location = res.headers.get('location') ?? ''
    expect(location).toContain('next=%2Fanalyze%3Fshot%3Dforehand')
  })

  it('lets an authed user who skipped onboarding pass through /analyze', async () => {
    mockGetUser.mockResolvedValue({
      data: {
        user: {
          user_metadata: { skipped_onboarding_at: '2024-06-01T00:00:00.000Z' },
        },
      },
    })

    const { middleware } = await import('@/middleware')
    const res = await middleware(makeRequest('/analyze'))

    // Skipped users should NOT be redirected to /onboarding — that's the
    // whole point of the skip option.
    expect(res.headers.get('location')).toBeNull()
    expect(res.status).toBe(200)
  })

  it('lets an authed user with both a full profile AND a skipped timestamp pass through', async () => {
    mockGetUser.mockResolvedValue({
      data: {
        user: {
          user_metadata: {
            ...VALID_METADATA,
            skipped_onboarding_at: '2024-06-01T00:00:00.000Z',
          },
        },
      },
    })

    const { middleware } = await import('@/middleware')
    const res = await middleware(makeRequest('/analyze'))

    expect(res.headers.get('location')).toBeNull()
    expect(res.status).toBe(200)
  })
})
