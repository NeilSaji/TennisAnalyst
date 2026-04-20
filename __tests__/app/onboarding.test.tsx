import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const mockReplace = vi.fn()
const mockRefresh = vi.fn()
const mockGet = vi.fn()
const mockUpdateUser = vi.fn()
const mockGetUser = vi.fn()
const mockOnAuthStateChange = vi.fn()
const mockUnsubscribe = vi.fn()

vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: mockReplace, refresh: mockRefresh, push: vi.fn() }),
  useSearchParams: () => ({ get: mockGet }),
}))

vi.mock('@/lib/supabase/client', () => ({
  createClient: () => ({
    auth: {
      getUser: mockGetUser,
      updateUser: mockUpdateUser,
      onAuthStateChange: mockOnAuthStateChange,
    },
  }),
}))

// We mock useProfile since it's an internal concern for this page — the form
// behavior is what we care about here.
const mockSkipOnboarding = vi.fn()

const mockProfileState = {
  profile: null as null | Record<string, unknown>,
  loading: false,
  isOnboarded: false,
  skipped: false,
  error: null,
  refresh: vi.fn(),
  updateProfile: vi.fn(),
  skipOnboarding: mockSkipOnboarding,
}

vi.mock('@/hooks/useProfile', () => ({
  useProfile: () => mockProfileState,
}))

import OnboardingPage from '@/app/onboarding/page'

const authedUser = {
  id: 'u1',
  email: 't@t.com',
  user_metadata: {},
  app_metadata: {},
  aud: 'authenticated',
  created_at: '2026-01-01T00:00:00.000Z',
}

beforeEach(() => {
  vi.clearAllMocks()
  mockGet.mockReturnValue(null)
  mockOnAuthStateChange.mockReturnValue({
    data: { subscription: { unsubscribe: mockUnsubscribe } },
  })
  mockGetUser.mockResolvedValue({ data: { user: authedUser }, error: null })
  mockUpdateUser.mockResolvedValue({ data: { user: authedUser }, error: null })
  mockProfileState.profile = null
  mockProfileState.isOnboarded = false
  mockProfileState.skipped = false
  mockProfileState.loading = false
  // Default: skipOnboarding writes to supabase via updateUser, matching how
  // the real hook behaves. Tests that need to verify payload shape can inspect
  // mockUpdateUser's call args, same as the submit path.
  mockSkipOnboarding.mockImplementation(async () => {
    await mockUpdateUser({ data: { skipped_onboarding_at: new Date().toISOString() } })
  })
})

describe('OnboardingPage', () => {
  it('renders all three sections after auth check', async () => {
    render(<OnboardingPage />)

    await waitFor(() =>
      expect(screen.getByText('How would you rate yourself?')).toBeInTheDocument(),
    )
    expect(screen.getByText('Your grip')).toBeInTheDocument()
    expect(screen.getByText('What do you want to work on?')).toBeInTheDocument()

    // All 4 tier buttons render
    expect(screen.getByText('New to tennis')).toBeInTheDocument()
    expect(screen.getByText('Intermediate')).toBeInTheDocument()
    expect(screen.getByText('Competitive')).toBeInTheDocument()
    expect(screen.getByText('Advanced or pro')).toBeInTheDocument()

    // Hand + backhand toggles
    expect(screen.getByText('Right')).toBeInTheDocument()
    expect(screen.getByText('Left')).toBeInTheDocument()
    expect(screen.getByText('One-handed')).toBeInTheDocument()
    expect(screen.getByText('Two-handed')).toBeInTheDocument()

    // Goal tiles
    expect(screen.getByText('More power')).toBeInTheDocument()
    expect(screen.getByText('More consistency')).toBeInTheDocument()
    expect(screen.getByText('Something else')).toBeInTheDocument()
  })

  it('redirects unauthed users to /login', async () => {
    mockGetUser.mockResolvedValue({ data: { user: null }, error: null })

    render(<OnboardingPage />)

    await waitFor(() => expect(mockReplace).toHaveBeenCalled())
    const arg = mockReplace.mock.calls[0][0] as string
    expect(arg).toContain('/login')
    expect(arg).toContain('next=')
  })

  it('redirects already-onboarded users to next URL', async () => {
    mockProfileState.profile = { skill_tier: 'advanced' } as unknown as Record<string, unknown>
    mockProfileState.isOnboarded = true
    mockGet.mockReturnValue('/baseline')

    render(<OnboardingPage />)

    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith('/baseline'))
  })

  it('submit is disabled until all 3 sections are picked', async () => {
    render(<OnboardingPage />)

    await waitFor(() =>
      expect(screen.getByText('How would you rate yourself?')).toBeInTheDocument(),
    )

    const submit = screen.getByRole('button', { name: /continue/i })
    expect(submit).toBeDisabled()

    fireEvent.click(screen.getByText('Intermediate'))
    expect(submit).toBeDisabled()

    fireEvent.click(screen.getByText('Right'))
    expect(submit).toBeDisabled()

    fireEvent.click(screen.getByText('Two-handed'))
    expect(submit).toBeDisabled()

    fireEvent.click(screen.getByText('More power'))
    expect(submit).not.toBeDisabled()
  })

  it('shows free-text only when "Something else" is selected', async () => {
    render(<OnboardingPage />)

    await waitFor(() =>
      expect(screen.getByText('What do you want to work on?')).toBeInTheDocument(),
    )

    // Not yet shown
    expect(screen.queryByPlaceholderText(/clean up my serve toss/i)).not.toBeInTheDocument()

    fireEvent.click(screen.getByText('More power'))
    expect(screen.queryByPlaceholderText(/clean up my serve toss/i)).not.toBeInTheDocument()

    fireEvent.click(screen.getByText('Something else'))
    expect(screen.getByPlaceholderText(/clean up my serve toss/i)).toBeInTheDocument()
  })

  it('submit calls supabase.auth.updateUser with all fields + onboarded_at', async () => {
    render(<OnboardingPage />)

    await waitFor(() =>
      expect(screen.getByText('How would you rate yourself?')).toBeInTheDocument(),
    )

    fireEvent.click(screen.getByText('Competitive'))
    fireEvent.click(screen.getByText('Left'))
    fireEvent.click(screen.getByText('One-handed'))
    fireEvent.click(screen.getByText('Cleaner topspin'))

    const submit = screen.getByRole('button', { name: /continue/i })
    fireEvent.click(submit)

    await waitFor(() => expect(mockUpdateUser).toHaveBeenCalled())
    const payload = mockUpdateUser.mock.calls[0][0].data
    expect(payload.skill_tier).toBe('competitive')
    expect(payload.dominant_hand).toBe('left')
    expect(payload.backhand_style).toBe('one_handed')
    expect(payload.primary_goal).toBe('topspin')
    expect(payload.primary_goal_note).toBeNull()
    expect(typeof payload.onboarded_at).toBe('string')
  })

  it('submit stays disabled with "Something else" selected until note is filled', async () => {
    render(<OnboardingPage />)

    await waitFor(() =>
      expect(screen.getByText('How would you rate yourself?')).toBeInTheDocument(),
    )

    fireEvent.click(screen.getByText('Intermediate'))
    fireEvent.click(screen.getByText('Right'))
    fireEvent.click(screen.getByText('Two-handed'))
    fireEvent.click(screen.getByText('Something else'))

    const submit = screen.getByRole('button', { name: /continue/i })
    // All 3 sections picked but note still empty → disabled.
    expect(submit).toBeDisabled()

    const note = screen.getByPlaceholderText(/clean up my serve toss/i) as HTMLInputElement
    // Whitespace-only doesn't count.
    fireEvent.change(note, { target: { value: '   ' } })
    expect(submit).toBeDisabled()

    fireEvent.change(note, { target: { value: 'work on my toss' } })
    expect(submit).not.toBeDisabled()
  })

  it('enforces the 120-char max on the "Something else" free-text field', async () => {
    render(<OnboardingPage />)

    await waitFor(() =>
      expect(screen.getByText('What do you want to work on?')).toBeInTheDocument(),
    )

    fireEvent.click(screen.getByText('Something else'))

    const note = screen.getByPlaceholderText(/clean up my serve toss/i) as HTMLInputElement
    expect(note.maxLength).toBe(120)
  })

  it('prevents double-submit while a save is in flight', async () => {
    // Stall updateUser so the button stays in the "saving" state.
    let resolveUpdate: (v: { data: { user: unknown }; error: null }) => void = () => {}
    mockUpdateUser.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveUpdate = resolve as typeof resolveUpdate
        }),
    )

    render(<OnboardingPage />)

    await waitFor(() =>
      expect(screen.getByText('How would you rate yourself?')).toBeInTheDocument(),
    )

    fireEvent.click(screen.getByText('Intermediate'))
    fireEvent.click(screen.getByText('Right'))
    fireEvent.click(screen.getByText('Two-handed'))
    fireEvent.click(screen.getByText('More power'))

    const submit = screen.getByRole('button', { name: /continue/i })
    fireEvent.click(submit)

    await waitFor(() => expect(mockUpdateUser).toHaveBeenCalledTimes(1))

    // While saving, the button is disabled and a second click is a no-op.
    fireEvent.click(submit)
    fireEvent.click(submit)
    expect(mockUpdateUser).toHaveBeenCalledTimes(1)

    // Let the pending promise resolve so we don't leak it.
    resolveUpdate({ data: { user: authedUser }, error: null })
  })

  it('renders a "Skip for now" link', async () => {
    render(<OnboardingPage />)

    await waitFor(() =>
      expect(screen.getByText('How would you rate yourself?')).toBeInTheDocument(),
    )

    expect(screen.getByRole('button', { name: /skip for now/i })).toBeInTheDocument()
  })

  it('clicking skip calls updateUser with skipped_onboarding_at and redirects to next', async () => {
    mockGet.mockReturnValue('/baseline')

    render(<OnboardingPage />)

    await waitFor(() =>
      expect(screen.getByText('How would you rate yourself?')).toBeInTheDocument(),
    )

    fireEvent.click(screen.getByRole('button', { name: /skip for now/i }))

    await waitFor(() => expect(mockUpdateUser).toHaveBeenCalled())
    const payload = mockUpdateUser.mock.calls[0][0].data
    expect(typeof payload.skipped_onboarding_at).toBe('string')
    expect(Number.isNaN(Date.parse(payload.skipped_onboarding_at))).toBe(false)

    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith('/baseline'))
  })

  it('disables the skip link while a skip is in flight', async () => {
    let resolveSkip: () => void = () => {}
    mockSkipOnboarding.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          resolveSkip = resolve
        }),
    )

    render(<OnboardingPage />)

    await waitFor(() =>
      expect(screen.getByText('How would you rate yourself?')).toBeInTheDocument(),
    )

    const skipBtn = screen.getByRole('button', { name: /skip for now/i })
    fireEvent.click(skipBtn)

    // Immediately after clicking, the label should flip to the "Saving…"
    // indicator and re-clicks should be no-ops.
    await waitFor(() => expect(screen.getByRole('button', { name: /saving/i })).toBeDisabled())
    fireEvent.click(skipBtn)
    expect(mockSkipOnboarding).toHaveBeenCalledTimes(1)

    resolveSkip()
  })

  it('redirects skipped users out of onboarding on initial mount', async () => {
    mockProfileState.skipped = true
    mockGet.mockReturnValue('/analyze')

    render(<OnboardingPage />)

    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith('/analyze'))
  })

  it('surfaces an error message when updateUser rejects', async () => {
    mockUpdateUser.mockResolvedValue({ data: { user: null }, error: { message: 'boom' } })

    render(<OnboardingPage />)

    await waitFor(() =>
      expect(screen.getByText('How would you rate yourself?')).toBeInTheDocument(),
    )

    fireEvent.click(screen.getByText('Intermediate'))
    fireEvent.click(screen.getByText('Right'))
    fireEvent.click(screen.getByText('Two-handed'))
    fireEvent.click(screen.getByText('More power'))

    fireEvent.click(screen.getByRole('button', { name: /continue/i }))

    await waitFor(() => expect(screen.getByText('boom')).toBeInTheDocument())
    // Button re-enables so the user can retry.
    expect(screen.getByRole('button', { name: /continue/i })).not.toBeDisabled()
  })
})
