import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import Nav from '@/components/Nav'

vi.mock('next/link', () => ({
  default: ({ children, href, className }: { children: React.ReactNode; href: string; className?: string }) => (
    <a href={href} className={className}>{children}</a>
  ),
}))

// Nav reads auth state via useUser(). Don't pull a real Supabase client into
// the test — we're verifying nav structure, not auth rendering. The default
// export is the signed-out case; signed-in tests re-mock below.
const mockUseUser = vi.fn(() => ({ user: null, loading: false }) as { user: unknown; loading: boolean })
vi.mock('@/hooks/useUser', () => ({
  useUser: () => mockUseUser(),
}))

describe('Nav', () => {
  it('renders the app name', () => {
    render(<Nav />)
    expect(screen.getByText('TennisIQ')).toBeInTheDocument()
  })

  it('renders the tennis ball emoji logo', () => {
    render(<Nav />)
    expect(screen.getByText('TennisIQ').closest('a')).toBeInTheDocument()
  })

  it('renders Analyze navigation link', () => {
    render(<Nav />)
    expect(screen.getByText('Analyze')).toBeInTheDocument()
  })

  it('renders Baselines navigation link', () => {
    render(<Nav />)
    expect(screen.getByText('Baselines')).toBeInTheDocument()
  })

  it('logo links to home page', () => {
    render(<Nav />)
    const logoLink = screen.getByText('TennisIQ').closest('a')!
    expect(logoLink).toHaveAttribute('href', '/')
  })

  it('Analyze link has correct href', () => {
    render(<Nav />)
    const link = screen.getByText('Analyze').closest('a')!
    expect(link).toHaveAttribute('href', '/analyze')
  })

  it('Baselines link has correct href', () => {
    render(<Nav />)
    const link = screen.getByText('Baselines').closest('a')!
    expect(link).toHaveAttribute('href', '/baseline')
  })

  it('shows Profile link in AccountMenu when signed in', () => {
    mockUseUser.mockReturnValueOnce({ user: { email: 'test@example.com' }, loading: false })
    render(<Nav />)
    const profileLink = screen.getByText('Profile').closest('a')!
    expect(profileLink).toHaveAttribute('href', '/profile')
  })

  it('does not render Profile link when signed out', () => {
    render(<Nav />)
    expect(screen.queryByText('Profile')).not.toBeInTheDocument()
  })
})
