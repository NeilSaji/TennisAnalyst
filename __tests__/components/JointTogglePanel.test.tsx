import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import JointTogglePanel from '@/components/JointTogglePanel'

// Mock the store module
const mockToggleJoint = vi.fn()
const mockToggleSkeleton = vi.fn()
const mockToggleTrail = vi.fn()
const mockToggleRacket = vi.fn()
const mockSetAllVisible = vi.fn()
const mockSetVisibility = vi.fn()

const defaultVisible = {
  shoulders: true,
  elbows: true,
  wrists: true,
  hips: true,
  knees: true,
  ankles: true,
}

const defaultStoreState = {
  visible: { ...defaultVisible },
  showSkeleton: true,
  showTrail: true,
  showRacket: true,
  toggleJoint: mockToggleJoint,
  toggleSkeleton: mockToggleSkeleton,
  toggleTrail: mockToggleTrail,
  toggleRacket: mockToggleRacket,
  setAllVisible: mockSetAllVisible,
  setVisibility: mockSetVisibility,
}

vi.mock('@/store', () => ({
  useJointStore: vi.fn(() => defaultStoreState),
  useComparisonStore: vi.fn(() => ({
    activeProSwing: null,
  })),
}))

// Re-import so we can change the mock return per-test
import { useJointStore } from '@/store'

const mockedUseJointStore = vi.mocked(useJointStore)

beforeEach(() => {
  vi.clearAllMocks()
  mockedUseJointStore.mockReturnValue({ ...defaultStoreState })
})

describe('JointTogglePanel', () => {
  it('renders all 6 joint group toggle buttons', () => {
    render(<JointTogglePanel />)
    expect(screen.getByText('Shoulders')).toBeInTheDocument()
    expect(screen.getByText('Elbows')).toBeInTheDocument()
    expect(screen.getByText('Wrists / Racket')).toBeInTheDocument()
    expect(screen.getByText('Hips')).toBeInTheDocument()
    expect(screen.getByText('Knees')).toBeInTheDocument()
    expect(screen.getByText('Ankles')).toBeInTheDocument()
  })

  it('renders skeleton toggle, trail toggle, and racket toggle', () => {
    render(<JointTogglePanel />)
    expect(screen.getByText('Skeleton Lines')).toBeInTheDocument()
    expect(screen.getByText('Swing Path Trail')).toBeInTheDocument()
    expect(screen.getByText('Racket')).toBeInTheDocument()
  })

  it('clicking racket toggle calls toggleRacket', () => {
    render(<JointTogglePanel />)
    fireEvent.click(screen.getByText('Racket'))
    expect(mockToggleRacket).toHaveBeenCalledOnce()
  })

  it('clicking a joint group calls toggleJoint with correct group', () => {
    render(<JointTogglePanel />)

    fireEvent.click(screen.getByText('Shoulders'))
    expect(mockToggleJoint).toHaveBeenCalledWith('shoulders')

    fireEvent.click(screen.getByText('Elbows'))
    expect(mockToggleJoint).toHaveBeenCalledWith('elbows')

    fireEvent.click(screen.getByText('Knees'))
    expect(mockToggleJoint).toHaveBeenCalledWith('knees')
  })

  it('clicking skeleton toggle calls toggleSkeleton', () => {
    render(<JointTogglePanel />)
    fireEvent.click(screen.getByText('Skeleton Lines'))
    expect(mockToggleSkeleton).toHaveBeenCalledOnce()
  })

  it('clicking trail toggle calls toggleTrail', () => {
    render(<JointTogglePanel />)
    fireEvent.click(screen.getByText('Swing Path Trail'))
    expect(mockToggleTrail).toHaveBeenCalledOnce()
  })

  it('shows "Hide all" when all joints are visible and calls setAllVisible(false)', () => {
    render(<JointTogglePanel />)
    const hideBtn = screen.getByText('Hide all')
    expect(hideBtn).toBeInTheDocument()

    fireEvent.click(hideBtn)
    expect(mockSetAllVisible).toHaveBeenCalledWith(false)
  })

  it('shows "Show all" when some joints are hidden and calls setAllVisible(true)', () => {
    mockedUseJointStore.mockReturnValue({
      ...defaultStoreState,
      visible: { ...defaultVisible, shoulders: false },
    })

    render(<JointTogglePanel />)
    const showBtn = screen.getByText('Show all')
    expect(showBtn).toBeInTheDocument()

    fireEvent.click(showBtn)
    expect(mockSetAllVisible).toHaveBeenCalledWith(true)
  })

  it('active joint buttons have different styling than toggled-off joints', () => {
    mockedUseJointStore.mockReturnValue({
      ...defaultStoreState,
      visible: { ...defaultVisible, shoulders: false, elbows: true },
    })

    render(<JointTogglePanel />)

    // The button containing "Elbows" text (active) should have active styling
    const elbowsBtn = screen.getByText('Elbows').closest('button')!
    const shouldersBtn = screen.getByText('Shoulders').closest('button')!

    // Active: 'border-white/20 bg-white/10 text-white'
    expect(elbowsBtn.className).toContain('text-white')
    expect(elbowsBtn.className).toContain('bg-white/10')

    // Inactive: 'text-white/30'
    expect(shouldersBtn.className).toContain('text-white/30')
  })
})
