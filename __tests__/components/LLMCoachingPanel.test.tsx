import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import LLMCoachingPanel from '@/components/LLMCoachingPanel'

// Default store values
let mockAnalysisState = {
  feedback: '',
  loading: false,
  setFeedback: vi.fn(),
  appendFeedback: vi.fn(),
  setLoading: vi.fn(),
  reset: vi.fn(),
}

let mockPoseState = {
  framesData: [] as Array<Record<string, unknown>>,
  sessionId: null as string | null,
}

vi.mock('@/store', () => ({
  useAnalysisStore: Object.assign(
    vi.fn(() => mockAnalysisState),
    { getState: vi.fn(() => ({ feedback: '' })) }
  ),
  usePoseStore: vi.fn(() => mockPoseState),
}))

// useUser is mocked per-test via mockUseUser.mockReturnValue.
const mockUseUser = vi.fn(() => ({ user: { id: 'u1' }, loading: false }) as { user: unknown; loading: boolean })
vi.mock('@/hooks/useUser', () => ({
  useUser: () => mockUseUser(),
}))

// Helper: build a fetch Response whose body streams the given chunks and
// whose headers include the given event id. Mirrors what the real
// /api/analyze route emits.
function streamingResponse(chunks: string[], headers: Record<string, string> = {}) {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c))
      controller.close()
    },
  })
  return new Response(body, { status: 200, headers })
}

describe('LLMCoachingPanel', () => {
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.clearAllMocks()
    mockAnalysisState = {
      feedback: '',
      loading: false,
      setFeedback: vi.fn((v: string) => {
        mockAnalysisState.feedback = v
      }),
      // Mutate shared state so a re-render (triggered by any subsequent
      // setState in the component) picks up the streamed feedback text.
      appendFeedback: vi.fn((v: string) => {
        mockAnalysisState.feedback += v
      }),
      setLoading: vi.fn(),
      reset: vi.fn(() => {
        mockAnalysisState.feedback = ''
      }),
    }
    mockPoseState = {
      framesData: [],
      sessionId: null,
    }
    mockUseUser.mockReturnValue({ user: { id: 'u1' }, loading: false })
    globalThis.fetch = vi.fn()
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  it('disables Analyze Swing when no frames are loaded', () => {
    mockPoseState.framesData = []

    render(<LLMCoachingPanel />)

    const analyzeBtn = screen.getByText('Analyze Swing')
    expect(analyzeBtn).toBeInTheDocument()
    expect(analyzeBtn.closest('button')).toBeDisabled()
  })

  it('enables Analyze Swing once frames are present (solo mode)', () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    render(<LLMCoachingPanel />)

    const btn = screen.getByText('Analyze Swing')
    expect(btn.closest('button')).not.toBeDisabled()
  })

  it('shows loading state (spinner) during analysis', () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]
    mockAnalysisState.loading = true

    render(<LLMCoachingPanel />)

    expect(screen.getByText('Analyzing...')).toBeInTheDocument()
  })

  it('shows "Form Analysis" label when compareMode is custom', () => {
    render(<LLMCoachingPanel compareMode="custom" />)

    expect(screen.getByText('Form Analysis')).toBeInTheDocument()
  })

  it('shows baseline label when compareMode is baseline', () => {
    render(<LLMCoachingPanel compareMode="baseline" baselineLabel="May 3 rally" />)

    expect(screen.getByText(/May 3 rally/)).toBeInTheDocument()
  })

  it('renders feedback text correctly', () => {
    mockAnalysisState.feedback = '## Great Form\nYour **backswing** is solid.'

    render(<LLMCoachingPanel />)

    expect(screen.getByText('Expand')).toBeInTheDocument()
  })

  it('expand/collapse toggle works', () => {
    mockAnalysisState.feedback = 'Some coaching feedback here.'

    render(<LLMCoachingPanel />)

    const expandBtn = screen.getByText('Expand')
    fireEvent.click(expandBtn)

    expect(screen.getByText('Collapse')).toBeInTheDocument()
    expect(screen.getByText('Some coaching feedback here.')).toBeInTheDocument()
  })

  it('re-analyze button appears after first analysis', () => {
    mockAnalysisState.feedback = 'Analysis complete.'
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    render(<LLMCoachingPanel />)

    expect(screen.getByText('Re-analyze')).toBeInTheDocument()
  })

  it('shows markdown headers as h3 elements when panel is expanded', () => {
    mockAnalysisState.feedback = '## Swing Analysis\nLooks good.'

    render(<LLMCoachingPanel />)

    fireEvent.click(screen.getByText('Expand'))

    expect(screen.getByText('Swing Analysis')).toBeInTheDocument()
    expect(screen.getByText('Swing Analysis').tagName).toBe('H3')
  })

  it('renders bold text in markdown', () => {
    mockAnalysisState.feedback = 'Your **technique** is improving.'

    render(<LLMCoachingPanel />)

    fireEvent.click(screen.getByText('Expand'))

    const strongEl = screen.getByText('technique')
    expect(strongEl.tagName).toBe('STRONG')
  })

  // --- Thumbs feedback strip ----------------------------------------------

  it('does not render thumbs strip while loading (even with feedback)', () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]
    mockAnalysisState.feedback = 'partial...'
    mockAnalysisState.loading = true

    render(<LLMCoachingPanel />)

    expect(screen.queryByText('Was this coaching right for you?')).not.toBeInTheDocument()
  })

  it('does not render thumbs strip when feedback is empty', () => {
    mockAnalysisState.feedback = ''
    mockAnalysisState.loading = false

    render(<LLMCoachingPanel />)

    expect(screen.queryByText('Was this coaching right for you?')).not.toBeInTheDocument()
  })

  it('does not render thumbs strip when user is signed out', async () => {
    mockUseUser.mockReturnValue({ user: null, loading: false })
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    // Simulate a completed analysis: event id header present, stream closed.
    ;(globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      streamingResponse(['feedback body'], { 'X-Analysis-Event-Id': 'evt-1' })
    )

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })

    // Even after a successful analyze, signed-out users should never see the strip.
    expect(screen.queryByText('Was this coaching right for you?')).not.toBeInTheDocument()
  })

  it('clicking "Spot on" PATCHes the feedback endpoint with correct body', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    // First call: /api/analyze streaming response with event id header.
    fetchMock.mockResolvedValueOnce(
      streamingResponse(['coaching text'], { 'X-Analysis-Event-Id': 'evt-42' })
    )
    // Second call: PATCH feedback endpoint.
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }))

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })

    // Strip should now be visible.
    expect(screen.getByText('Was this coaching right for you?')).toBeInTheDocument()

    await act(async () => {
      fireEvent.click(screen.getByText('👍 Spot on'))
    })

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/analysis-events/evt-42/feedback',
        expect.objectContaining({
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ correction: 'correct' }),
        })
      )
    })
  })

  it('clicking "Too advanced" sends correction=too_hard', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse(['coaching'], { 'X-Analysis-Event-Id': 'evt-hard' })
    )
    fetchMock.mockResolvedValueOnce(new Response('{"ok":true}', { status: 200 }))

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })
    await act(async () => {
      fireEvent.click(screen.getByText('⬇️ Too advanced'))
    })

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/analysis-events/evt-hard/feedback',
        expect.objectContaining({
          body: JSON.stringify({ correction: 'too_hard' }),
        })
      )
    })
  })

  it('clicking "Too simple" sends correction=too_easy', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse(['coaching'], { 'X-Analysis-Event-Id': 'evt-easy' })
    )
    fetchMock.mockResolvedValueOnce(new Response('{"ok":true}', { status: 200 }))

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })
    await act(async () => {
      fireEvent.click(screen.getByText('⬆️ Too simple'))
    })

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/analysis-events/evt-easy/feedback',
        expect.objectContaining({
          body: JSON.stringify({ correction: 'too_easy' }),
        })
      )
    })
  })

  it('collapses to thank-you state after a successful PATCH', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse(['coaching'], { 'X-Analysis-Event-Id': 'evt-ok' })
    )
    fetchMock.mockResolvedValueOnce(new Response('{"ok":true}', { status: 200 }))

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })
    await act(async () => {
      fireEvent.click(screen.getByText('👍 Spot on'))
    })

    await waitFor(() => {
      expect(screen.getByText('Thanks — logged.')).toBeInTheDocument()
    })
    expect(screen.queryByText('Was this coaching right for you?')).not.toBeInTheDocument()
    expect(screen.queryByText('👍 Spot on')).not.toBeInTheDocument()
  })

  it('shows error message on PATCH failure and keeps buttons enabled', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse(['coaching'], { 'X-Analysis-Event-Id': 'evt-err' })
    )
    // PATCH returns 500.
    fetchMock.mockResolvedValueOnce(new Response('nope', { status: 500 }))

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })
    await act(async () => {
      fireEvent.click(screen.getByText('👍 Spot on'))
    })

    await waitFor(() => {
      expect(screen.getByText(/Couldn.t save/)).toBeInTheDocument()
    })

    // Buttons should still be present and enabled for retry.
    const spotOn = screen.getByText('👍 Spot on').closest('button')!
    expect(spotOn).not.toBeDisabled()
    const tooAdv = screen.getByText('⬇️ Too advanced').closest('button')!
    expect(tooAdv).not.toBeDisabled()
  })

  it('resets feedbackState when feedback text clears (new analysis cycle)', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse(['coaching'], { 'X-Analysis-Event-Id': 'evt-reset' })
    )
    fetchMock.mockResolvedValueOnce(new Response('{"ok":true}', { status: 200 }))

    const { rerender } = render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })
    await act(async () => {
      fireEvent.click(screen.getByText('👍 Spot on'))
    })
    await waitFor(() => expect(screen.getByText('Thanks — logged.')).toBeInTheDocument())

    // Simulate a new analysis starting: feedback wiped, loading flips on.
    mockAnalysisState = {
      ...mockAnalysisState,
      feedback: '',
      loading: true,
    }
    rerender(<LLMCoachingPanel />)

    // Thank-you and strip should both be gone while the next analysis runs.
    expect(screen.queryByText('Thanks — logged.')).not.toBeInTheDocument()
    expect(screen.queryByText('Was this coaching right for you?')).not.toBeInTheDocument()
  })
})
