import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import type { LiveSessionResult, UseLiveCaptureReturn } from '@/hooks/useLiveCapture'
import type { StreamedSwing } from '@/lib/liveSwingDetector'

// --- Stubs ----------------------------------------------------------------

// Avoid the real router (which needs Next's app-router context).
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    replace: vi.fn(),
    push: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
    prefetch: vi.fn(),
  }),
}))

// We need to control useLiveCapture so the test can drive Stop and produce
// a deterministic result without going through MediaRecorder/getUserMedia.
const stopMock = vi.fn<() => Promise<LiveSessionResult | null>>()
const startMock = vi.fn(async () => undefined)
const abortMock = vi.fn()
let captureRecording = false

vi.mock('@/hooks/useLiveCapture', () => {
  return {
    useLiveCapture: (): UseLiveCaptureReturn => ({
      start: startMock,
      stop: stopMock,
      abort: abortMock,
      status: 'idle',
      error: null,
      isRecording: captureRecording,
      pickedMimeType: null,
    }),
  }
})

// useLiveCoach: avoid TTS / store interactions, expose all required handles.
vi.mock('@/hooks/useLiveCoach', () => {
  return {
    useLiveCoach: () => ({
      pushSwing: vi.fn(),
      setShotType: vi.fn(),
      markSessionStart: vi.fn(),
      primeTts: vi.fn(),
      reset: vi.fn(),
      isRequestInFlight: vi.fn(() => false),
      awaitInFlight: vi.fn(async () => undefined),
    }),
  }
})

// Vercel Blob client upload — controllable from the test.
const uploadMock = vi.fn()
vi.mock('@vercel/blob/client', () => ({
  upload: (...args: unknown[]) => uploadMock(...args),
}))

// IndexedDB recovery — replaced with simple in-memory mocks.
const getOrphanMock = vi.fn<() => Promise<unknown>>()
const saveOrphanMock = vi.fn(async () => undefined)
const clearOrphanMock = vi.fn(async () => undefined)
vi.mock('@/lib/liveSessionRecovery', () => ({
  getOrphanedSession: () => getOrphanMock(),
  saveOrphanedSession: (...args: unknown[]) => saveOrphanMock(...(args as [])),
  clearOrphanedSession: () => clearOrphanMock(),
}))

import LiveCapturePanel from '@/components/LiveCapturePanel'
import { useLiveStore } from '@/store/live'

// --- Fixtures -------------------------------------------------------------

function makeSwing(i: number): StreamedSwing {
  return {
    swingIndex: i + 1,
    startFrameIndex: i * 10,
    endFrameIndex: i * 10 + 9,
    peakFrameIndex: i * 10 + 5,
    startMs: i * 1000,
    endMs: i * 1000 + 900,
    frames: [],
  }
}

function makeLiveResult(swings = 3, durationMs = 12_000): LiveSessionResult {
  return {
    blob: new Blob(['v'], { type: 'video/webm' }),
    blobMimeType: 'video/webm',
    keypoints: { fps_sampled: 15, frame_count: 0, frames: [], schema_version: 2 },
    swings: Array.from({ length: swings }, (_, i) => makeSwing(i)),
    durationMs,
  }
}

// jsdom doesn't implement createObjectURL/revoke — stub them.
beforeEach(() => {
  global.URL.createObjectURL = vi.fn(() => 'blob:mock')
  global.URL.revokeObjectURL = vi.fn()

  stopMock.mockReset()
  startMock.mockClear()
  abortMock.mockClear()
  uploadMock.mockReset()
  getOrphanMock.mockReset()
  getOrphanMock.mockResolvedValue(null)
  saveOrphanMock.mockClear()
  clearOrphanMock.mockClear()
  captureRecording = false

  // Reset Zustand live store between tests so error-message bleed-through
  // can't fail an unrelated test.
  useLiveStore.setState({
    status: 'idle',
    errorMessage: null,
    swingCount: 0,
    transcript: [],
    sessionStartedAtMs: null,
  })
})

afterEach(() => {
  vi.clearAllMocks()
})

// Helper that simulates "user finished a recording" — populates stop() and
// flips through Start → Stop. Returns once the review screen is visible.
async function simulateStop(result: LiveSessionResult) {
  stopMock.mockResolvedValue(result)
  // Pretend we're recording so the Stop button renders.
  captureRecording = true
  const utils = render(<LiveCapturePanel />)
  // Click Stop.
  const stopBtn = screen.getByRole('button', { name: /^stop$/i })
  await act(async () => {
    fireEvent.click(stopBtn)
  })
  // Stop flips the panel out of recording mode for the review screen.
  captureRecording = false
  await waitFor(() => {
    expect(screen.getByTestId('review-screen')).toBeInTheDocument()
  })
  return utils
}

// --- Tests ----------------------------------------------------------------

describe('LiveCapturePanel — post-stop review flow', () => {
  it('renders the review screen with swing count and duration after Stop', async () => {
    await simulateStop(makeLiveResult(3, 12_400))

    const review = screen.getByTestId('review-screen')
    expect(review).toBeInTheDocument()
    expect(review).toHaveTextContent('3')
    expect(review).toHaveTextContent('swings')
    // 12.4s rounds to 12s.
    expect(review).toHaveTextContent('12s')
    // The inline player is mounted with the object URL.
    expect(screen.getByTestId('review-video')).toHaveAttribute('src', 'blob:mock')
    // Save / Discard / Re-record all present.
    expect(screen.getByRole('button', { name: /^save$/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^discard$/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^re-record$/i })).toBeInTheDocument()
    // Upload was NOT triggered automatically — Save is now opt-in.
    expect(uploadMock).not.toHaveBeenCalled()
  })

  it('Discard clears the review screen and returns to pre-start', async () => {
    await simulateStop(makeLiveResult(2))

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /^discard$/i }))
    })

    expect(screen.queryByTestId('review-screen')).not.toBeInTheDocument()
    // Start button is back, Stop is gone.
    expect(screen.getByRole('button', { name: /^start$/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^stop$/i })).not.toBeInTheDocument()
    // No upload happened.
    expect(uploadMock).not.toHaveBeenCalled()
  })

  it('Retry re-fires the upload after a simulated failure', async () => {
    await simulateStop(makeLiveResult(3))

    // First Save fails at the upload step.
    uploadMock.mockRejectedValueOnce(new Error('network down'))

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /^save$/i }))
    })

    // Error box with Retry button now visible.
    await waitFor(() => {
      expect(screen.getByTestId('error-box')).toBeInTheDocument()
    })
    expect(screen.getByText(/network down/i)).toBeInTheDocument()
    expect(saveOrphanMock).toHaveBeenCalledTimes(1)

    // Configure the second attempt to succeed at upload AND save.
    uploadMock.mockResolvedValueOnce({
      url: 'https://blob.test.public.blob.vercel-storage.com/live/x.webm',
    })
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ sessionId: 'sess-1' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    globalThis.fetch = fetchMock as unknown as typeof globalThis.fetch

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /^retry$/i }))
    })

    await waitFor(() => {
      expect(uploadMock).toHaveBeenCalledTimes(2)
    })
    // The session-save API was hit on the retry.
    expect(fetchMock).toHaveBeenCalled()
    // A successful save clears any prior orphan.
    await waitFor(() => {
      expect(clearOrphanMock).toHaveBeenCalled()
    })
  })

  it('shows the resume prompt when an orphan exists in IndexedDB on mount', async () => {
    getOrphanMock.mockResolvedValueOnce({
      blob: new Blob(['orphaned'], { type: 'video/webm' }),
      keypoints: { fps_sampled: 15, frame_count: 0, frames: [], schema_version: 2 },
      swings: [makeSwing(0), makeSwing(1)],
      shotType: 'forehand',
      savedAt: Date.now() - 60_000,
    })

    render(<LiveCapturePanel />)

    await waitFor(() => {
      expect(screen.getByTestId('resume-orphan-prompt')).toBeInTheDocument()
    })
    const prompt = screen.getByTestId('resume-orphan-prompt')
    expect(prompt).toHaveTextContent(/we saved a session/i)
    expect(prompt).toHaveTextContent('2')
    // Resume + Discard buttons are present.
    expect(screen.getAllByRole('button', { name: /^resume$/i })).toHaveLength(1)
  })

  it('does NOT render the resume prompt when no orphan exists', async () => {
    getOrphanMock.mockResolvedValueOnce(null)
    render(<LiveCapturePanel />)
    // Nothing to await — the absence of the orphan should leave the prompt
    // hidden after the initial effect ticks.
    await waitFor(() => {
      expect(screen.queryByTestId('resume-orphan-prompt')).not.toBeInTheDocument()
    })
  })

  it('Save (happy path) navigates without rendering the error box', async () => {
    await simulateStop(makeLiveResult(1))

    uploadMock.mockResolvedValueOnce({
      url: 'https://blob.test.public.blob.vercel-storage.com/live/x.webm',
    })
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ sessionId: 'sess-2' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    globalThis.fetch = fetchMock as unknown as typeof globalThis.fetch

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /^save$/i }))
    })

    await waitFor(() => {
      expect(uploadMock).toHaveBeenCalledTimes(1)
    })
    expect(screen.queryByTestId('error-box')).not.toBeInTheDocument()
  })
})
