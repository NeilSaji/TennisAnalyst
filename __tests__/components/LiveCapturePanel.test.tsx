/**
 * Phase 3 — Visibility worker tests for LiveCapturePanel's
 * recording-state UI surface:
 *   - Tracking-quality pill copy + color for each onPoseQuality state
 *   - Canvas overlay mounts only while recording
 *   - Swing-detected pulse triggers when onSwing fires
 *   - Status footer renders the human-readable label, not the bare
 *     store status string
 *
 * The capture panel pulls in useLiveCapture (MediaPipe + camera) and
 * useLiveCoach (TTS + fetch). We mock both at the hook level so the
 * test stays a pure UI assertion against the panel's state-driven
 * rendering — and so we can synthetically fire `onPoseQuality` /
 * `onSwing` without spinning up real device APIs.
 */
import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import type { PoseQuality } from '@/hooks/useLiveCapture'
import type { StreamedSwing } from '@/lib/liveSwingDetector'

// --- Mocks ------------------------------------------------------------------

// Pull useLiveCapture's options out so the test can drive its callbacks.
type CaptureOpts = {
  onSwing?: (s: StreamedSwing) => void
  onStatus?: (s: string) => void
  onPoseQuality?: (q: PoseQuality) => void
  onPoseFrame?: (f: unknown) => void
}
let lastCaptureOpts: CaptureOpts | null = null
let captureIsRecording = true
const captureStart = vi.fn(async () => {})
const captureStop = vi.fn(async () => null)
const captureAbort = vi.fn()

vi.mock('@/hooks/useLiveCapture', async (orig) => {
  const actual = (await orig()) as Record<string, unknown>
  return {
    ...actual,
    useLiveCapture: (opts: CaptureOpts) => {
      lastCaptureOpts = opts
      return {
        start: captureStart,
        stop: captureStop,
        abort: captureAbort,
        status: captureIsRecording ? 'recording' : 'idle',
        error: null,
        isRecording: captureIsRecording,
        pickedMimeType: 'video/webm',
      }
    },
  }
})

// useLiveCoach is a shared dep — we don't drive it here, just stub
// every method the panel calls.
const coachInFlightRef = { value: false }
const coachStub = {
  pushSwing: vi.fn(),
  setShotType: vi.fn(),
  markSessionStart: vi.fn(),
  primeTts: vi.fn(),
  reset: vi.fn(),
  isRequestInFlight: () => coachInFlightRef.value,
}
vi.mock('@/hooks/useLiveCoach', () => ({
  useLiveCoach: () => coachStub,
}))

// next/navigation: panel uses router.replace on success.
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}))

// Vercel blob upload — the stop/save flow is Worker 5's territory but
// the import chain reaches it. Stub so it's a no-op when (rare) test
// paths hit it.
vi.mock('@vercel/blob/client', () => ({
  upload: vi.fn(),
}))

// PoseRenderer: the rAF loop calls renderPose on every paint. Spy on
// it so the swing-pulse alpha-scale assertion is observable.
const renderPoseSpy = vi.fn()
vi.mock('@/components/PoseRenderer', async () => {
  return {
    renderPose: (...args: unknown[]) => renderPoseSpy(...args),
  }
})

// jsdom doesn't implement getContext('2d') — give the panel a no-op
// context so the rAF loop's draw step can run.
beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = (() => ({
    drawImage: vi.fn(),
    clearRect: vi.fn(),
    setTransform: vi.fn(),
    scale: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    fillRect: vi.fn(),
    fillText: vi.fn(),
    measureText: vi.fn(() => ({ width: 0 })),
    translate: vi.fn(),
    rotate: vi.fn(),
    quadraticCurveTo: vi.fn(),
    closePath: vi.fn(),
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    globalAlpha: 1,
    font: '',
    textAlign: 'left',
    textBaseline: 'alphabetic',
  })) as unknown as HTMLCanvasElement['getContext']

  lastCaptureOpts = null
  renderPoseSpy.mockClear()
  coachInFlightRef.value = false
  captureIsRecording = true
})

import LiveCapturePanel from '@/components/LiveCapturePanel'
import { useLiveStore } from '@/store/live'

function fakeSwing(index: number): StreamedSwing {
  return {
    swingIndex: index,
    startFrameIndex: index * 10,
    endFrameIndex: index * 10 + 5,
    peakFrameIndex: index * 10 + 3,
    startMs: index * 1000,
    endMs: index * 1000 + 500,
    frames: [],
  }
}

// --- Tests ------------------------------------------------------------------

describe('LiveCapturePanel — Phase 3 visibility surface', () => {
  beforeEach(() => {
    // Reset shared store state between tests so swing counts /
    // statuses don't leak.
    useLiveStore.setState({
      status: 'recording',
      swingCount: 0,
      transcript: [],
      coachRequestInFlight: false,
    })
  })

  it('mounts the canvas overlay while recording', () => {
    captureIsRecording = true
    render(<LiveCapturePanel />)
    const canvas = screen.getByTestId('pose-overlay-canvas')
    expect(canvas.tagName).toBe('CANVAS')
    // Mirrored to match the video's selfie-flip so MediaPipe coords land
    // on the on-screen body.
    expect((canvas as HTMLCanvasElement).style.transform).toContain('scaleX(-1)')
  })

  it('does not render the canvas overlay when not recording', () => {
    captureIsRecording = false
    render(<LiveCapturePanel />)
    expect(screen.queryByTestId('pose-overlay-canvas')).toBeNull()
  })

  it('renders the green "Tracking" pill on good quality', () => {
    captureIsRecording = true
    render(<LiveCapturePanel />)
    act(() => {
      lastCaptureOpts?.onPoseQuality?.('good')
    })
    const pill = screen.getByTestId('tracking-quality-pill')
    expect(pill).toBeInTheDocument()
    expect(pill).toHaveAttribute('data-quality', 'good')
    expect(pill.textContent).toContain('Tracking')
    // Green dot.
    expect(pill.innerHTML).toContain('bg-emerald-400')
  })

  it('renders the amber "Step back" pill on weak quality', () => {
    captureIsRecording = true
    render(<LiveCapturePanel />)
    act(() => {
      lastCaptureOpts?.onPoseQuality?.('weak')
    })
    const pill = screen.getByTestId('tracking-quality-pill')
    expect(pill).toHaveAttribute('data-quality', 'weak')
    expect(pill.textContent).toContain('Step back so your full body is in frame')
    expect(pill.innerHTML).toContain('bg-amber-400')
  })

  it('renders the red "Move into frame" pill on no-body', () => {
    captureIsRecording = true
    render(<LiveCapturePanel />)
    act(() => {
      lastCaptureOpts?.onPoseQuality?.('no-body')
    })
    const pill = screen.getByTestId('tracking-quality-pill')
    expect(pill).toHaveAttribute('data-quality', 'no-body')
    expect(pill.textContent).toContain('Move into frame')
    expect(pill.innerHTML).toContain('bg-red-400')
  })

  it('does not render the pill before any onPoseQuality has fired', () => {
    captureIsRecording = true
    render(<LiveCapturePanel />)
    expect(screen.queryByTestId('tracking-quality-pill')).toBeNull()
  })

  it('triggers the swing-counter pulse when onSwing fires', () => {
    captureIsRecording = true
    render(<LiveCapturePanel />)

    const before = screen.getByTestId('swing-counter-pulse')
    expect(before.getAttribute('data-pulse-tick')).toBe('0')

    // Synthetically fire onSwing the way useLiveCapture would.
    act(() => {
      lastCaptureOpts?.onSwing?.(fakeSwing(0))
    })

    const after = screen.getByTestId('swing-counter-pulse')
    expect(after.getAttribute('data-pulse-tick')).toBe('1')
    // Visible scale bump class signals the CSS keyframe is ready to play.
    expect(after.className).toContain('scale-110')

    // Each subsequent swing increments the tick so the keyed remount
    // re-runs the transition.
    act(() => {
      lastCaptureOpts?.onSwing?.(fakeSwing(1))
    })
    expect(screen.getByTestId('swing-counter-pulse').getAttribute('data-pulse-tick')).toBe('2')
  })

  it('renders the human-readable status label, not the raw store status', () => {
    captureIsRecording = true
    useLiveStore.setState({ status: 'recording' })
    render(<LiveCapturePanel />)

    const status = screen.getByTestId('status-label')
    expect(status.textContent).toBe('Coaching live')

    // Switch the store status — re-render happens through useLiveStore.
    act(() => {
      useLiveStore.setState({ status: 'uploading' })
    })
    expect(screen.getByTestId('status-label').textContent).toBe('Saving session')

    act(() => {
      useLiveStore.setState({ status: 'complete' })
    })
    expect(screen.getByTestId('status-label').textContent).toBe('Done')

    act(() => {
      useLiveStore.setState({ status: 'idle' })
    })
    expect(screen.getByTestId('status-label').textContent).toBe('Ready')
  })

  it('passes a reduced alpha scale to renderPose during the swing-detected pulse', async () => {
    captureIsRecording = true
    render(<LiveCapturePanel />)

    // Stuff a frame into the pose-frame ref via the captured callback —
    // otherwise the rAF draw loop has nothing to render and renderPose
    // is never invoked.
    const fakeFrame = { frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }
    act(() => {
      lastCaptureOpts?.onPoseFrame?.(fakeFrame)
      lastCaptureOpts?.onSwing?.(fakeSwing(0))
    })

    // Pump rAF a few times so the loop draws at least once. jsdom has
    // requestAnimationFrame stubbed to ~16ms; vi.useFakeTimers() would
    // also work but we keep this test out of fake-timer territory to
    // avoid interfering with the rAF loop's own timestamping.
    await new Promise((r) => setTimeout(r, 50))

    expect(renderPoseSpy).toHaveBeenCalled()
    const lastCall = renderPoseSpy.mock.calls.at(-1)!
    // Args: ctx, frame, w, h, options
    const opts = lastCall[4] as { scale?: number }
    expect(opts.scale).toBeLessThan(1)
    expect(opts.scale).toBeGreaterThanOrEqual(0.4)
  })
})
