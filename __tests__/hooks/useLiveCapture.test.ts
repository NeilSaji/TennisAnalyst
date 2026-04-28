/**
 * Phase 3 — Visibility worker tests for useLiveCapture, updated for
 * Phase 5D's MediaPipe → onnxruntime-web migration. The test surface is
 * unchanged because the gate logic above the detector is unchanged; only
 * the mock plumbing moved from `@/lib/mediapipe` to `@/lib/browserPose`.
 *
 * Two behaviors guarded here:
 *   1. onPoseFrame fires for every detection-tick that survives the
 *      strict body-presence gate (and NOT for face-only / no-pose ticks).
 *   2. onPoseQuality emits transitions only — good → weak → no-body —
 *      not one event per frame.
 *
 * To keep the test deterministic without spinning up a real ONNX runtime
 * + getUserMedia + MediaRecorder stack, we mock the pose detector,
 * camera, and recorder layers and drive the loop through a controllable
 * fake `setInterval` (the fallback path that the hook uses when
 * requestVideoFrameCallback is unavailable, which is the case in jsdom).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { LANDMARK_INDICES } from '@/lib/jointAngles'
import type { Landmark } from '@/lib/supabase'

// --- Mocks ------------------------------------------------------------------

// browserPose.detect returns Landmark[] | null directly. Tests queue up
// per-tick return values via this stack; once empty, falls back to
// returning null (no-body).
const detectQueue: Array<Landmark[] | null> = []
const detectMock = vi.fn(async () => {
  if (detectQueue.length === 0) return null
  return detectQueue.shift() ?? null
})
const disposeMock = vi.fn()
type CreatePoseDetectorOpts = {
  onProgress?: (
    loaded: number,
    total: number,
    label: 'yolo' | 'rtmpose',
  ) => void
}
const createPoseDetectorMock = vi.fn(async (_opts?: CreatePoseDetectorOpts) => ({
  detect: detectMock,
  dispose: disposeMock,
}))

vi.mock('@/lib/browserPose', () => ({
  createPoseDetector: (opts?: CreatePoseDetectorOpts) =>
    createPoseDetectorMock(opts),
}))

// LiveSwingDetector: we don't want a real one churning on synthetic
// frames in this test (it has its own dedicated suite). A no-op feed
// is enough to verify onPoseFrame plumbing.
vi.mock('@/lib/liveSwingDetector', () => ({
  LiveSwingDetector: class {
    feed() {
      return null
    }
  },
}))

// poseSmoothing: keep the real isFrameConfident / isBodyVisible behavior
// so transitions are honest. smoothFrames isn't exercised in these tests.

import { useLiveCapture, type PoseQuality } from '@/hooks/useLiveCapture'
import type { PoseFrame } from '@/lib/supabase'

// --- Helpers ----------------------------------------------------------------

function withIds(
  raw: Array<{ x: number; y: number; z?: number; visibility?: number }>,
): Landmark[] {
  // Coerce a flat raw landmark list into the 33-entry BlazePose Landmark[]
  // shape browserPose returns. Same x/y/visibility values; just the
  // explicit `id` and `name` fields the rest of the app expects.
  return raw.map((lm, id) => ({
    id,
    name: `landmark_${id}`,
    x: lm.x,
    y: lm.y,
    z: lm.z ?? 0,
    visibility: lm.visibility ?? 1,
  }))
}

function fullBodyLandmarks(): Landmark[] {
  // Constructs a head-to-ankle pose with all required landmarks at
  // visibility 0.95 (passes both isFrameConfident and isBodyVisible).
  // Must include ankles for vertical-extent-check >= 0.35.
  return withIds([
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 0 nose
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 1 left eye inner
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 2 left eye
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 3 left eye outer
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 4 right eye inner
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 5 right eye
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 6 right eye outer
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 7 left ear
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 8 right ear
    { x: 0.5, y: 0.12, visibility: 0.9 }, // 9 mouth left
    { x: 0.5, y: 0.12, visibility: 0.9 }, // 10 mouth right
    { x: 0.55, y: 0.25, visibility: 0.95 }, // 11 left shoulder
    { x: 0.45, y: 0.25, visibility: 0.95 }, // 12 right shoulder
    { x: 0.6, y: 0.4, visibility: 0.95 }, // 13 left elbow
    { x: 0.4, y: 0.4, visibility: 0.95 }, // 14 right elbow
    { x: 0.6, y: 0.55, visibility: 0.95 }, // 15 left wrist
    { x: 0.4, y: 0.55, visibility: 0.95 }, // 16 right wrist
    { x: 0.6, y: 0.55, visibility: 0.5 }, // 17
    { x: 0.4, y: 0.55, visibility: 0.5 }, // 18
    { x: 0.6, y: 0.55, visibility: 0.5 }, // 19
    { x: 0.4, y: 0.55, visibility: 0.5 }, // 20
    { x: 0.6, y: 0.55, visibility: 0.5 }, // 21
    { x: 0.4, y: 0.55, visibility: 0.5 }, // 22
    { x: 0.53, y: 0.6, visibility: 0.95 }, // 23 left hip
    { x: 0.47, y: 0.6, visibility: 0.95 }, // 24 right hip
    { x: 0.54, y: 0.75, visibility: 0.9 }, // 25 left knee
    { x: 0.46, y: 0.75, visibility: 0.9 }, // 26 right knee
    { x: 0.54, y: 0.9, visibility: 0.9 }, // 27 left ankle
    { x: 0.46, y: 0.9, visibility: 0.9 }, // 28 right ankle
    { x: 0.54, y: 0.95, visibility: 0.9 }, // 29 left heel
    { x: 0.46, y: 0.95, visibility: 0.9 }, // 30 right heel
    { x: 0.54, y: 0.95, visibility: 0.9 }, // 31 left foot index
    { x: 0.46, y: 0.95, visibility: 0.9 }, // 32 right foot index
  ])
}

function faceOnlyLandmarks(): Landmark[] {
  // Passes isFrameConfident (avg visibility >= 0.4 + non-degenerate
  // bbox) but FAILS isBodyVisible (wrists at 0.3 < 0.5 cutoff). The
  // upper-body landmarks read confidently but the racket-arm wrists
  // do not — that's the "head and shoulders only" / "wrists below
  // frame" failure mode the gate is designed to reject.
  return withIds([
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 0 nose
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 1
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 2
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 3
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 4
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 5
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 6
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 7
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 8
    { x: 0.5, y: 0.12, visibility: 0.95 }, // 9
    { x: 0.5, y: 0.12, visibility: 0.95 }, // 10
    { x: 0.55, y: 0.25, visibility: 0.95 }, // 11 left shoulder
    { x: 0.45, y: 0.25, visibility: 0.95 }, // 12 right shoulder
    { x: 0.6, y: 0.4, visibility: 0.6 }, // 13 elbows OK
    { x: 0.4, y: 0.4, visibility: 0.6 }, // 14
    { x: 0.6, y: 0.55, visibility: 0.3 }, // 15 wrists below cutoff
    { x: 0.4, y: 0.55, visibility: 0.3 }, // 16
    { x: 0.6, y: 0.55, visibility: 0.3 }, // 17
    { x: 0.4, y: 0.55, visibility: 0.3 }, // 18
    { x: 0.6, y: 0.55, visibility: 0.3 }, // 19
    { x: 0.4, y: 0.55, visibility: 0.3 }, // 20
    { x: 0.6, y: 0.55, visibility: 0.3 }, // 21
    { x: 0.4, y: 0.55, visibility: 0.3 }, // 22
    { x: 0.53, y: 0.6, visibility: 0.95 }, // 23 left hip (high vis but...)
    { x: 0.47, y: 0.6, visibility: 0.95 }, // 24 right hip
    { x: 0.54, y: 0.75, visibility: 0.3 }, // 25 knees low
    { x: 0.46, y: 0.75, visibility: 0.3 }, // 26
    { x: 0.54, y: 0.9, visibility: 0.2 }, // 27 ankles barely visible
    { x: 0.46, y: 0.9, visibility: 0.2 }, // 28
    { x: 0.54, y: 0.95, visibility: 0.2 }, // 29
    { x: 0.46, y: 0.95, visibility: 0.2 }, // 30
    { x: 0.54, y: 0.95, visibility: 0.2 }, // 31
    { x: 0.46, y: 0.95, visibility: 0.2 }, // 32
  ])
}

// --- Browser-API mocks ------------------------------------------------------

// MediaRecorder spy that exposes its onstop hook so the hook's stop()
// path can resolve the recorder.onstop promise. Not exercised in
// these tests beyond start(), but the hook constructs one immediately.
class FakeMediaRecorder {
  state: 'inactive' | 'recording' = 'inactive'
  mimeType: string
  ondataavailable: ((ev: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  constructor(_stream: unknown, opts?: { mimeType?: string }) {
    this.mimeType = opts?.mimeType ?? 'video/webm'
  }
  start() {
    this.state = 'recording'
  }
  stop() {
    this.state = 'inactive'
    this.onstop?.()
  }
  static isTypeSupported(_mime: string) {
    return true
  }
}

function installBrowserMocks() {
  ;(globalThis as unknown as { MediaRecorder: typeof FakeMediaRecorder }).MediaRecorder =
    FakeMediaRecorder
  ;(navigator.mediaDevices as unknown as {
    getUserMedia: (c: MediaStreamConstraints) => Promise<MediaStream>
  }) = {
    getUserMedia: vi.fn(
      async () =>
        ({
          getTracks: () => [{ stop: vi.fn() }],
        }) as unknown as MediaStream,
    ),
  }

  // jsdom doesn't implement getContext('2d') and the hook needs one
  // for its offscreen sampling canvas. Patch HTMLCanvasElement to
  // return a no-op context — useLiveCapture only calls drawImage and
  // never reads pixels.
  const noopCtx = {
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
  } as unknown as CanvasRenderingContext2D
  HTMLCanvasElement.prototype.getContext = (function () {
    return noopCtx
  }) as unknown as HTMLCanvasElement['getContext']
}

function makeVideoEl(): HTMLVideoElement {
  // jsdom doesn't implement video playback. Stub the surface useLiveCapture
  // touches so start() can complete: play(), srcObject setter (tolerant),
  // videoWidth/Height (query-time access).
  const el = document.createElement('video')
  Object.defineProperty(el, 'play', {
    value: () => Promise.resolve(),
    writable: true,
  })
  Object.defineProperty(el, 'videoWidth', { value: 640, writable: true, configurable: true })
  Object.defineProperty(el, 'videoHeight', { value: 360, writable: true, configurable: true })
  // requestVideoFrameCallback is intentionally omitted so the hook
  // falls back to setInterval — that's the path we drive in tests
  // because it's pumpable with vi.useFakeTimers().
  return el
}

// detect() is async, so each fake-timer tick schedules a microtask the
// `await` is waiting on. Drain microtasks twice (drawImage → detect →
// continuation → emit) so onPoseFrame/onPoseQuality have run before we
// assert.
async function flushMicrotasks() {
  for (let i = 0; i < 5; i++) {
    await Promise.resolve()
  }
}

// --- Tests ------------------------------------------------------------------

describe('useLiveCapture — onPoseFrame + onPoseQuality', () => {
  beforeEach(() => {
    detectMock.mockClear()
    disposeMock.mockClear()
    createPoseDetectorMock.mockClear()
    detectQueue.length = 0
    installBrowserMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('fires onPoseFrame for every detection tick that passes isBodyVisible', async () => {
    const onPoseFrame = vi.fn<(frame: PoseFrame) => void>()
    const onPoseQuality = vi.fn<(q: PoseQuality) => void>()

    // Three good frames in a row.
    detectQueue.push(fullBodyLandmarks(), fullBodyLandmarks(), fullBodyLandmarks())

    const { result } = renderHook(() =>
      useLiveCapture({ onPoseFrame, onPoseQuality, targetDetectionFps: 15 }),
    )

    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })

    // Pump the fallback setInterval (1000/15 ≈ 67ms) past three ticks.
    // detect() is async so we drain microtasks in between to let the
    // continuation run before the next interval tick.
    for (let i = 0; i < 6; i++) {
      await act(async () => {
        vi.advanceTimersByTime(70)
        await flushMicrotasks()
      })
    }

    expect(onPoseFrame).toHaveBeenCalledTimes(3)
    // Frames are well-formed PoseFrame objects with the expected wrist
    // landmark mapped.
    const firstFrame = onPoseFrame.mock.calls[0][0]
    expect(firstFrame.landmarks).toBeDefined()
    expect(
      firstFrame.landmarks.find((lm) => lm.id === LANDMARK_INDICES.RIGHT_WRIST),
    ).toBeTruthy()
    // First quality emission must be 'good'.
    expect(onPoseQuality.mock.calls[0][0]).toBe('good')
    // No 'weak' or 'no-body' transitions across three good frames in a row.
    const qualities = onPoseQuality.mock.calls.map((c) => c[0])
    expect(qualities.filter((q) => q !== 'good')).toHaveLength(0)
  })

  it('does NOT fire onPoseFrame for face-only frames (weak quality)', async () => {
    const onPoseFrame = vi.fn<(frame: PoseFrame) => void>()
    const onPoseQuality = vi.fn<(q: PoseQuality) => void>()

    detectQueue.push(faceOnlyLandmarks(), faceOnlyLandmarks())

    const { result } = renderHook(() =>
      useLiveCapture({ onPoseFrame, onPoseQuality, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()

    await act(async () => {
      await result.current.start(video)
    })
    for (let i = 0; i < 4; i++) {
      await act(async () => {
        vi.advanceTimersByTime(70)
        await flushMicrotasks()
      })
    }

    // Face-only frames are gated out before onPoseFrame, but they
    // still trigger a 'weak' quality transition.
    expect(onPoseFrame).not.toHaveBeenCalled()
    expect(onPoseQuality.mock.calls.some((c) => c[0] === 'weak')).toBe(true)
  })

  it('emits onPoseQuality only on transitions, not per frame', async () => {
    const onPoseQuality = vi.fn<(q: PoseQuality) => void>()

    // Five good frames back-to-back. The first should fire 'good'; the
    // remaining four should NOT re-fire 'good' (no transition).
    for (let i = 0; i < 5; i++) {
      detectQueue.push(fullBodyLandmarks())
    }

    const { result } = renderHook(() =>
      useLiveCapture({ onPoseQuality, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()

    await act(async () => {
      await result.current.start(video)
    })
    for (let i = 0; i < 8; i++) {
      await act(async () => {
        vi.advanceTimersByTime(70)
        await flushMicrotasks()
      })
    }

    const goodEmissions = onPoseQuality.mock.calls.filter((c) => c[0] === 'good')
    expect(goodEmissions).toHaveLength(1)
  })

  it('transitions good → weak → no-body across the right fixtures', async () => {
    const onPoseQuality = vi.fn<(q: PoseQuality) => void>()

    // 1 good, 1 weak (face-only), then several "no detection" returns
    // long enough that the no-body 1s timeout fires. Once the queue is
    // drained the mock falls back to returning null (no person).
    detectQueue.push(fullBodyLandmarks())
    detectQueue.push(faceOnlyLandmarks())

    const { result } = renderHook(() =>
      useLiveCapture({ onPoseQuality, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })

    // First two ticks land good then weak.
    for (let i = 0; i < 4; i++) {
      await act(async () => {
        vi.advanceTimersByTime(70)
        await flushMicrotasks()
      })
    }
    const earlyQualities = onPoseQuality.mock.calls.map((c) => c[0])
    expect(earlyQualities).toContain('good')
    expect(earlyQualities.indexOf('good')).toBeLessThan(earlyQualities.indexOf('weak'))

    // Now sit through >1s of "no landmarks" frames (queue is drained)
    // so the no-body timeout triggers.
    for (let i = 0; i < 25; i++) {
      await act(async () => {
        vi.advanceTimersByTime(70)
        await flushMicrotasks()
      })
    }
    const finalQuality = onPoseQuality.mock.calls.at(-1)?.[0]
    expect(finalQuality).toBe('no-body')
  })

  it('forwards onModelLoadProgress to createPoseDetector', async () => {
    const onModelLoadProgress = vi.fn()

    detectQueue.push(fullBodyLandmarks())

    const { result } = renderHook(() =>
      useLiveCapture({ onModelLoadProgress, targetDetectionFps: 15 }),
    )

    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })

    expect(createPoseDetectorMock).toHaveBeenCalledTimes(1)
    const opts = createPoseDetectorMock.mock.calls[0][0]
    expect(opts?.onProgress).toBe(onModelLoadProgress)
  })
})
