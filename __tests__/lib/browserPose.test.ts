/**
 * Phase 5B — Browser ONNX inference module tests.
 *
 * Mocks `onnxruntime-web` so the test suite stays fast (no ~50 MB model
 * download, no native WASM/WebGPU init). Real model behavior is verified
 * by an opt-in integration test in Worker D's milestone, not here. What
 * we cover here:
 *
 *   1. Letterbox math (preprocessing).
 *   2. NMS reduces overlapping boxes.
 *   3. Person picker picks highest score.
 *   4. detect() returns null when no person is found.
 *   5. SimCC decoding produces correct keypoint coords.
 *   6. Coord transform: crop pixel → original frame pixel.
 *   7. End-to-end pipeline with mocked sessions.
 *   8. Execution-provider fallback.
 *   9. dispose() releases sessions.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  computeLetterboxTransform,
  decodeSimCC,
  decodeYoloPersonDetections,
  expandBox,
  imageDataToNCHWImageNet,
  imageDataToNCHWNormalized,
  mapBoxFromLetterbox,
  mapKeypointToFrame,
  nonMaxSuppression,
  pickHighestPerson,
  type PersonDetection,
} from '@/lib/browserPose'

// ---------------------------------------------------------------------------
// onnxruntime-web mock — must be registered before importing browserPose's
// async-detector path. We expose hooks so individual tests can swap the
// session.run() return values per test.
// ---------------------------------------------------------------------------

interface MockSession {
  inputNames: string[]
  outputNames: string[]
  outputMetadata: Array<{
    name: string
    isTensor: true
    type: 'float32'
    shape: ReadonlyArray<number | string>
  }>
  inputMetadata: Array<{ name: string; isTensor: true; type: 'float32'; shape: number[] }>
  run: ReturnType<typeof vi.fn>
  release: ReturnType<typeof vi.fn>
}

interface MockState {
  // Per-call create() behavior. Tests push entries; create() consumes them.
  // If empty, create() returns a default session.
  createPlan: Array<
    | { kind: 'throw'; error: Error }
    | { kind: 'return'; session: MockSession }
  >
  // The most recent successfully created sessions (in creation order).
  createdSessions: MockSession[]
  createCalls: Array<{ ep: string }>
}

// Module-scoped state so vi.mock factory + tests both see it.
const mockState: MockState = {
  createPlan: [],
  createdSessions: [],
  createCalls: [],
}

// vi.mock is hoisted to the top of the file before all imports, so the
// factory cannot close over module-level test fixtures. We hang the
// mutable mock state off `globalThis` (which IS available at hoist time
// via lazy access) and define a Tensor stub inline.
vi.mock('onnxruntime-web', () => {
  class MockTensor {
    constructor(
      public type: string,
      public data: Float32Array | Uint8Array | Int32Array,
      public dims: readonly number[],
    ) {}
  }
  const getState = (): MockState => {
    const g = globalThis as Record<string, unknown>
    return g.__browserPoseMockState as MockState
  }
  return {
    Tensor: MockTensor,
    InferenceSession: {
      create: vi.fn(async (_buf: unknown, opts: { executionProviders?: string[] }) => {
        const state = getState()
        const ep = (opts?.executionProviders ?? ['?'])[0] ?? '?'
        state.createCalls.push({ ep })
        const planEntry = state.createPlan.shift()
        if (planEntry) {
          if (planEntry.kind === 'throw') throw planEntry.error
          state.createdSessions.push(planEntry.session)
          return planEntry.session as unknown
        }
        const sess = makeDefaultSession()
        state.createdSessions.push(sess)
        return sess as unknown
      }),
    },
  }
})

// Bootstrap the mock state on globalThis so the (hoisted) vi.mock factory
// can reach it. We swap fields per-test below; we never reassign the
// outer object reference.
;(globalThis as Record<string, unknown>).__browserPoseMockState = mockState

// Mock modelLoader so loadModel() doesn't try to fetch /models/*.
vi.mock('@/lib/modelLoader', () => ({
  loadModel: vi.fn(async (_url: string, _opts?: unknown) => new Uint8Array([0])),
}))

function makeDefaultSession(): MockSession {
  return {
    inputNames: ['images'],
    outputNames: ['output0'],
    outputMetadata: [
      {
        name: 'output0',
        isTensor: true,
        type: 'float32',
        shape: [1, 84, 8400],
      },
    ],
    inputMetadata: [
      { name: 'images', isTensor: true, type: 'float32', shape: [1, 3, 640, 640] },
    ],
    run: vi.fn(),
    release: vi.fn(),
  }
}

function makeYoloSession(runImpl: () => Record<string, { data: Float32Array; dims: number[] }>): MockSession {
  const s = makeDefaultSession()
  s.run = vi.fn(async () => runImpl())
  return s
}

function makeRtmposeSession(
  outX: Float32Array,
  outY: Float32Array,
  xBins = 512,
  yBins = 384,
): MockSession {
  return {
    inputNames: ['input'],
    outputNames: ['simcc_x', 'simcc_y'],
    outputMetadata: [
      { name: 'simcc_x', isTensor: true, type: 'float32', shape: [1, 17, xBins] },
      { name: 'simcc_y', isTensor: true, type: 'float32', shape: [1, 17, yBins] },
    ],
    inputMetadata: [
      { name: 'input', isTensor: true, type: 'float32', shape: [1, 3, 192, 256] },
    ],
    run: vi.fn(async () => ({
      simcc_x: { data: outX, dims: [1, 17, xBins] },
      simcc_y: { data: outY, dims: [1, 17, yBins] },
    })),
    release: vi.fn(),
  }
}

// ---------------------------------------------------------------------------
// jsdom doesn't ship OffscreenCanvas. The detector falls back to
// document.createElement('canvas'); we polyfill enough of the 2d context
// surface that browserPose actually reads/writes.
// ---------------------------------------------------------------------------

class FakeImageData {
  width: number
  height: number
  data: Uint8ClampedArray
  constructor(w: number, h: number, fillRgb: [number, number, number] = [0, 0, 0]) {
    this.width = w
    this.height = h
    this.data = new Uint8ClampedArray(w * h * 4)
    const [r, g, b] = fillRgb
    for (let i = 0; i < w * h; i++) {
      this.data[i * 4] = r
      this.data[i * 4 + 1] = g
      this.data[i * 4 + 2] = b
      this.data[i * 4 + 3] = 255
    }
  }
}

beforeEach(() => {
  mockState.createPlan = []
  mockState.createdSessions = []
  mockState.createCalls = []

  // Stub HTMLCanvasElement.getContext (jsdom returns null by default for 2d).
  // Our context just records draws and returns a flat-fill ImageData on
  // getImageData(); detect() inputs are model-mocked anyway, so the pixel
  // contents don't have to be physically meaningful.
  const ctxStub = {
    drawImage: vi.fn(),
    fillRect: vi.fn(),
    set fillStyle(_s: string) {},
    get fillStyle() {
      return ''
    },
    getImageData: vi.fn((_x: number, _y: number, w: number, h: number) =>
      new FakeImageData(w, h),
    ),
  }
  // jsdom's prototype get/set chain is finicky; assign on the prototype
  // for both HTMLCanvasElement and OffscreenCanvas (latter is rarely
  // available but defensive).
  Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
    configurable: true,
    writable: true,
    value: () => ctxStub,
  })
  // Pretend OffscreenCanvas does not exist so the detector takes the
  // HTMLCanvasElement path uniformly.
  ;(globalThis as Record<string, unknown>).OffscreenCanvas = undefined
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ===========================================================================
// 1) Letterbox math.
// ===========================================================================

describe('computeLetterboxTransform', () => {
  it('1280x720 -> 640x640 produces scale=0.5, padX=0, padY=140', () => {
    // Aspect-preserving letterbox into a square. Scale picks the
    // tighter axis: min(640/1280=0.5, 640/720=0.889) = 0.5. New image
    // is 640x360; remaining 280px split equally → padY=140 each side.
    //
    // (The Phase 5B spec said "padY=80" but that's wrong for a square
    // letterbox of a 16:9 input; padY=80 would imply newH=480, which
    // requires scale=0.667, which would push newW to 853 > 640.)
    const tx = computeLetterboxTransform(1280, 720, 640, 640)
    expect(tx.scale).toBe(0.5)
    expect(tx.padX).toBe(0)
    expect(tx.padY).toBe(140)
    expect(tx.outW).toBe(640)
    expect(tx.outH).toBe(640)
    expect(tx.srcW).toBe(1280)
    expect(tx.srcH).toBe(720)
  })

  it('720x1280 -> 640x640 produces scale=0.5, padX=180, padY=0', () => {
    const tx = computeLetterboxTransform(720, 1280, 640, 640)
    // min(640/720, 640/1280) = 640/1280 = 0.5
    expect(tx.scale).toBe(0.5)
    // newW = 360, padX = (640-360)/2 = 140
    expect(tx.padX).toBe(140)
    expect(tx.padY).toBe(0)
  })

  it('square 600x600 -> 640x640 scales up to fit, pads 0 on both axes', () => {
    const tx = computeLetterboxTransform(600, 600, 640, 640)
    // min(640/600, 640/600) = 1.0666... — same on both axes; no padding.
    expect(tx.scale).toBeCloseTo(640 / 600, 6)
    expect(tx.padX).toBe(0)
    expect(tx.padY).toBe(0)
  })

  it('crop 200x200 -> 256x192 picks scale=0.96 (limited by H), padX=4, padY=0', () => {
    const tx = computeLetterboxTransform(200, 200, 256, 192)
    // min(256/200=1.28, 192/200=0.96) = 0.96
    expect(tx.scale).toBeCloseTo(0.96, 6)
    // newW = 200*0.96 = 192; padX = (256 - 192)/2 = 32
    expect(tx.padX).toBe(32)
    // newH = 200*0.96 = 192; padY = 0
    expect(tx.padY).toBe(0)
  })
})

// ===========================================================================
// 2) NMS.
// ===========================================================================

describe('nonMaxSuppression', () => {
  it('reduces 5 dets (3 overlapping at 100,100, 2 distinct) to 3 boxes', () => {
    // Three highly overlapping boxes around (100,100):
    const overlap1: PersonDetection = { x1: 100, y1: 100, x2: 200, y2: 200, score: 0.9 }
    const overlap2: PersonDetection = { x1: 105, y1: 105, x2: 205, y2: 205, score: 0.85 }
    const overlap3: PersonDetection = { x1: 95, y1: 95, x2: 195, y2: 195, score: 0.8 }
    // Two distinct positions:
    const distinct1: PersonDetection = { x1: 400, y1: 400, x2: 500, y2: 500, score: 0.7 }
    const distinct2: PersonDetection = { x1: 800, y1: 200, x2: 900, y2: 300, score: 0.6 }
    const out = nonMaxSuppression(
      [overlap1, overlap2, overlap3, distinct1, distinct2],
      0.45,
    )
    expect(out).toHaveLength(3)
    // Top from cluster + the two distinct boxes survive.
    expect(out[0]).toEqual(overlap1)
    expect(out.map((b) => b.score).sort()).toEqual([0.6, 0.7, 0.9])
  })

  it('keeps all when no boxes overlap', () => {
    const a: PersonDetection = { x1: 0, y1: 0, x2: 100, y2: 100, score: 0.9 }
    const b: PersonDetection = { x1: 200, y1: 0, x2: 300, y2: 100, score: 0.8 }
    const c: PersonDetection = { x1: 400, y1: 0, x2: 500, y2: 100, score: 0.7 }
    expect(nonMaxSuppression([a, b, c], 0.45)).toEqual([a, b, c])
  })

  it('drops everything but top when all boxes are identical', () => {
    const a: PersonDetection = { x1: 0, y1: 0, x2: 100, y2: 100, score: 0.9 }
    const b: PersonDetection = { x1: 0, y1: 0, x2: 100, y2: 100, score: 0.5 }
    const c: PersonDetection = { x1: 0, y1: 0, x2: 100, y2: 100, score: 0.3 }
    expect(nonMaxSuppression([a, b, c], 0.45)).toEqual([a])
  })
})

// ===========================================================================
// 3) Person picker.
// ===========================================================================

describe('pickHighestPerson', () => {
  it('returns the box with the highest score', () => {
    const dets: PersonDetection[] = [
      { x1: 0, y1: 0, x2: 1, y2: 1, score: 0.4 },
      { x1: 0, y1: 0, x2: 1, y2: 1, score: 0.95 },
      { x1: 0, y1: 0, x2: 1, y2: 1, score: 0.7 },
    ]
    expect(pickHighestPerson(dets)?.score).toBe(0.95)
  })

  it('returns null on empty', () => {
    expect(pickHighestPerson([])).toBeNull()
  })
})

// ===========================================================================
// 4) decodeYoloPersonDetections — null when nothing meets threshold.
// ===========================================================================

describe('decodeYoloPersonDetections', () => {
  it('returns [] when all class-0 scores fall below threshold', () => {
    // Build a tensor with 4 anchors, 80 classes, all near-zero.
    const numAnchors = 4
    const numClasses = 80
    const out = new Float32Array((4 + numClasses) * numAnchors)
    // Set tiny class-0 scores so even the max is < 0.3.
    for (let a = 0; a < numAnchors; a++) {
      out[(4 + 0) * numAnchors + a] = 0.05
    }
    const dets = decodeYoloPersonDetections(out, numAnchors, numClasses, 0, 0.3)
    expect(dets).toEqual([])
  })

  it('keeps a high-confidence person detection and yields xyxy in letterbox space', () => {
    const numAnchors = 2
    const numClasses = 80
    const out = new Float32Array((4 + numClasses) * numAnchors)
    // Anchor 0: cx=320, cy=320, w=100, h=200, person score=0.95
    out[0 * numAnchors + 0] = 320
    out[1 * numAnchors + 0] = 320
    out[2 * numAnchors + 0] = 100
    out[3 * numAnchors + 0] = 200
    out[(4 + 0) * numAnchors + 0] = 0.95
    // Anchor 1: random non-person.
    out[(4 + 0) * numAnchors + 1] = 0.05
    out[(4 + 5) * numAnchors + 1] = 0.4 // bus class above thresh but wrong class
    const dets = decodeYoloPersonDetections(out, numAnchors, numClasses, 0, 0.3)
    expect(dets).toHaveLength(1)
    // Float32 round-trip turns 0.95 into ~0.94999998, so we deep-test
    // the bbox edges and check score with a tolerance.
    expect(dets[0].x1).toBe(270) // 320 - 50
    expect(dets[0].y1).toBe(220) // 320 - 100
    expect(dets[0].x2).toBe(370)
    expect(dets[0].y2).toBe(420)
    expect(dets[0].score).toBeCloseTo(0.95, 4)
  })

  it('skips anchors whose argmax class is not the target class', () => {
    const numAnchors = 1
    const numClasses = 80
    const out = new Float32Array((4 + numClasses) * numAnchors)
    out[0] = 100
    out[1] = 100
    out[2] = 50
    out[3] = 50
    // person score = 0.4, but bus score = 0.6 — argmax class is bus.
    out[(4 + 0) * numAnchors + 0] = 0.4
    out[(4 + 5) * numAnchors + 0] = 0.6
    const dets = decodeYoloPersonDetections(out, numAnchors, numClasses, 0, 0.3)
    expect(dets).toEqual([])
  })
})

// ===========================================================================
// 5) SimCC decoding.
// ===========================================================================

describe('decodeSimCC', () => {
  it('produces correct keypoint coords given known peaks', () => {
    const xBins = 512
    const yBins = 384
    const numKpts = 17
    const split = 2.0
    const simccX = new Float32Array(numKpts * xBins)
    const simccY = new Float32Array(numKpts * yBins)
    // Three keypoints with hand-set peaks; remaining default to zero argmax=0.
    // kpt 0: peak x=200, y=100, score derived from peak heights.
    simccX[0 * xBins + 200] = 7.5
    simccY[0 * yBins + 100] = 8.0
    // kpt 5: peak x=500 (right edge), y=383 (last bin).
    simccX[5 * xBins + 500] = 4.0
    simccY[5 * yBins + 383] = 6.0
    // kpt 10: peak x=0 (left edge), y=0.
    simccX[10 * xBins + 0] = 0.5
    simccY[10 * yBins + 0] = 0.5

    const out = decodeSimCC(simccX, simccY, numKpts, xBins, yBins, split)
    expect(out).toHaveLength(numKpts)
    expect(out[0].x).toBe(100) // 200/2
    expect(out[0].y).toBe(50) // 100/2
    expect(out[0].score).toBe(7.5) // min(7.5, 8.0)
    expect(out[5].x).toBe(250) // 500/2
    expect(out[5].y).toBeCloseTo(191.5, 6) // 383/2
    expect(out[5].score).toBe(4.0) // min(4, 6)
    expect(out[10].x).toBe(0)
    expect(out[10].y).toBe(0)
    expect(out[10].score).toBe(0.5)
  })
})

// ===========================================================================
// 6) Coord transform crop -> original frame.
// ===========================================================================

describe('mapKeypointToFrame', () => {
  it('maps (128,96) inside a 256x192 input back through a known crop', () => {
    // Crop bbox in original frame (50, 80) -> (250, 280) -> crop is 200x200.
    const cropX = 50
    const cropY = 80
    const cropW = 200
    const cropH = 200
    const tx = computeLetterboxTransform(cropW, cropH, 256, 192)
    // 200x200 -> 256x192: scale = min(1.28, 0.96) = 0.96; padX=(256-192)/2=32; padY=0.
    expect(tx.scale).toBeCloseTo(0.96, 6)
    expect(tx.padX).toBe(32)
    expect(tx.padY).toBe(0)
    // Center of input (128,96):
    //   crop_px_x = (128 - 32) / 0.96 = 100
    //   crop_px_y = (96 - 0)  / 0.96 = 100
    //   frame_px_x = 100 + 50 = 150
    //   frame_px_y = 100 + 80 = 180
    const mapped = mapKeypointToFrame({ x: 128, y: 96, score: 0.9 }, tx, cropX, cropY)
    expect(mapped.x).toBeCloseTo(150, 6)
    expect(mapped.y).toBeCloseTo(180, 6)
    expect(mapped.score).toBe(0.9)
  })
})

// ===========================================================================
// expandBox sanity (exercised inside detect, but we lean on it for the
// e2e expected-frame coords too).
// ===========================================================================

describe('expandBox', () => {
  it('expands by pct on each side, clipped to image', () => {
    const expanded = expandBox(
      { x1: 100, y1: 100, x2: 200, y2: 300, score: 0.9 },
      400,
      400,
      0.2,
    )
    // bw=100, bh=200; dx=20, dy=40
    expect(expanded.x1).toBe(80)
    expect(expanded.y1).toBe(60)
    expect(expanded.x2).toBe(220)
    expect(expanded.y2).toBe(340)
  })

  it('clips at image edges', () => {
    const expanded = expandBox(
      { x1: 5, y1: 5, x2: 95, y2: 95, score: 0.9 },
      100,
      100,
      0.2,
    )
    // bw=90, bh=90; dx=18, dy=18 -> would go to (-13, -13, 113, 113); clipped.
    expect(expanded.x1).toBe(0)
    expect(expanded.y1).toBe(0)
    expect(expanded.x2).toBe(99)
    expect(expanded.y2).toBe(99)
  })
})

// ===========================================================================
// mapBoxFromLetterbox sanity.
// ===========================================================================

describe('mapBoxFromLetterbox', () => {
  it('inverts a 1280x720 -> 640x640 letterbox correctly', () => {
    const tx = computeLetterboxTransform(1280, 720, 640, 640)
    // For 1280x720 -> 640x640: scale=0.5, padX=0, padY=140 (image band
    // sits at y=140..500 inside the square).
    // A box at letterbox (160, 200)-(480, 480):
    //   x1 = (160 - 0)/0.5 = 320; x2 = (480 - 0)/0.5 = 960
    //   y1 = (200 - 140)/0.5 = 120; y2 = (480 - 140)/0.5 = 680
    const box: PersonDetection = { x1: 160, y1: 200, x2: 480, y2: 480, score: 0.9 }
    const mapped = mapBoxFromLetterbox(box, tx)
    expect(mapped.x1).toBeCloseTo(320, 6)
    expect(mapped.x2).toBeCloseTo(960, 6)
    expect(mapped.y1).toBeCloseTo(120, 6)
    expect(mapped.y2).toBeCloseTo(680, 6)
  })
})

// ===========================================================================
// Image preprocessing helpers (sanity).
// ===========================================================================

describe('imageDataToNCHWNormalized', () => {
  it('produces NCHW float32 with /255 scaling', () => {
    const img = new FakeImageData(2, 2, [255, 128, 0]) as unknown as ImageData
    const out = imageDataToNCHWNormalized(img)
    // 3 channels * 4 pixels = 12.
    expect(out.length).toBe(12)
    // R plane (first 4) = 1.0
    for (let i = 0; i < 4; i++) expect(out[i]).toBeCloseTo(1, 4)
    // G plane (next 4) = 128/255
    for (let i = 4; i < 8; i++) expect(out[i]).toBeCloseTo(128 / 255, 4)
    // B plane (last 4) = 0
    for (let i = 8; i < 12; i++) expect(out[i]).toBeCloseTo(0, 4)
  })
})

describe('imageDataToNCHWImageNet', () => {
  it('applies (px/255 - mean) / std per channel', () => {
    const img = new FakeImageData(1, 1, [255, 128, 0]) as unknown as ImageData
    const out = imageDataToNCHWImageNet(img)
    // mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    expect(out[0]).toBeCloseTo((1 - 0.485) / 0.229, 4)
    expect(out[1]).toBeCloseTo((128 / 255 - 0.456) / 0.224, 4)
    expect(out[2]).toBeCloseTo((0 - 0.406) / 0.225, 4)
  })
})

// ===========================================================================
// 7) End-to-end pipeline with mocked sessions.
// 8) Execution-provider fallback.
// 9) dispose() releases sessions.
//
// These need to dynamically import the module *after* mocks are set up,
// which they already are at the top. We also drive the InferenceSession.create
// plan directly via mockState.
// ===========================================================================

// Build a YOLO output that produces exactly one strong person detection.
function buildYoloOutputWithOnePerson(
  anchorIdx: number,
  cx: number,
  cy: number,
  w: number,
  h: number,
  score: number,
  numAnchors = 8400,
  numClasses = 80,
): Float32Array {
  const out = new Float32Array((4 + numClasses) * numAnchors)
  out[0 * numAnchors + anchorIdx] = cx
  out[1 * numAnchors + anchorIdx] = cy
  out[2 * numAnchors + anchorIdx] = w
  out[3 * numAnchors + anchorIdx] = h
  out[(4 + 0) * numAnchors + anchorIdx] = score
  return out
}

function buildSimccWithPeak(
  numKpts: number,
  bins: number,
  perKptPeakIdx: number[],
  peakValue = 5.0,
): Float32Array {
  const out = new Float32Array(numKpts * bins)
  for (let k = 0; k < numKpts; k++) {
    out[k * bins + perKptPeakIdx[k]] = peakValue
  }
  return out
}

// Make a fake video element that satisfies the detector's source check.
function makeFakeVideo(w: number, h: number): HTMLVideoElement {
  // Construct a real <video> so the `instanceof HTMLVideoElement` branch
  // fires; override the videoWidth/videoHeight getters to our fixture.
  const v = document.createElement('video')
  Object.defineProperty(v, 'videoWidth', { configurable: true, get: () => w })
  Object.defineProperty(v, 'videoHeight', { configurable: true, get: () => h })
  return v
}

describe('createPoseDetector — pipeline', () => {
  it('end-to-end: returns 33-entry Landmark[] with mapped slots from canned outputs', async () => {
    // YOLO mock: emit a single strong person at letterbox (320, 320, 200, 400).
    // For a 1280x720 source -> 640x640 letterbox, scale=0.5, padX=0, padY=80.
    //   frame x1 = (320-100-0)/0.5 = 440
    //   frame y1 = (320-200-80)/0.5 = 80
    //   frame x2 = (320+100-0)/0.5 = 840
    //   frame y2 = (320+200-80)/0.5 = 880 -> clipped to 719
    const yoloOut = buildYoloOutputWithOnePerson(0, 320, 320, 200, 400, 0.9)
    const yoloSession = makeYoloSession(() => ({
      output0: { data: yoloOut, dims: [1, 84, 8400] },
    }))

    // RTMPose mock: 17 keypoints, peaks at simple positions.
    // x peak idx = 256 -> x_pixel = 128 (center of the 256-wide input).
    // y peak idx = 192 -> y_pixel = 96 (center of the 192-tall input).
    const xBins = 512
    const yBins = 384
    const xPeaks = new Array<number>(17).fill(256)
    const yPeaks = new Array<number>(17).fill(192)
    const simccX = buildSimccWithPeak(17, xBins, xPeaks)
    const simccY = buildSimccWithPeak(17, yBins, yPeaks)
    const rtmposeSession = makeRtmposeSession(simccX, simccY, xBins, yBins)

    mockState.createPlan = [
      { kind: 'return', session: yoloSession },
      { kind: 'return', session: rtmposeSession },
    ]

    const { createPoseDetector } = await import('@/lib/browserPose')
    const detector = await createPoseDetector({
      executionProviders: ['wasm'],
    })

    const fakeVideo = makeFakeVideo(1280, 720)
    const result = await detector.detect(fakeVideo)
    expect(result).not.toBeNull()
    expect(result).toHaveLength(33)

    // Every mapped BlazePose slot (the 17 we set in COCO17_TO_BLAZEPOSE33)
    // should have visibility > 0; unmapped slots should be 0.
    const mappedBlazeIds = new Set([0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
    for (let i = 0; i < 33; i++) {
      const lm = result![i]
      if (mappedBlazeIds.has(i)) {
        expect(lm.visibility).toBeGreaterThan(0)
      } else {
        expect(lm).toEqual({
          id: i,
          name: lm.name,
          x: 0,
          y: 0,
          z: 0,
          visibility: 0,
        })
      }
    }
    // detect() called run on both sessions exactly once.
    expect(yoloSession.run).toHaveBeenCalledTimes(1)
    expect(rtmposeSession.run).toHaveBeenCalledTimes(1)
  })

  it('returns null when no person detection passes threshold', async () => {
    const numAnchors = 8400
    const numClasses = 80
    const yoloOut = new Float32Array((4 + numClasses) * numAnchors)
    // All zeros => argmax tied at class 0 with score 0; below threshold 0.3.
    const yoloSession = makeYoloSession(() => ({
      output0: { data: yoloOut, dims: [1, 84, 8400] },
    }))
    const rtmposeSession = makeRtmposeSession(
      new Float32Array(17 * 512),
      new Float32Array(17 * 384),
    )
    mockState.createPlan = [
      { kind: 'return', session: yoloSession },
      { kind: 'return', session: rtmposeSession },
    ]

    const { createPoseDetector } = await import('@/lib/browserPose')
    const detector = await createPoseDetector({ executionProviders: ['wasm'] })
    const result = await detector.detect(makeFakeVideo(1280, 720))
    expect(result).toBeNull()
    // RTMPose was not called because YOLO emitted nothing.
    expect(rtmposeSession.run).not.toHaveBeenCalled()
  })

  it('falls through webgpu -> webgl when webgpu InferenceSession.create throws', async () => {
    const yoloSession = makeYoloSession(() => ({
      output0: { data: new Float32Array(84 * 8400), dims: [1, 84, 8400] },
    }))
    const rtmposeSession = makeRtmposeSession(
      new Float32Array(17 * 512),
      new Float32Array(17 * 384),
    )
    // Plan: YOLO step throws on webgpu, succeeds on webgl. RTMPose then
    // succeeds on webgl on the first try (we put the webgl-success entry
    // there).
    mockState.createPlan = [
      { kind: 'throw', error: new Error('webgpu unavailable') },
      { kind: 'return', session: yoloSession },
      { kind: 'return', session: rtmposeSession },
    ]

    const { createPoseDetector } = await import('@/lib/browserPose')
    const detector = await createPoseDetector({
      executionProviders: ['webgpu', 'webgl', 'wasm'],
    })
    expect(detector).toBeDefined()
    // Calls: webgpu (throw), webgl (yolo ok), webgl (rtm ok)
    expect(mockState.createCalls.map((c) => c.ep)).toEqual([
      'webgpu',
      'webgl',
      'webgl',
    ])
  })

  it('throws when all execution providers fail', async () => {
    mockState.createPlan = [
      { kind: 'throw', error: new Error('webgpu fail') },
      { kind: 'throw', error: new Error('webgl fail') },
      { kind: 'throw', error: new Error('wasm fail') },
    ]
    const { createPoseDetector } = await import('@/lib/browserPose')
    await expect(
      createPoseDetector({ executionProviders: ['webgpu', 'webgl', 'wasm'] }),
    ).rejects.toThrow(/No ONNX execution provider available/)
  })

  it('dispose() releases both YOLO and RTMPose sessions', async () => {
    const yoloSession = makeYoloSession(() => ({
      output0: { data: new Float32Array(84 * 8400), dims: [1, 84, 8400] },
    }))
    const rtmposeSession = makeRtmposeSession(
      new Float32Array(17 * 512),
      new Float32Array(17 * 384),
    )
    mockState.createPlan = [
      { kind: 'return', session: yoloSession },
      { kind: 'return', session: rtmposeSession },
    ]
    const { createPoseDetector } = await import('@/lib/browserPose')
    const detector = await createPoseDetector({ executionProviders: ['wasm'] })
    detector.dispose()
    expect(yoloSession.release).toHaveBeenCalledTimes(1)
    expect(rtmposeSession.release).toHaveBeenCalledTimes(1)
  })

  it('detect() after dispose() throws', async () => {
    const yoloSession = makeYoloSession(() => ({
      output0: { data: new Float32Array(84 * 8400), dims: [1, 84, 8400] },
    }))
    const rtmposeSession = makeRtmposeSession(
      new Float32Array(17 * 512),
      new Float32Array(17 * 384),
    )
    mockState.createPlan = [
      { kind: 'return', session: yoloSession },
      { kind: 'return', session: rtmposeSession },
    ]
    const { createPoseDetector } = await import('@/lib/browserPose')
    const detector = await createPoseDetector({ executionProviders: ['wasm'] })
    detector.dispose()
    await expect(detector.detect(makeFakeVideo(640, 480))).rejects.toThrow(
      /after dispose/,
    )
  })
})
