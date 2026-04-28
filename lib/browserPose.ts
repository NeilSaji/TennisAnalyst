// Browser-side pose detector built on onnxruntime-web. Phase 5B of the
// MediaPipe → RTMPose migration. Mirrors the YOLO11n + RTMPose-m pipeline
// in `railway-service/pose_rtmpose.py` so live and /analyze produce the
// same 33-entry BlazePose-indexed Landmark[] for the same input frame.
//
// Pipeline per detect() call:
//   1. Draw the source onto an internal offscreen canvas at native res.
//   2. Letterbox the frame to 640×640 and run YOLO11n. Filter to person
//      class (id 0) above 0.3 conf, NMS at IoU 0.45, pick the highest
//      score detection.
//   3. Expand the bbox ~20% so RTMPose has room around the torso, crop
//      the original frame, and letterbox-resize the crop to 256×192
//      (the ONNX shape is [1, 3, 192, 256] — height-major NCHW).
//   4. ImageNet-normalize the crop and run RTMPose-m. The model emits
//      SimCC outputs: simcc_x and simcc_y, 1-D distributions over
//      x and y bins. Decode each by argmax / 2 → keypoint position in
//      crop pixel space.
//   5. Map keypoints back through the inverse letterbox + crop offset
//      to original-frame pixel coords.
//   6. Hand to coco17ToBlazepose33 to build the 33-entry Landmark[].
//
// The detector lazy-loads both ONNX models via lib/modelLoader.ts on
// first instantiation; once an InferenceSession is built it is reused
// across detect() calls (and across detector instances if the user
// builds multiple — onnxruntime-web caches its WASM artifacts).
//
// Execution provider fallback: webgpu → webgl → wasm. Both YOLO and
// RTMPose use the same provider — pick once at init.
//
// IMPORTANT for Worker D / future maintainers: the RTMPose ONNX shape
// (1, 3, 192, 256) and the SimCC output dims (256·2=512 for x,
// 192·2=384 for y) are documented assumptions. The implementation
// reads the actual shapes from session.inputMetadata /
// session.outputMetadata at init and adapts. If the upstream model
// changes (e.g. a new RTMPose variant with different SimCC scale),
// detect() should still work because we use the runtime-reported dims
// rather than hard-coded constants.

'use client'

import * as ort from 'onnxruntime-web'
import { loadModel, type LoadModelOptions } from '@/lib/modelLoader'
import { coco17ToBlazepose33 } from '@/lib/cocoToBlazepose'
import type { Landmark } from '@/lib/supabase'

// Silence onnxruntime-web's "some nodes fell back to CPU" warning. It's
// informational (the session is fine), but Next.js's dev overlay shows
// any console.warn from libs as a top-level error which spooks the user.
// 'error' lets real failures through but suppresses normal init noise.
// Defensive: vitest mocks of onnxruntime-web may not define `env`.
if (typeof window !== 'undefined') {
  try {
    if (ort.env) ort.env.logLevel = 'error'
  } catch {
    // If `env` isn't on the runtime (mocks, future API change), skip
    // — the warning is cosmetic, not load-bearing.
  }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type ExecutionProvider = 'webgpu' | 'webgl' | 'wasm'

export interface CreatePoseDetectorOptions {
  /** Defaults to '/models/yolo11n.onnx'. */
  yoloModelUrl?: string
  /** Defaults to '/models/rtmpose-m.onnx'. */
  rtmposeModelUrl?: string
  /**
   * Fired as bytes arrive for either model. `label` distinguishes them
   * so the UX can render a per-model progress bar or aggregate them.
   * Both models share the same `loadModel` cache so repeat sessions
   * fire one tick each at `loaded === total` (cache hit).
   */
  onProgress?: (loaded: number, total: number, label: 'yolo' | 'rtmpose') => void
  /**
   * Order of execution providers to try. The first one that successfully
   * builds an InferenceSession wins. Default tries WebGPU → WebGL → WASM.
   */
  executionProviders?: ReadonlyArray<ExecutionProvider>
  /** Override the cache key for the YOLO model (see modelLoader). */
  yoloCacheKey?: string
  /** Override the cache key for the RTMPose model. */
  rtmposeCacheKey?: string
}

/**
 * Per-call pipeline diagnostics — populated on every `detect()` invocation
 * so a live UI can surface where detection failed without DevTools.
 * Useful when "Move into frame" fires unexpectedly: yoloDetections=0 means
 * YOLO didn't see anyone, kpMaxConf=0 means RTMPose didn't run, etc.
 */
export interface PoseDetectStats {
  yoloDetections: number
  yoloTopScore: number
  kpCount: number
  kpMaxConf: number
  inferenceMs: number
  /** Provider that actually ran (webgpu / webgl / wasm). */
  provider: ExecutionProvider | null
}

export interface PoseDetector {
  /**
   * Run the pipeline on a single frame source. Returns a 33-entry
   * BlazePose-indexed landmark list, or null when no person was
   * detected with sufficient confidence.
   */
  detect(
    source: HTMLCanvasElement | OffscreenCanvas | HTMLVideoElement,
  ): Promise<Landmark[] | null>
  /** Snapshot of the most recent detect() call's diagnostics. */
  getLastStats(): PoseDetectStats
  /** Release the underlying ONNX sessions and clear cached buffers. */
  dispose(): void
}

// ---------------------------------------------------------------------------
// Constants — see `pose_rtmpose.py` for source of truth
// ---------------------------------------------------------------------------

const YOLO_INPUT_SIZE = 640
const YOLO_PERSON_CLASS_ID = 0
// Lowered from 0.3 -> 0.15 after live testing where the "Move into
// frame" pill kept firing despite the user being clearly in frame.
// On portrait phone aspect with the player at a court-back distance,
// YOLO's letterboxed input shows the player at ~10-20% of the 640px
// frame, which produces person-class scores in the 0.15-0.25 range.
// 0.15 catches those without picking up phantom non-person blobs.
const YOLO_PERSON_CONF_THRESHOLD = 0.15
const YOLO_NMS_IOU_THRESHOLD = 0.45
const YOLO_NUM_CLASSES = 80

// RTMPose-m body7 256×192 — the (W=256, H=192) crop fed to the model.
const RTMPOSE_INPUT_W = 256
const RTMPOSE_INPUT_H = 192

// Bbox expansion before RTMPose crop. Matches the Railway value (8%)
// after live testing showed the previous 20% padding was wasting input
// resolution — the player filled less of the 256x192 input, blurring
// keypoints. /analyze keypoints (server-side at 8%) read as "immaculate"
// per user; matching the value brings live in line.
const BBOX_EXPAND_PCT = 0.08

// ImageNet mean/std applied per channel after [0,1] normalization. These
// are the defaults baked into rtmlib's preprocessing for RTMPose.
const IMAGENET_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406]
const IMAGENET_STD: readonly [number, number, number] = [0.229, 0.224, 0.225]

// Letterbox pad value for YOLO (gray=114/255). Matches railway _letterbox_for_yolo.
const YOLO_PAD_VALUE_FLOAT = 114 / 255

// Number of COCO keypoints RTMPose emits.
const NUM_COCO_KEYPOINTS = 17

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/** Affine transform from source-image px → letterboxed-input px. */
export interface LetterboxTransform {
  scale: number
  padX: number
  padY: number
  /** The padded image size in (w, h). */
  outW: number
  outH: number
  /** The original source image dims used to build the transform. */
  srcW: number
  srcH: number
}

/** A YOLO detection in original-image pixel xyxy coords. */
export interface PersonDetection {
  x1: number
  y1: number
  x2: number
  y2: number
  score: number
}

// ---------------------------------------------------------------------------
// Letterbox math (preprocessing)
// ---------------------------------------------------------------------------

/**
 * Compute the affine transform that resizes an `srcW × srcH` image to
 * fit inside `outW × outH` while preserving aspect ratio. The remaining
 * area is centered with equal padding on opposite sides.
 *
 * Used for both YOLO (square 640×640) and RTMPose (rectangular 256×192).
 */
export function computeLetterboxTransform(
  srcW: number,
  srcH: number,
  outW: number,
  outH: number,
): LetterboxTransform {
  const scale = Math.min(outW / srcW, outH / srcH)
  const newW = Math.round(srcW * scale)
  const newH = Math.round(srcH * scale)
  const padX = Math.floor((outW - newW) / 2)
  const padY = Math.floor((outH - newH) / 2)
  return { scale, padX, padY, outW, outH, srcW, srcH }
}

// ---------------------------------------------------------------------------
// Canvas-based image preprocessing
// ---------------------------------------------------------------------------

/**
 * Draw `source` onto a destination 2D context, letterboxed into the
 * target size. Returns the affine transform so callers can invert it
 * to map model-space coordinates back to the source.
 */
function drawLetterboxedTo(
  source: CanvasImageSource,
  srcW: number,
  srcH: number,
  destCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  outW: number,
  outH: number,
  padFillStyle: string,
): LetterboxTransform {
  const tx = computeLetterboxTransform(srcW, srcH, outW, outH)
  destCtx.fillStyle = padFillStyle
  destCtx.fillRect(0, 0, outW, outH)
  const drawW = Math.round(srcW * tx.scale)
  const drawH = Math.round(srcH * tx.scale)
  destCtx.drawImage(source, tx.padX, tx.padY, drawW, drawH)
  return tx
}

/**
 * Convert raw RGBA ImageData → Float32Array NCHW with /255 normalization.
 * No mean/std applied; pure [0,1] scaling. Used by YOLO.
 */
export function imageDataToNCHWNormalized(
  imageData: ImageData,
): Float32Array {
  const { width: w, height: h, data } = imageData
  const out = new Float32Array(3 * w * h)
  const planeSize = w * h
  for (let i = 0; i < planeSize; i++) {
    const r = data[i * 4]
    const g = data[i * 4 + 1]
    const b = data[i * 4 + 2]
    out[i] = r / 255
    out[planeSize + i] = g / 255
    out[2 * planeSize + i] = b / 255
  }
  return out
}

/**
 * Convert raw RGBA ImageData → Float32Array NCHW with ImageNet mean/std
 * normalization. Used by RTMPose.
 */
export function imageDataToNCHWImageNet(
  imageData: ImageData,
): Float32Array {
  const { width: w, height: h, data } = imageData
  const out = new Float32Array(3 * w * h)
  const planeSize = w * h
  const [mr, mg, mb] = IMAGENET_MEAN
  const [sr, sg, sb] = IMAGENET_STD
  for (let i = 0; i < planeSize; i++) {
    const r = data[i * 4] / 255
    const g = data[i * 4 + 1] / 255
    const b = data[i * 4 + 2] / 255
    out[i] = (r - mr) / sr
    out[planeSize + i] = (g - mg) / sg
    out[2 * planeSize + i] = (b - mb) / sb
  }
  return out
}

// ---------------------------------------------------------------------------
// YOLO postprocessing — decode + NMS
// ---------------------------------------------------------------------------

/**
 * Decode the YOLO11n output tensor `(1, 4 + nc, 8400)` into a list of
 * person-class detections in *letterbox* pixel space. Caller maps back
 * to original image coords with `mapBoxFromLetterbox`.
 *
 * Filters by class id and confidence threshold; the higher-level
 * `pickHighestPerson` picks the winner after NMS.
 */
export function decodeYoloPersonDetections(
  output: Float32Array,
  numAnchors: number,
  numClasses: number,
  classId: number,
  confThreshold: number,
): PersonDetection[] {
  // Layout: channel-major. Channel offsets within the flat tensor:
  //   cx[a]    = output[0 * numAnchors + a]
  //   cy[a]    = output[1 * numAnchors + a]
  //   w[a]     = output[2 * numAnchors + a]
  //   h[a]     = output[3 * numAnchors + a]
  //   cls_c[a] = output[(4 + c) * numAnchors + a]
  const detections: PersonDetection[] = []
  for (let a = 0; a < numAnchors; a++) {
    // Find argmax class for this anchor.
    let bestCls = -1
    let bestScore = -Infinity
    for (let c = 0; c < numClasses; c++) {
      const s = output[(4 + c) * numAnchors + a]
      if (s > bestScore) {
        bestScore = s
        bestCls = c
      }
    }
    if (bestCls !== classId) continue
    if (bestScore < confThreshold) continue

    const cx = output[a]
    const cy = output[numAnchors + a]
    const bw = output[2 * numAnchors + a]
    const bh = output[3 * numAnchors + a]
    detections.push({
      x1: cx - bw / 2,
      y1: cy - bh / 2,
      x2: cx + bw / 2,
      y2: cy + bh / 2,
      score: bestScore,
    })
  }
  return detections
}

function iou(a: PersonDetection, b: PersonDetection): number {
  const x1 = Math.max(a.x1, b.x1)
  const y1 = Math.max(a.y1, b.y1)
  const x2 = Math.min(a.x2, b.x2)
  const y2 = Math.min(a.y2, b.y2)
  const interW = Math.max(0, x2 - x1)
  const interH = Math.max(0, y2 - y1)
  const inter = interW * interH
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1)
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1)
  const union = areaA + areaB - inter
  return union > 0 ? inter / union : 0
}

/**
 * Standard greedy NMS. Sorts by score desc, picks the top, suppresses
 * any subsequent detection whose IoU with the picked box exceeds
 * `iouThreshold`, repeats. Returns survivors in score-desc order.
 */
export function nonMaxSuppression(
  dets: PersonDetection[],
  iouThreshold: number,
): PersonDetection[] {
  const sorted = [...dets].sort((a, b) => b.score - a.score)
  const kept: PersonDetection[] = []
  const suppressed = new Array<boolean>(sorted.length).fill(false)
  for (let i = 0; i < sorted.length; i++) {
    if (suppressed[i]) continue
    const a = sorted[i]
    kept.push(a)
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed[j]) continue
      if (iou(a, sorted[j]) > iouThreshold) suppressed[j] = true
    }
  }
  return kept
}

/**
 * Pick the highest-confidence detection from a list. Returns null on
 * empty input.
 */
export function pickHighestPerson(
  dets: PersonDetection[],
): PersonDetection | null {
  if (dets.length === 0) return null
  let best = dets[0]
  for (let i = 1; i < dets.length; i++) {
    if (dets[i].score > best.score) best = dets[i]
  }
  return best
}

/**
 * Map a bbox from letterboxed model-input coords back to original-image
 * pixel coords. Inverse of `computeLetterboxTransform`.
 */
export function mapBoxFromLetterbox(
  box: PersonDetection,
  tx: LetterboxTransform,
): PersonDetection {
  return {
    x1: clamp((box.x1 - tx.padX) / tx.scale, 0, tx.srcW - 1),
    y1: clamp((box.y1 - tx.padY) / tx.scale, 0, tx.srcH - 1),
    x2: clamp((box.x2 - tx.padX) / tx.scale, 0, tx.srcW - 1),
    y2: clamp((box.y2 - tx.padY) / tx.scale, 0, tx.srcH - 1),
    score: box.score,
  }
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x))
}

/** Expand bbox outward by `pct` on each side, clipped to image. */
export function expandBox(
  box: PersonDetection,
  imgW: number,
  imgH: number,
  pct: number,
): PersonDetection {
  const bw = box.x2 - box.x1
  const bh = box.y2 - box.y1
  const dx = bw * pct
  const dy = bh * pct
  return {
    x1: Math.max(0, box.x1 - dx),
    y1: Math.max(0, box.y1 - dy),
    x2: Math.min(imgW - 1, box.x2 + dx),
    y2: Math.min(imgH - 1, box.y2 + dy),
    score: box.score,
  }
}

// ---------------------------------------------------------------------------
// SimCC decoding (RTMPose postprocessing)
// ---------------------------------------------------------------------------

export interface DecodedKeypoint {
  /** x in crop pixel space, [0, RTMPOSE_INPUT_W). */
  x: number
  /** y in crop pixel space, [0, RTMPOSE_INPUT_H). */
  y: number
  /** Joint visibility / confidence — peak height in the SimCC dist. */
  score: number
}

/**
 * Decode RTMPose SimCC outputs into per-keypoint (x, y, score) tuples
 * in crop-pixel space.
 *
 * `simccX` shape: (1, K, Wbins) where Wbins = RTMPOSE_INPUT_W * simccSplit.
 * `simccY` shape: (1, K, Hbins) where Hbins = RTMPOSE_INPUT_H * simccSplit.
 *
 * Argmax bin / simccSplit → pixel position. Score = max value of the
 * x-axis distribution (rtmlib uses min(max_x, max_y); we follow the same
 * convention which is the conservative confidence estimator).
 */
export function decodeSimCC(
  simccX: Float32Array,
  simccY: Float32Array,
  numKeypoints: number,
  xBins: number,
  yBins: number,
  simccSplit: number,
): DecodedKeypoint[] {
  const out: DecodedKeypoint[] = []
  for (let k = 0; k < numKeypoints; k++) {
    let bestXIdx = 0
    let bestXVal = -Infinity
    const xOffset = k * xBins
    for (let i = 0; i < xBins; i++) {
      const v = simccX[xOffset + i]
      if (v > bestXVal) {
        bestXVal = v
        bestXIdx = i
      }
    }
    let bestYIdx = 0
    let bestYVal = -Infinity
    const yOffset = k * yBins
    for (let i = 0; i < yBins; i++) {
      const v = simccY[yOffset + i]
      if (v > bestYVal) {
        bestYVal = v
        bestYIdx = i
      }
    }
    out.push({
      x: bestXIdx / simccSplit,
      y: bestYIdx / simccSplit,
      // rtmlib's per-kpt conf is min(max_x, max_y). We follow suit because
      // either dimension being uncertain implies the keypoint as a whole is.
      score: Math.min(bestXVal, bestYVal),
    })
  }
  return out
}

/**
 * Map a keypoint from RTMPose 256×192 input space → original frame
 * pixel coords. Composes the inverse of the input-letterbox affine
 * with the original crop offset.
 */
export function mapKeypointToFrame(
  kpt: DecodedKeypoint,
  cropTx: LetterboxTransform,
  cropOffsetX: number,
  cropOffsetY: number,
): { x: number; y: number; score: number } {
  // Undo letterbox: kpt pixel → crop-source pixel.
  const cropX = (kpt.x - cropTx.padX) / cropTx.scale
  const cropY = (kpt.y - cropTx.padY) / cropTx.scale
  // Add crop offset to land in original-frame pixel coords.
  return {
    x: cropX + cropOffsetX,
    y: cropY + cropOffsetY,
    score: kpt.score,
  }
}

// ---------------------------------------------------------------------------
// Source dimension helpers
// ---------------------------------------------------------------------------

function getSourceDims(
  source: HTMLCanvasElement | OffscreenCanvas | HTMLVideoElement,
): { w: number; h: number } {
  if (typeof HTMLVideoElement !== 'undefined' && source instanceof HTMLVideoElement) {
    return { w: source.videoWidth, h: source.videoHeight }
  }
  // HTMLCanvasElement and OffscreenCanvas both expose width/height directly.
  // `source.width` / `source.height` are the bitmap dimensions, which is
  // what we want for pose inference.
  const c = source as HTMLCanvasElement | OffscreenCanvas
  return { w: c.width, h: c.height }
}

// ---------------------------------------------------------------------------
// Session creation with execution-provider fallback
// ---------------------------------------------------------------------------

async function createSessionWithFallback(
  buffer: Uint8Array,
  providers: ReadonlyArray<ExecutionProvider>,
): Promise<{ session: ort.InferenceSession; provider: ExecutionProvider }> {
  let lastErr: unknown = null
  for (const ep of providers) {
    try {
      const session = await ort.InferenceSession.create(buffer, {
        executionProviders: [ep],
      })
      return { session, provider: ep }
    } catch (err) {
      lastErr = err
      // Best-effort warning; intentionally non-fatal — we move to the
      // next provider in the chain.
      // eslint-disable-next-line no-console
      console.warn(
        `[browserPose] onnxruntime-web: ${ep} unavailable, falling back`,
        err,
      )
    }
  }
  throw new Error(
    `[browserPose] No ONNX execution provider available. Last error: ${
      lastErr instanceof Error ? lastErr.message : String(lastErr)
    }`,
  )
}

// ---------------------------------------------------------------------------
// Detector implementation
// ---------------------------------------------------------------------------

interface InternalState {
  yoloSession: ort.InferenceSession
  rtmposeSession: ort.InferenceSession
  yoloInputName: string
  rtmposeInputName: string
  // Output names for RTMPose. Most variants expose two outputs in this
  // order: simcc_x, simcc_y. We resolve them from session.outputNames at
  // init so a permuted shipping artifact still works.
  rtmposeOutXName: string
  rtmposeOutYName: string
  // Bins per axis — we read these from outputMetadata when available
  // and fall back to the canonical (W*2, H*2) values otherwise.
  simccXBins: number
  simccYBins: number
  // simcc_split factor: bins / pixels. Canonical RTMPose-m emits 2.0.
  simccSplit: number
  // Reused offscreen canvases — avoid re-allocating per detect() call.
  yoloCanvas: OffscreenCanvas | HTMLCanvasElement
  yoloCtx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D
  rtmposeCanvas: OffscreenCanvas | HTMLCanvasElement
  rtmposeCtx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D
  // Source-sized canvas for the original frame (so we can readback for
  // the RTMPose crop). Resized lazily as the source dims change.
  sourceCanvas: OffscreenCanvas | HTMLCanvasElement
  sourceCtx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D
  disposed: boolean
  // Mutated in-place after every detect() call so the live UI can read
  // the latest pipeline stats without a round trip.
  lastStats: PoseDetectStats
}

function makeOffscreen(
  w: number,
  h: number,
): {
  canvas: OffscreenCanvas | HTMLCanvasElement
  ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D
} {
  if (typeof OffscreenCanvas !== 'undefined') {
    const c = new OffscreenCanvas(w, h)
    const ctx = c.getContext('2d')
    if (!ctx) {
      throw new Error('[browserPose] OffscreenCanvas 2d context unavailable')
    }
    return { canvas: c, ctx: ctx as OffscreenCanvasRenderingContext2D }
  }
  // Fallback to HTMLCanvasElement. Should only matter outside the browser
  // (e.g., test environments where OffscreenCanvas is missing).
  if (typeof document === 'undefined') {
    throw new Error(
      '[browserPose] No canvas API available (no OffscreenCanvas, no document)',
    )
  }
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')
  if (!ctx) {
    throw new Error('[browserPose] HTMLCanvasElement 2d context unavailable')
  }
  return { canvas: c, ctx }
}

function resizeCanvas(
  canvas: OffscreenCanvas | HTMLCanvasElement,
  w: number,
  h: number,
): void {
  if (canvas.width !== w) canvas.width = w
  if (canvas.height !== h) canvas.height = h
}

function metadataNumericDim(
  meta: ort.InferenceSession.ValueMetadata | undefined,
  axis: number,
): number | null {
  if (!meta || !meta.isTensor) return null
  const dim = meta.shape[axis]
  return typeof dim === 'number' && dim > 0 ? dim : null
}

/**
 * Build a PoseDetector by lazy-loading both ONNX models and creating
 * shared inference sessions. The returned detector is reusable across
 * detect() calls.
 */
export async function createPoseDetector(
  opts: CreatePoseDetectorOptions = {},
): Promise<PoseDetector> {
  const yoloUrl = opts.yoloModelUrl ?? '/models/yolo11n.onnx'
  const rtmposeUrl = opts.rtmposeModelUrl ?? '/models/rtmpose-m.onnx'
  const providers = opts.executionProviders ?? (['webgpu', 'webgl', 'wasm'] as const)

  const yoloLoadOpts: LoadModelOptions = {
    onProgress: opts.onProgress
      ? (l, t) => opts.onProgress!(l, t, 'yolo')
      : undefined,
    cacheKey: opts.yoloCacheKey,
  }
  const rtmposeLoadOpts: LoadModelOptions = {
    onProgress: opts.onProgress
      ? (l, t) => opts.onProgress!(l, t, 'rtmpose')
      : undefined,
    cacheKey: opts.rtmposeCacheKey,
  }

  const [yoloBytes, rtmposeBytes] = await Promise.all([
    loadModel(yoloUrl, yoloLoadOpts),
    loadModel(rtmposeUrl, rtmposeLoadOpts),
  ])

  // Both sessions share the chosen provider — pick once on the first
  // session that succeeds, then reuse it for the second.
  const { session: yoloSession, provider } = await createSessionWithFallback(
    yoloBytes,
    providers,
  )
  // Try the same provider first for RTMPose; if that specific model
  // fails on this provider (unlikely but possible for op-coverage
  // reasons), fall back through the rest of the chain.
  const rtmFallback: ReadonlyArray<ExecutionProvider> = [
    provider,
    ...providers.filter((p) => p !== provider),
  ]
  const { session: rtmposeSession } = await createSessionWithFallback(
    rtmposeBytes,
    rtmFallback,
  )

  const yoloInputName = yoloSession.inputNames[0]
  const rtmposeInputName = rtmposeSession.inputNames[0]
  // RTMPose output naming: most variants ship as ['simcc_x', 'simcc_y'] in
  // that order. We name-match where possible and fall back to positional.
  const outNames = rtmposeSession.outputNames
  let outXName = outNames[0]
  let outYName = outNames[1] ?? outNames[0]
  for (const n of outNames) {
    const lower = n.toLowerCase()
    if (lower.includes('simcc_x') || lower.endsWith('_x')) outXName = n
    if (lower.includes('simcc_y') || lower.endsWith('_y')) outYName = n
  }

  // Read SimCC bin counts from output metadata when shape is fully static.
  // Fall back to the canonical RTMPose-m values (W*2, H*2).
  let simccXBins = RTMPOSE_INPUT_W * 2
  let simccYBins = RTMPOSE_INPUT_H * 2
  try {
    const outMeta = rtmposeSession.outputMetadata
    if (outMeta && outMeta.length >= 2) {
      const xMeta = outMeta.find((m) => m.name === outXName)
      const yMeta = outMeta.find((m) => m.name === outYName)
      const xDim = metadataNumericDim(xMeta, 2)
      const yDim = metadataNumericDim(yMeta, 2)
      if (xDim) simccXBins = xDim
      if (yDim) simccYBins = yDim
    }
  } catch {
    // outputMetadata access is best-effort — older onnxruntime-web versions
    // don't expose it. Fall through to defaults.
  }

  // simcc_split = bins / pixels along an axis. Should be ~2.0 for body7.
  const simccSplit = simccXBins / RTMPOSE_INPUT_W

  const { canvas: yoloCanvas, ctx: yoloCtx } = makeOffscreen(
    YOLO_INPUT_SIZE,
    YOLO_INPUT_SIZE,
  )
  const { canvas: rtmposeCanvas, ctx: rtmposeCtx } = makeOffscreen(
    RTMPOSE_INPUT_W,
    RTMPOSE_INPUT_H,
  )
  // Source canvas resized lazily on first detect().
  const { canvas: sourceCanvas, ctx: sourceCtx } = makeOffscreen(1, 1)

  const state: InternalState = {
    yoloSession,
    rtmposeSession,
    yoloInputName,
    rtmposeInputName,
    rtmposeOutXName: outXName,
    rtmposeOutYName: outYName,
    simccXBins,
    simccYBins,
    simccSplit,
    yoloCanvas,
    yoloCtx,
    rtmposeCanvas,
    rtmposeCtx,
    sourceCanvas,
    sourceCtx,
    disposed: false,
    lastStats: {
      yoloDetections: 0,
      yoloTopScore: 0,
      kpCount: 0,
      kpMaxConf: 0,
      inferenceMs: 0,
      provider,
    } as PoseDetectStats,
  }

  return {
    detect: (source) => detectImpl(state, source),
    getLastStats: () => state.lastStats,
    dispose: () => disposeImpl(state),
  }
}

// ---------------------------------------------------------------------------
// Per-frame detection
// ---------------------------------------------------------------------------

async function detectImpl(
  state: InternalState,
  source: HTMLCanvasElement | OffscreenCanvas | HTMLVideoElement,
): Promise<Landmark[] | null> {
  if (state.disposed) {
    throw new Error('[browserPose] detect() called after dispose()')
  }

  const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now())
  // Reset per-call stats up-front. We'll fill them in as the pipeline
  // progresses; an early null-return below leaves the zeros visible to
  // the UI so the user can see "yolo found nothing".
  state.lastStats.yoloDetections = 0
  state.lastStats.yoloTopScore = 0
  state.lastStats.kpCount = 0
  state.lastStats.kpMaxConf = 0
  state.lastStats.inferenceMs = 0

  const { w: srcW, h: srcH } = getSourceDims(source)
  if (!srcW || !srcH || !Number.isFinite(srcW) || !Number.isFinite(srcH)) {
    return null
  }

  // ---- Stage the source frame on a private same-size canvas. We need
  // pixel access for the RTMPose crop step; drawing the source straight
  // onto the YOLO and RTMPose canvases also works but means two reads
  // from the source for each frame. One staging copy saves a video
  // texture upload on the second draw.
  resizeCanvas(state.sourceCanvas, srcW, srcH)
  state.sourceCtx.drawImage(source, 0, 0, srcW, srcH)

  // ---- (1) YOLO: letterbox + run + decode + NMS + pick top.
  const yoloTx = drawLetterboxedTo(
    state.sourceCanvas,
    srcW,
    srcH,
    state.yoloCtx,
    YOLO_INPUT_SIZE,
    YOLO_INPUT_SIZE,
    `rgb(114, 114, 114)`,
  )
  const yoloImageData = state.yoloCtx.getImageData(
    0,
    0,
    YOLO_INPUT_SIZE,
    YOLO_INPUT_SIZE,
  )
  const yoloInput = imageDataToNCHWNormalized(yoloImageData)
  const yoloTensor = new ort.Tensor('float32', yoloInput, [
    1,
    3,
    YOLO_INPUT_SIZE,
    YOLO_INPUT_SIZE,
  ])
  const yoloFeed: Record<string, ort.Tensor> = {}
  yoloFeed[state.yoloInputName] = yoloTensor
  const yoloResult = await state.yoloSession.run(yoloFeed)
  const yoloOut = yoloResult[state.yoloSession.outputNames[0]]
  const yoloOutData = yoloOut.data as Float32Array
  // Output dims: (1, 4 + nc, N). We trust YOLO_NUM_CLASSES=80 (COCO) but
  // derive numAnchors from the tensor shape so the decode handles export
  // variations gracefully.
  const yoloDims = yoloOut.dims
  const numAnchors =
    yoloDims.length === 3 ? Number(yoloDims[2]) : yoloOutData.length / (4 + YOLO_NUM_CLASSES)

  const rawDets = decodeYoloPersonDetections(
    yoloOutData,
    numAnchors,
    YOLO_NUM_CLASSES,
    YOLO_PERSON_CLASS_ID,
    YOLO_PERSON_CONF_THRESHOLD,
  )
  const nmsDets = nonMaxSuppression(rawDets, YOLO_NMS_IOU_THRESHOLD)
  const topDet = pickHighestPerson(nmsDets)
  state.lastStats.yoloDetections = nmsDets.length
  state.lastStats.yoloTopScore = topDet?.score ?? 0
  if (!topDet) {
    state.lastStats.inferenceMs = Math.round(
      (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0,
    )
    return null
  }

  // Map back from letterbox space to original frame pixels.
  const detInFrame = mapBoxFromLetterbox(topDet, yoloTx)

  // ---- (2) Crop + letterbox to RTMPose 256×192.
  const expanded = expandBox(detInFrame, srcW, srcH, BBOX_EXPAND_PCT)
  const cropX = Math.max(0, Math.floor(expanded.x1))
  const cropY = Math.max(0, Math.floor(expanded.y1))
  const cropW = Math.max(1, Math.ceil(expanded.x2 - expanded.x1))
  const cropH = Math.max(1, Math.ceil(expanded.y2 - expanded.y1))

  // Build the letterbox transform from the crop dims to RTMPose input dims,
  // and draw the crop directly onto the RTMPose canvas.
  const rtmTx = computeLetterboxTransform(
    cropW,
    cropH,
    RTMPOSE_INPUT_W,
    RTMPOSE_INPUT_H,
  )
  state.rtmposeCtx.fillStyle = 'rgb(128, 128, 128)'
  state.rtmposeCtx.fillRect(0, 0, RTMPOSE_INPUT_W, RTMPOSE_INPUT_H)
  state.rtmposeCtx.drawImage(
    state.sourceCanvas,
    cropX,
    cropY,
    cropW,
    cropH,
    rtmTx.padX,
    rtmTx.padY,
    Math.round(cropW * rtmTx.scale),
    Math.round(cropH * rtmTx.scale),
  )

  const rtmImageData = state.rtmposeCtx.getImageData(
    0,
    0,
    RTMPOSE_INPUT_W,
    RTMPOSE_INPUT_H,
  )
  const rtmInput = imageDataToNCHWImageNet(rtmImageData)
  // ONNX input shape is [1, 3, H, W] — height-major NCHW.
  const rtmTensor = new ort.Tensor('float32', rtmInput, [
    1,
    3,
    RTMPOSE_INPUT_H,
    RTMPOSE_INPUT_W,
  ])
  const rtmFeed: Record<string, ort.Tensor> = {}
  rtmFeed[state.rtmposeInputName] = rtmTensor
  const rtmResult = await state.rtmposeSession.run(rtmFeed)
  const simccX = rtmResult[state.rtmposeOutXName].data as Float32Array
  const simccY = rtmResult[state.rtmposeOutYName].data as Float32Array

  const decoded = decodeSimCC(
    simccX,
    simccY,
    NUM_COCO_KEYPOINTS,
    state.simccXBins,
    state.simccYBins,
    state.simccSplit,
  )

  // ---- (3) Map keypoints back to original frame pixel coords.
  // The crop offset is (cropX, cropY); the RTMPose letterbox transform is
  // from crop → 256×192. Composing them:
  //   crop_px = (kpt - rtm_pad) / rtm_scale
  //   frame_px = crop_px + (cropX, cropY)
  const cocoKpts = new Float32Array(NUM_COCO_KEYPOINTS * 2)
  const cocoScores = new Float32Array(NUM_COCO_KEYPOINTS)
  let kpMaxConf = 0
  for (let i = 0; i < NUM_COCO_KEYPOINTS; i++) {
    const mapped = mapKeypointToFrame(decoded[i], rtmTx, cropX, cropY)
    cocoKpts[i * 2] = mapped.x
    cocoKpts[i * 2 + 1] = mapped.y
    cocoScores[i] = mapped.score
    if (mapped.score > kpMaxConf) kpMaxConf = mapped.score
  }

  state.lastStats.kpCount = NUM_COCO_KEYPOINTS
  state.lastStats.kpMaxConf = kpMaxConf
  state.lastStats.inferenceMs = Math.round(
    (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0,
  )

  return coco17ToBlazepose33(cocoKpts, cocoScores, srcW, srcH)
}

function disposeImpl(state: InternalState): void {
  if (state.disposed) return
  state.disposed = true
  // ort.InferenceSession.release() is async (returns Promise<void>) but
  // the dispose() contract is sync — we kick it off and ignore. Failures
  // here are non-fatal: the JS side is gone either way.
  try {
    state.yoloSession.release()
  } catch {
    // ignore
  }
  try {
    state.rtmposeSession.release()
  } catch {
    // ignore
  }
}
