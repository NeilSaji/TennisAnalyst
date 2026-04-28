// Browser-side helper for loading ONNX model weights with progress
// reporting and Cache Storage caching. Phase 5A — used by the
// browserPose module (Worker B) to pull yolo11n.onnx and rtmpose-m.onnx
// from /models/ on first run, then read them from cache on every
// subsequent session so cold-starts are <500ms after the first.
//
// Why Cache Storage and not IndexedDB?
//   - Files are public static assets; Cache API is the natural fit and
//     stores Response objects whole, so we don't burn time round-tripping
//     to ArrayBuffer.
//   - onnxruntime-web also caches its own WASM artifacts via Cache API,
//     so we stay consistent with what's already in the browser's cache
//     bucket for this origin.
//
// Concurrency: two near-simultaneous loadModel() calls for the same URL
// must share a single network fetch. We keep an in-memory map of
// in-flight Promises keyed on `cacheKey ?? url`.

const CACHE_NAME = 'tennis-analyst-models'

// Module-level so concurrent callers (e.g. YOLO + RTMPose initializing
// in parallel) share the work. Cleared on resolution / rejection.
const inFlight = new Map<string, Promise<Uint8Array>>()

export interface LoadModelOptions {
  /**
   * Fires as bytes arrive. `loaded` is cumulative; `total` comes from
   * Content-Length. When the response is missing Content-Length (rare
   * for our static assets, but possible behind some CDNs) `total` will
   * equal `loaded` so callers can still render a sane progress bar
   * that finishes at 100% on the final tick.
   *
   * On a cache hit, fires once with `loaded === total` so the UX layer
   * can transition out of its "loading" state without special-casing.
   */
  onProgress?: (loadedBytes: number, totalBytes: number) => void

  /**
   * Override the cache lookup key. Default: the URL. Useful for
   * cache-busting on model version bumps without changing the URL
   * served to other consumers.
   */
  cacheKey?: string
}

/**
 * Fetches `url`, returning the body as a Uint8Array suitable for
 * `ort.InferenceSession.create(buf)`. On the first call: streams the
 * download with progress reporting and stores the response in Cache
 * Storage. On subsequent calls: short-circuits to the cached response.
 *
 * Throws if the network fetch fails AND nothing is cached.
 */
export async function loadModel(
  url: string,
  opts: LoadModelOptions = {},
): Promise<Uint8Array> {
  const key = opts.cacheKey ?? url
  const existing = inFlight.get(key)
  if (existing) return existing

  const promise = doLoad(url, key, opts).finally(() => {
    inFlight.delete(key)
  })
  inFlight.set(key, promise)
  return promise
}

async function doLoad(
  url: string,
  key: string,
  opts: LoadModelOptions,
): Promise<Uint8Array> {
  const cache = await openCacheSafe()

  if (cache) {
    const hit = await cache.match(key)
    if (hit) {
      const buf = await hit.arrayBuffer()
      const bytes = new Uint8Array(buf)
      // Fire one progress event so callers can drop their loading UI
      // without having to detect "cache hit" themselves.
      opts.onProgress?.(bytes.byteLength, bytes.byteLength)
      return bytes
    }
  }

  // Cache miss (or no Cache Storage). Stream the network fetch so the
  // UI can render a real progress bar — these models are 10-50 MB.
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(
      `loadModel: failed to fetch ${url} — HTTP ${response.status} ${response.statusText}`,
    )
  }

  const contentLengthHeader = response.headers.get('content-length')
  const total = contentLengthHeader ? Number(contentLengthHeader) : 0

  // Tee the stream so we can both report progress AND hand the original
  // Response to Cache Storage. Without teeing, reading the body would
  // consume it before cache.put() can store it.
  let bytes: Uint8Array
  if (response.body && typeof response.body.getReader === 'function') {
    const [forRead, forCache] = response.body.tee()
    const cachePromise = cache
      ? cache.put(key, new Response(forCache, { headers: response.headers })).catch(() => {
          // Cache write failures are non-fatal — caller still gets the bytes
          // back via the read side. (Quota exceeded, private mode, etc.)
        })
      : Promise.resolve()
    bytes = await readWithProgress(forRead.getReader(), total, opts.onProgress)
    await cachePromise
  } else {
    // Fallback: no streaming reader available (very old browsers, or a
    // mocked Response without a body stream). Buffer the whole thing.
    const buf = await response.arrayBuffer()
    bytes = new Uint8Array(buf)
    opts.onProgress?.(bytes.byteLength, total || bytes.byteLength)
    if (cache) {
      try {
        // TS 5.7 tightened Uint8Array's generic to track its underlying
        // buffer type. `bytes` is Uint8Array<ArrayBufferLike> (because
        // arrayBuffer() returns ArrayBufferLike, not ArrayBuffer), and
        // BodyInit only accepts Uint8Array<ArrayBuffer>. Pass the
        // backing buffer directly — it's the right type and avoids an
        // unnecessary copy.
        await cache.put(
          key,
          new Response(bytes.buffer as ArrayBuffer, { headers: response.headers }),
        )
      } catch {
        // ignore — see above
      }
    }
  }

  return bytes
}

async function readWithProgress(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  total: number,
  onProgress: LoadModelOptions['onProgress'],
): Promise<Uint8Array> {
  const chunks: Uint8Array[] = []
  let loaded = 0
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    if (!value) continue
    chunks.push(value)
    loaded += value.byteLength
    onProgress?.(loaded, total > 0 ? total : loaded)
  }
  // Final tick guarantees a 100% reading even if Content-Length was off.
  onProgress?.(loaded, total > 0 ? total : loaded)

  if (chunks.length === 1) return chunks[0]
  const out = new Uint8Array(loaded)
  let pos = 0
  for (const chunk of chunks) {
    out.set(chunk, pos)
    pos += chunk.byteLength
  }
  return out
}

async function openCacheSafe(): Promise<Cache | null> {
  // `caches` is undefined in older browsers, in some private modes, and
  // in non-window contexts (workers without it set up). Fail open: just
  // skip caching and let the model re-download next time.
  if (typeof caches === 'undefined') return null
  try {
    return await caches.open(CACHE_NAME)
  } catch {
    return null
  }
}

// Exposed for tests so we can clear the dedup map between cases. Not
// part of the public API; do not import from app code.
export function __resetInFlightForTests(): void {
  inFlight.clear()
}
