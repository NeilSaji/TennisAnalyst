import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// ---------------------------------------------------------------------------
// Integration coverage for the capture-quality telemetry helper that the
// analyze route fires after inserting an analysis_events row. Strategy:
//
// 1. Stub env + global fetch + the supabaseAdmin client BEFORE importing
//    lib/captureQuality (which reads env at module scope via
//    `export const RAILWAY_CONFIGURED = ...`).
// 2. Call classifyAndTagCaptureQuality(eventId, blobUrl) and then await a
//    microtask flush so the fire-and-forget promise chain completes before
//    assertions run.
// 3. Assert the Railway POST shape and the supabaseAdmin UPDATE payload.
//
// We deliberately test the helper directly instead of spinning up the full
// analyze route -- the route pulls in Anthropic, Supabase auth, and the
// MediaPipe summary pipeline, none of which are relevant to this telemetry
// wiring.
// ---------------------------------------------------------------------------

const MOCK_RAILWAY_URL = 'https://railway.test.example.com'
const MOCK_API_KEY = 'test-extract-key'
const MOCK_EVENT_ID = 'event-abc-123'
const MOCK_BLOB_URL = 'https://blob.vercel-storage.com/clip.mp4'

// Helpers the updateEq mock returns -- declared outside beforeEach so tests
// can inspect them across setup boundaries.
let updateSpy: ReturnType<typeof vi.fn>
let eqSpy: ReturnType<typeof vi.fn>
let fromSpy: ReturnType<typeof vi.fn>

function installSupabaseMock(eqResult: { error: unknown } = { error: null }) {
  eqSpy = vi.fn().mockResolvedValue(eqResult)
  updateSpy = vi.fn().mockReturnValue({ eq: eqSpy })
  fromSpy = vi.fn().mockReturnValue({ update: updateSpy })
  vi.doMock('@/lib/supabase', () => ({
    supabaseAdmin: { from: fromSpy },
    supabase: { from: vi.fn() },
  }))
}

// The background chain does: fetch -> resp.json() -> (optional) supabaseAdmin
// update -> .eq() resolves. Each `await` is at least one microtask; the
// Response.json() polyfill adds a couple more. Flushing 20 times is cheap
// insurance so assertions see the settled state.
async function flushMicrotasks(times = 20) {
  for (let i = 0; i < times; i++) {
    await Promise.resolve()
  }
}

describe('classifyAndTagCaptureQuality (analyze route telemetry)', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>
  let errorSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.resetModules()
    vi.unstubAllEnvs()
    vi.unstubAllGlobals()
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    vi.unstubAllGlobals()
    warnSpy.mockRestore()
    errorSpy.mockRestore()
  })

  it('POSTs to /classify-angle and UPDATEs analysis_events with the returned flag', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          capture_quality_flag: 'green_side',
          raw_label: 'side',
          samples_considered: 5,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    )
    vi.stubGlobal('fetch', fetchMock)

    installSupabaseMock()
    const { classifyAndTagCaptureQuality, RAILWAY_CONFIGURED } = await import(
      '@/lib/captureQuality'
    )
    expect(RAILWAY_CONFIGURED).toBe(true)

    classifyAndTagCaptureQuality(MOCK_EVENT_ID, MOCK_BLOB_URL)
    await flushMicrotasks()

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, options] = fetchMock.mock.calls[0]
    expect(url).toBe(`${MOCK_RAILWAY_URL}/classify-angle`)
    expect(options.method).toBe('POST')
    expect(options.headers['Content-Type']).toBe('application/json')
    expect(options.headers.Authorization).toBe(`Bearer ${MOCK_API_KEY}`)
    expect(JSON.parse(options.body)).toEqual({ video_url: MOCK_BLOB_URL })

    expect(fromSpy).toHaveBeenCalledWith('analysis_events')
    expect(updateSpy).toHaveBeenCalledWith({ capture_quality_flag: 'green_side' })
    expect(eqSpy).toHaveBeenCalledWith('id', MOCK_EVENT_ID)
  })

  it('writes red_front_or_back when Railway returns it', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            capture_quality_flag: 'red_front_or_back',
            raw_label: 'behind',
            samples_considered: 5,
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    )

    installSupabaseMock()
    const { classifyAndTagCaptureQuality } = await import('@/lib/captureQuality')

    classifyAndTagCaptureQuality(MOCK_EVENT_ID, MOCK_BLOB_URL)
    await flushMicrotasks()

    expect(updateSpy).toHaveBeenCalledWith({
      capture_quality_flag: 'red_front_or_back',
    })
  })

  it('does NOT call UPDATE when Railway fetch rejects (network error)', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)

    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('ECONNREFUSED')))

    installSupabaseMock()
    const { classifyAndTagCaptureQuality } = await import('@/lib/captureQuality')

    classifyAndTagCaptureQuality(MOCK_EVENT_ID, MOCK_BLOB_URL)
    await flushMicrotasks()

    expect(updateSpy).not.toHaveBeenCalled()
    expect(errorSpy).toHaveBeenCalled()
  })

  it('does NOT call UPDATE when Railway returns non-2xx', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response('server exploded', { status: 500 }),
      ),
    )

    installSupabaseMock()
    const { classifyAndTagCaptureQuality } = await import('@/lib/captureQuality')

    classifyAndTagCaptureQuality(MOCK_EVENT_ID, MOCK_BLOB_URL)
    await flushMicrotasks()

    expect(updateSpy).not.toHaveBeenCalled()
  })

  it('does NOT call UPDATE when the flag from Railway is not in the DB enum', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ capture_quality_flag: 'chartreuse' }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    )

    installSupabaseMock()
    const { classifyAndTagCaptureQuality } = await import('@/lib/captureQuality')

    classifyAndTagCaptureQuality(MOCK_EVENT_ID, MOCK_BLOB_URL)
    await flushMicrotasks()

    expect(updateSpy).not.toHaveBeenCalled()
    expect(errorSpy).toHaveBeenCalled()
  })

  it('is a no-op when blob_url is null (inline-keypoints analyze)', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)

    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    installSupabaseMock()
    const { classifyAndTagCaptureQuality } = await import('@/lib/captureQuality')

    classifyAndTagCaptureQuality(MOCK_EVENT_ID, null)
    await flushMicrotasks()

    expect(fetchMock).not.toHaveBeenCalled()
    expect(updateSpy).not.toHaveBeenCalled()
  })

  it('is a no-op when eventId is null', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)

    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    installSupabaseMock()
    const { classifyAndTagCaptureQuality } = await import('@/lib/captureQuality')

    classifyAndTagCaptureQuality(null, MOCK_BLOB_URL)
    await flushMicrotasks()

    expect(fetchMock).not.toHaveBeenCalled()
    expect(updateSpy).not.toHaveBeenCalled()
  })

  it('silently skips when RAILWAY_SERVICE_URL is missing (local dev)', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', '')
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)

    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    installSupabaseMock()
    const { classifyAndTagCaptureQuality, RAILWAY_CONFIGURED } = await import(
      '@/lib/captureQuality'
    )
    expect(RAILWAY_CONFIGURED).toBe(false)

    classifyAndTagCaptureQuality(MOCK_EVENT_ID, MOCK_BLOB_URL)
    await flushMicrotasks()

    expect(fetchMock).not.toHaveBeenCalled()
    expect(updateSpy).not.toHaveBeenCalled()
  })

  it('only warns once per process when env vars are missing', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', '')
    vi.stubEnv('EXTRACT_API_KEY', '')

    installSupabaseMock()
    const { classifyAndTagCaptureQuality } = await import('@/lib/captureQuality')

    classifyAndTagCaptureQuality(MOCK_EVENT_ID, MOCK_BLOB_URL)
    classifyAndTagCaptureQuality(MOCK_EVENT_ID, MOCK_BLOB_URL)
    classifyAndTagCaptureQuality(MOCK_EVENT_ID, MOCK_BLOB_URL)
    await flushMicrotasks()

    expect(warnSpy).toHaveBeenCalledTimes(1)
  })

  it('logs but does not throw when the DB UPDATE itself fails', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ capture_quality_flag: 'green_side' }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    )

    installSupabaseMock({ error: { message: 'RLS blocked' } })
    const { classifyAndTagCaptureQuality } = await import('@/lib/captureQuality')

    expect(() =>
      classifyAndTagCaptureQuality(MOCK_EVENT_ID, MOCK_BLOB_URL),
    ).not.toThrow()
    await flushMicrotasks()

    expect(updateSpy).toHaveBeenCalled()
    expect(errorSpy).toHaveBeenCalled()
  })
})
