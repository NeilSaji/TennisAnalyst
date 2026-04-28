import { describe, it, expect, beforeEach, vi } from 'vitest'
import { NextRequest } from 'next/server'

// --- Supabase mock harness ------------------------------------------------
//
// Mirrors the chained-builder shape used in sessions.test.ts. We track every
// .insert() / .update() call so the tests can assert that swings are
// persisted verbatim — i.e. the route does not re-run detectSwings server-
// side, it just writes the rows the client sent.

const mockSelect = vi.fn()
const mockSingle = vi.fn()
const mockEq = vi.fn()
const mockIn = vi.fn()
const mockUpdate = vi.fn()
const mockInsert = vi.fn()
const mockUpsert = vi.fn()
const mockFrom = vi.fn()

function chainMock() {
  // Clear call history but keep the mock implementations from accumulating
  // mockResolvedValueOnce queues across tests.
  mockSelect.mockClear()
  mockSingle.mockClear()
  mockEq.mockClear()
  mockIn.mockClear()
  mockUpdate.mockClear()
  mockInsert.mockClear()
  mockUpsert.mockClear()
  mockFrom.mockClear()
  // Also drop any leftover impls (the *Once stacks etc.).
  for (const m of [mockSelect, mockSingle, mockEq, mockIn, mockUpdate, mockInsert, mockUpsert, mockFrom]) {
    m.mockReset()
  }
  mockSelect.mockReturnThis()
  mockSingle.mockReturnThis()
  mockEq.mockReturnThis()
  mockIn.mockReturnThis()
  mockUpdate.mockReturnThis()
  mockInsert.mockReturnThis()
  mockUpsert.mockReturnThis()
  mockFrom.mockReturnValue({
    select: mockSelect,
    single: mockSingle,
    eq: mockEq,
    in: mockIn,
    update: mockUpdate,
    insert: mockInsert,
    upsert: mockUpsert,
  })
  mockUpdate.mockReturnValue({ eq: mockEq, in: mockIn })
  mockEq.mockReturnValue({ eq: mockEq, in: mockIn, select: mockSelect, single: mockSingle })
  mockIn.mockReturnValue({ eq: mockEq, select: mockSelect })
  mockSelect.mockReturnValue({ single: mockSingle, eq: mockEq })
  mockUpsert.mockReturnValue({ select: mockSelect })
  mockInsert.mockReturnValue({ select: mockSelect })
}

vi.mock('@/lib/supabase', () => ({
  supabaseAdmin: new Proxy({}, {
    get: (_t: object, prop: string) => {
      if (prop === 'from') return mockFrom
      return undefined
    },
  }),
}))

// Auth client returns a signed-in user for every test in this suite.
// We control the user shape per-test via mockGetUser so the 401 case can
// flip it to a null user without re-mocking the whole module.
const mockGetUser = vi.fn<() => Promise<{ data: { user: { id: string } | null } }>>(
  async () => ({ data: { user: { id: 'user-1' } } }),
)
vi.mock('@/lib/supabase/server', () => ({
  createClient: async () => ({
    auth: { getUser: mockGetUser },
  }),
}))

// --- Tests ----------------------------------------------------------------

const VALID_BLOB = 'https://abc.public.blob.vercel-storage.com/live/x.webm'

function makeRequest(body: Record<string, unknown>) {
  return new NextRequest('http://localhost/api/sessions/live', {
    method: 'POST',
    body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json' },
  })
}

const baseValidBody = {
  blobUrl: VALID_BLOB,
  shotType: 'forehand',
  keypointsJson: { fps_sampled: 15, frame_count: 0, frames: [] },
  swings: [
    { startFrame: 0, endFrame: 12, startMs: 0, endMs: 800 },
    { startFrame: 30, endFrame: 45, startMs: 2000, endMs: 3000 },
    { startFrame: 70, endFrame: 90, startMs: 4666, endMs: 6000 },
  ],
  batchEventIds: [],
}

describe('POST /api/sessions/live', () => {
  beforeEach(() => {
    chainMock()
    // Reset the auth mock to the signed-in default before every test.
    mockGetUser.mockReset()
    mockGetUser.mockResolvedValue({ data: { user: { id: 'user-1' } } })
  })

  it('returns 401 when the user is not signed in', async () => {
    mockGetUser.mockResolvedValueOnce({ data: { user: null } })
    const { POST } = await import('@/app/api/sessions/live/route')
    const res = await POST(makeRequest(baseValidBody))
    expect(res.status).toBe(401)
  })

  it('returns 400 when blobUrl is not on the vercel-blob host', async () => {
    const { POST } = await import('@/app/api/sessions/live/route')
    const res = await POST(
      makeRequest({ ...baseValidBody, blobUrl: 'https://evil.example.com/x.mp4' }),
    )
    expect(res.status).toBe(400)
    const body = await res.json()
    expect(body.error).toMatch(/blobUrl/i)
  })

  it('returns 400 when shotType is invalid', async () => {
    const { POST } = await import('@/app/api/sessions/live/route')
    const res = await POST(
      makeRequest({ ...baseValidBody, shotType: 'lob' }),
    )
    expect(res.status).toBe(400)
  })

  it('returns 400 when keypointsJson.frames is missing', async () => {
    const { POST } = await import('@/app/api/sessions/live/route')
    const res = await POST(
      makeRequest({ ...baseValidBody, keypointsJson: { fps_sampled: 15 } }),
    )
    expect(res.status).toBe(400)
  })

  it('persists the swings list verbatim — no re-running of detectSwings', async () => {
    // 1) user_sessions upsert chain: from().upsert().select().single()
    //    Build it from scratch so we don't fight chainMock's defaults.
    let capturedUpsertArgs: Record<string, unknown> | null = null
    let capturedInsertArgs: Array<Record<string, unknown>> | null = null
    mockFrom.mockImplementation((table: string) => {
      if (table === 'user_sessions') {
        return {
          upsert: (args: Record<string, unknown>) => {
            capturedUpsertArgs = args
            return {
              select: () => ({
                single: async () => ({
                  data: { id: 'sess-live-1' },
                  error: null,
                }),
              }),
            }
          },
        }
      }
      if (table === 'video_segments') {
        return {
          insert: (args: Array<Record<string, unknown>>) => {
            capturedInsertArgs = args
            return {
              select: async () => ({
                data: [
                  { id: 'seg-1', segment_index: 1 },
                  { id: 'seg-2', segment_index: 2 },
                  { id: 'seg-3', segment_index: 3 },
                ],
                error: null,
              }),
            }
          },
        }
      }
      return {}
    })

    const { POST } = await import('@/app/api/sessions/live/route')
    const res = await POST(makeRequest(baseValidBody))
    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body.sessionId).toBe('sess-live-1')
    expect(body.segmentCount).toBe(3)

    // The video_segments insert MUST have received exactly the 3 swings the
    // client sent — same start_frame / end_frame / start_ms / end_ms — and
    // no others. This is the "no re-running of detectSwings" guarantee.
    expect(capturedInsertArgs).not.toBeNull()
    const segments = capturedInsertArgs as unknown as Array<{
      session_id: string
      segment_index: number
      shot_type: string
      start_frame: number
      end_frame: number
      start_ms: number
      end_ms: number
    }>
    expect(segments).toHaveLength(3)
    expect(segments[0]).toMatchObject({
      session_id: 'sess-live-1',
      segment_index: 1,
      shot_type: 'forehand',
      start_frame: 0,
      end_frame: 12,
      start_ms: 0,
      end_ms: 800,
    })
    expect(segments[1]).toMatchObject({
      segment_index: 2,
      start_frame: 30,
      end_frame: 45,
      start_ms: 2000,
      end_ms: 3000,
    })
    expect(segments[2]).toMatchObject({
      segment_index: 3,
      start_frame: 70,
      end_frame: 90,
      start_ms: 4666,
      end_ms: 6000,
    })

    // The user_sessions upsert carries blob_url + session_mode='live' and
    // segment_count matching the swings count.
    expect(capturedUpsertArgs).not.toBeNull()
    const upsertArg = capturedUpsertArgs as unknown as Record<string, unknown>
    expect(upsertArg.blob_url).toBe(VALID_BLOB)
    expect(upsertArg.session_mode).toBe('live')
    expect(upsertArg.segment_count).toBe(3)
    expect(upsertArg.is_multi_shot).toBe(true)
  })

  it('accepts an empty swings list (zero-swing session)', async () => {
    let videoSegmentInsertCalled = false
    mockFrom.mockImplementation((table: string) => {
      if (table === 'user_sessions') {
        return {
          upsert: () => ({
            select: () => ({
              single: async () => ({
                data: { id: 'sess-empty' },
                error: null,
              }),
            }),
          }),
        }
      }
      if (table === 'video_segments') {
        videoSegmentInsertCalled = true
        return {
          insert: () => ({ select: async () => ({ data: [], error: null }) }),
        }
      }
      return {}
    })

    const { POST } = await import('@/app/api/sessions/live/route')
    const res = await POST(
      makeRequest({ ...baseValidBody, swings: [] }),
    )
    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body.segmentCount).toBe(0)
    // No video_segments insert when the client sent zero swings — the route
    // skips the insert path entirely for empty swing lists.
    expect(videoSegmentInsertCalled).toBe(false)
  })
})
