import { describe, it, expect, vi, beforeEach } from 'vitest'
import { NextRequest } from 'next/server'

// We mock the server-side Supabase factory. Every test hands it a tailored
// fake client so we can exercise the 401 / 400 / 404 / 200 branches without
// hitting a real DB. The RLS-denial path is simulated by making the
// `.update(...).eq(...).select(...).single()` chain resolve with an error —
// matching the "no row visible to this user" outcome of a real RLS denial.

type FakeUser = { id: string } | null

interface FakeClient {
  auth: {
    getUser: () => Promise<{ data: { user: FakeUser }; error: { message: string } | null }>
  }
  from: (table: string) => unknown
}

let nextClient: FakeClient

vi.mock('@/lib/supabase/server', () => ({
  createClient: vi.fn(async () => nextClient),
}))

function makeJsonRequest(body: unknown): NextRequest {
  return new NextRequest('http://localhost/api/analysis-events/event-1/feedback', {
    method: 'PATCH',
    body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json' },
  })
}

function makeRawRequest(raw: string): NextRequest {
  return new NextRequest('http://localhost/api/analysis-events/event-1/feedback', {
    method: 'PATCH',
    body: raw,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('PATCH /api/analysis-events/[id]/feedback', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('returns 401 when the request is anonymous', async () => {
    nextClient = {
      auth: {
        getUser: async () => ({ data: { user: null }, error: null }),
      },
      from: () => {
        throw new Error('should not hit DB when anon')
      },
    }

    const { PATCH } = await import('@/app/api/analysis-events/[id]/feedback/route')
    const req = makeJsonRequest({ correction: 'correct' })
    const res = await PATCH(req, { params: Promise.resolve({ id: 'event-1' }) })
    expect(res.status).toBe(401)
  })

  it('returns 400 when the body is not JSON', async () => {
    nextClient = {
      auth: {
        getUser: async () => ({ data: { user: { id: 'user-1' } }, error: null }),
      },
      from: () => {
        throw new Error('should not hit DB when body invalid')
      },
    }
    const { PATCH } = await import('@/app/api/analysis-events/[id]/feedback/route')
    const req = makeRawRequest('not json')
    const res = await PATCH(req, { params: Promise.resolve({ id: 'event-1' }) })
    expect(res.status).toBe(400)
  })

  it('returns 400 when correction is missing', async () => {
    nextClient = {
      auth: {
        getUser: async () => ({ data: { user: { id: 'user-1' } }, error: null }),
      },
      from: () => {
        throw new Error('should not hit DB when correction missing')
      },
    }
    const { PATCH } = await import('@/app/api/analysis-events/[id]/feedback/route')
    const req = makeJsonRequest({})
    const res = await PATCH(req, { params: Promise.resolve({ id: 'event-1' }) })
    expect(res.status).toBe(400)
  })

  it('returns 400 when correction is an unknown enum value', async () => {
    nextClient = {
      auth: {
        getUser: async () => ({ data: { user: { id: 'user-1' } }, error: null }),
      },
      from: () => {
        throw new Error('should not hit DB when correction invalid')
      },
    }
    const { PATCH } = await import('@/app/api/analysis-events/[id]/feedback/route')
    const req = makeJsonRequest({ correction: 'absolutely_perfect' })
    const res = await PATCH(req, { params: Promise.resolve({ id: 'event-1' }) })
    expect(res.status).toBe(400)
  })

  it('returns 400 when note is present but not a string', async () => {
    nextClient = {
      auth: {
        getUser: async () => ({ data: { user: { id: 'user-1' } }, error: null }),
      },
      from: () => {
        throw new Error('should not hit DB when note invalid')
      },
    }
    const { PATCH } = await import('@/app/api/analysis-events/[id]/feedback/route')
    const req = makeJsonRequest({ correction: 'correct', note: 42 })
    const res = await PATCH(req, { params: Promise.resolve({ id: 'event-1' }) })
    expect(res.status).toBe(400)
  })

  it('returns 404 when RLS blocks the update (non-owner or missing row)', async () => {
    const single = vi.fn().mockResolvedValue({
      data: null,
      error: { message: 'Row not found', code: 'PGRST116' },
    })
    const select = vi.fn().mockReturnValue({ single })
    const eq = vi.fn().mockReturnValue({ select })
    const update = vi.fn().mockReturnValue({ eq })

    nextClient = {
      auth: {
        getUser: async () => ({ data: { user: { id: 'user-1' } }, error: null }),
      },
      from: vi.fn().mockReturnValue({ update }),
    }

    const { PATCH } = await import('@/app/api/analysis-events/[id]/feedback/route')
    const req = makeJsonRequest({ correction: 'too_hard' })
    const res = await PATCH(req, { params: Promise.resolve({ id: 'event-1' }) })
    expect(res.status).toBe(404)
    expect(update).toHaveBeenCalledWith(
      expect.objectContaining({ user_correction: 'too_hard' }),
    )
  })

  it('returns 200 and writes the correction when the caller owns the row', async () => {
    const single = vi.fn().mockResolvedValue({ data: { id: 'event-1' }, error: null })
    const select = vi.fn().mockReturnValue({ single })
    const eq = vi.fn().mockReturnValue({ select })
    const update = vi.fn().mockReturnValue({ eq })
    const from = vi.fn().mockReturnValue({ update })

    nextClient = {
      auth: {
        getUser: async () => ({ data: { user: { id: 'user-1' } }, error: null }),
      },
      from,
    }

    const { PATCH } = await import('@/app/api/analysis-events/[id]/feedback/route')
    const req = makeJsonRequest({ correction: 'correct', note: 'felt right on' })
    const res = await PATCH(req, { params: Promise.resolve({ id: 'event-1' }) })
    expect(res.status).toBe(200)

    const body = await res.json()
    expect(body).toEqual({ ok: true })

    expect(from).toHaveBeenCalledWith('analysis_events')
    expect(update).toHaveBeenCalledWith({
      user_correction: 'correct',
      user_correction_note: 'felt right on',
    })
    expect(eq).toHaveBeenCalledWith('id', 'event-1')
  })

  it('sanitizes the note (strips control chars) and caps at 200 chars', async () => {
    const single = vi.fn().mockResolvedValue({ data: { id: 'event-1' }, error: null })
    const select = vi.fn().mockReturnValue({ single })
    const eq = vi.fn().mockReturnValue({ select })
    const update = vi.fn().mockReturnValue({ eq })
    const from = vi.fn().mockReturnValue({ update })

    nextClient = {
      auth: {
        getUser: async () => ({ data: { user: { id: 'user-1' } }, error: null }),
      },
      from,
    }

    const longNote = 'a'.repeat(500)
    const noteWithControls = `hello\x00world\n${longNote}`

    const { PATCH } = await import('@/app/api/analysis-events/[id]/feedback/route')
    const req = makeJsonRequest({ correction: 'too_easy', note: noteWithControls })
    const res = await PATCH(req, { params: Promise.resolve({ id: 'event-1' }) })
    expect(res.status).toBe(200)

    const call = update.mock.calls[0][0]
    expect(call.user_correction_note).not.toMatch(/\x00/)
    expect(call.user_correction_note.length).toBeLessThanOrEqual(200)
  })
})
