import { NextRequest, NextResponse } from 'next/server'
import { timingSafeEqual } from 'node:crypto'

export const ADMIN_TOKEN_HEADER = 'x-clip-admin-token'

/**
 * Server-side guard for admin-only API routes. Returns null on success, or
 * a NextResponse the caller should return immediately.
 *
 *  - If `CLIP_ADMIN_PASSWORD` env is not set: 404 (opaque body — avoids
 *    leaking the existence of admin routes to unauthenticated callers).
 *  - If the header is missing or wrong: 401.
 *
 * Validation is constant-time so a wrong password can't be brute-forced
 * byte-by-byte from response timings.
 */
export function requireAdminAuth(request: NextRequest): NextResponse | null {
  if (!process.env.CLIP_ADMIN_PASSWORD) {
    return new NextResponse(null, { status: 404 })
  }
  const provided = request.headers.get(ADMIN_TOKEN_HEADER)
  if (!isAdminTokenValid(provided)) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }
  return null
}

/**
 * Constant-time admin-token check usable outside a request handler (e.g. to
 * validate a token embedded in a Vercel Blob clientPayload). Length-independent
 * so it doesn't leak the correct password length via early return.
 */
export function isAdminTokenValid(candidate: string | null | undefined): boolean {
  const configured = process.env.CLIP_ADMIN_PASSWORD
  if (!configured || !candidate) return false
  // Pad both to the same length so timingSafeEqual (which requires equal
  // lengths) can be used without branching on length first. We still XOR the
  // length difference into the final bit so a short input can never match.
  const a = Buffer.from(candidate, 'utf8')
  const b = Buffer.from(configured, 'utf8')
  const maxLen = Math.max(a.length, b.length)
  const aPad = Buffer.alloc(maxLen)
  const bPad = Buffer.alloc(maxLen)
  a.copy(aPad)
  b.copy(bPad)
  const bytesEqual = timingSafeEqual(aPad, bPad)
  const lengthsEqual = a.length === b.length
  return bytesEqual && lengthsEqual
}
