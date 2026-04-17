import { NextRequest, NextResponse } from 'next/server'
import { requireAdminAuth } from '@/lib/adminAuth'

// Per-IP sliding-window rate limit for the password-verify endpoint.
// Module-scope Map so state survives between invocations on the same serverless
// instance. Different instances won't share state (Vercel runs multiple), but
// this still meaningfully slows a brute-forcer and is zero-dependency.
const WINDOW_MS = 60_000
const MAX_ATTEMPTS_PER_WINDOW = 15
const attemptLog = new Map<string, number[]>()

function isRateLimited(ip: string): boolean {
  const now = Date.now()
  const cutoff = now - WINDOW_MS
  const prior = attemptLog.get(ip) ?? []
  const fresh = prior.filter((t) => t > cutoff)
  fresh.push(now)
  attemptLog.set(ip, fresh)
  // Best-effort cleanup to keep the Map from growing forever
  if (attemptLog.size > 10_000) {
    for (const [k, v] of attemptLog) {
      if (v.every((t) => t <= cutoff)) attemptLog.delete(k)
    }
  }
  return fresh.length > MAX_ATTEMPTS_PER_WINDOW
}

function clientIp(request: NextRequest): string {
  const fwd = request.headers.get('x-forwarded-for')
  if (fwd) {
    // Format: "client, proxy1, proxy2". Take the leftmost.
    return fwd.split(',')[0].trim()
  }
  return request.headers.get('x-real-ip') ?? 'unknown'
}

export async function GET(request: NextRequest) {
  // Rate-limit BEFORE doing the token compare so a loop of requests can't
  // keep the CPU busy comparing tokens even after the limit kicks in.
  const ip = clientIp(request)
  if (isRateLimited(ip)) {
    return NextResponse.json(
      { error: 'Too many attempts. Try again later.' },
      { status: 429 },
    )
  }
  const guard = requireAdminAuth(request)
  if (guard) return guard
  return NextResponse.json({ ok: true })
}
