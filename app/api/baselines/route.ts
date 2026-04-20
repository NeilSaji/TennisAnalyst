import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

const VALID_SHOTS = ['forehand', 'backhand', 'serve', 'volley', 'slice'] as const
type ShotType = (typeof VALID_SHOTS)[number]

// The Vercel Blob public store hostname. We pin the origin so a malicious
// client can't store a baseline pointing at an attacker-controlled URL —
// otherwise any later viewer would fetch the "baseline video" from a
// hostile origin.
const ALLOWED_BLOB_HOST_SUFFIX = '.public.blob.vercel-storage.com'

function isShotType(v: unknown): v is ShotType {
  return typeof v === 'string' && (VALID_SHOTS as readonly string[]).includes(v)
}

function isAllowedBlobUrl(raw: string): boolean {
  try {
    const u = new URL(raw)
    return u.protocol === 'https:' && u.hostname.endsWith(ALLOWED_BLOB_HOST_SUFFIX)
  } catch {
    return false
  }
}

// GET /api/baselines — list baselines for the current auth user.
// RLS enforces ownership; we pass the user's session via the SSR client.
export async function GET() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) {
    return NextResponse.json({ error: 'Not signed in' }, { status: 401 })
  }

  const { data, error } = await supabase
    .from('user_baselines')
    .select('*')
    .order('is_active', { ascending: false })
    .order('created_at', { ascending: false })

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  return NextResponse.json({ baselines: data ?? [] })
}

// POST /api/baselines — create a baseline for the current user.
// Copies keypoints_json in-line so the source user_sessions row's 24h TTL
// cannot orphan the baseline later.
export async function POST(request: NextRequest) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) {
    return NextResponse.json({ error: 'Not signed in' }, { status: 401 })
  }

  let body
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const { blobUrl, shotType, keypointsJson, label, sourceSessionId } = body ?? {}

  if (typeof blobUrl !== 'string' || !blobUrl) {
    return NextResponse.json({ error: 'blobUrl is required' }, { status: 400 })
  }
  if (!isAllowedBlobUrl(blobUrl)) {
    return NextResponse.json(
      { error: `blobUrl must be an https URL on ${ALLOWED_BLOB_HOST_SUFFIX}` },
      { status: 400 },
    )
  }
  if (!isShotType(shotType)) {
    return NextResponse.json({ error: `shotType must be one of ${VALID_SHOTS.join(', ')}` }, { status: 400 })
  }
  if (!keypointsJson || typeof keypointsJson !== 'object' || !Array.isArray(keypointsJson.frames)) {
    return NextResponse.json({ error: 'keypointsJson.frames is required' }, { status: 400 })
  }
  if (keypointsJson.frames.length === 0) {
    return NextResponse.json({ error: 'keypointsJson.frames cannot be empty' }, { status: 400 })
  }

  // Flip any existing active baseline of the same shot type to inactive.
  // Serial rather than transactional — single-user race is not a practical concern.
  const now = new Date().toISOString()
  const { error: deactivateErr } = await supabase
    .from('user_baselines')
    .update({ is_active: false, replaced_at: now })
    .eq('shot_type', shotType)
    .eq('is_active', true)

  if (deactivateErr) {
    return NextResponse.json({ error: deactivateErr.message }, { status: 500 })
  }

  const insertRow: Record<string, unknown> = {
    user_id: user.id,
    label: typeof label === 'string' && label.trim() ? label.trim().slice(0, 120) : 'My baseline',
    shot_type: shotType,
    blob_url: blobUrl,
    keypoints_json: keypointsJson,
    is_active: true,
  }
  if (typeof sourceSessionId === 'string' && /^[0-9a-f-]{36}$/i.test(sourceSessionId)) {
    insertRow.source_session_id = sourceSessionId
  }

  const { data, error } = await supabase
    .from('user_baselines')
    .insert(insertRow)
    .select()
    .single()

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  return NextResponse.json({ baseline: data })
}
