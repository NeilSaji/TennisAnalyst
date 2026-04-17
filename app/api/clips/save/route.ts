import { NextRequest, NextResponse } from 'next/server'
import { supabaseAdmin as supabase } from '@/lib/supabase'
import { requireAdminAuth } from '@/lib/adminAuth'
import { VALID_SHOT_TYPES, type ShotType } from '@/lib/shotTypeConfig'

const VALID_CAMERA_ANGLES = ['side', 'behind', 'front', 'court_level'] as const
const ALLOWED_SPEED_FACTORS = [1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4] as const
type CameraAngle = (typeof VALID_CAMERA_ANGLES)[number]

// The Vercel Blob public store hostname follows this shape; we pin it so an
// attacker can't submit a pro_swings row pointing at an arbitrary URL.
const ALLOWED_BLOB_HOST_SUFFIX = '.public.blob.vercel-storage.com'

function isAllowedSpeedFactor(n: number): boolean {
  return ALLOWED_SPEED_FACTORS.some((f) => Math.abs(n - f) < 1e-6)
}

function levenshtein(a: string, b: string): number {
  if (a === b) return 0
  if (!a.length) return b.length
  if (!b.length) return a.length
  const prev = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i++) {
    let last = prev[0]
    prev[0] = i
    for (let j = 1; j <= b.length; j++) {
      const tmp = prev[j]
      prev[j] =
        a[i - 1] === b[j - 1]
          ? last
          : 1 + Math.min(last, prev[j - 1], prev[j])
      last = tmp
    }
  }
  return prev[b.length]
}

function fuzzyThreshold(name: string): number {
  const len = name.trim().length
  if (len <= 4) return 0
  if (len <= 8) return 1
  return 2
}

type Body = {
  blobUrl?: unknown
  proName?: unknown
  nationality?: unknown
  shotType?: unknown
  cameraAngle?: unknown
  speedFactor?: unknown
  durationSec?: unknown
  sourceLabel?: unknown // free-form note about where the clip came from (e.g. local filename)
  confirmNewPro?: unknown
}

function validate(body: Body): string | null {
  if (typeof body.blobUrl !== 'string') return 'blobUrl must be a string'
  let parsed: URL
  try {
    parsed = new URL(body.blobUrl)
  } catch {
    return 'blobUrl must be a valid URL'
  }
  if (parsed.protocol !== 'https:') return 'blobUrl must be https'
  if (!parsed.hostname.endsWith(ALLOWED_BLOB_HOST_SUFFIX)) {
    return `blobUrl must be on ${ALLOWED_BLOB_HOST_SUFFIX}`
  }
  if (typeof body.proName !== 'string' || !body.proName.trim()) {
    return 'proName is required'
  }
  if (
    typeof body.shotType !== 'string' ||
    !VALID_SHOT_TYPES.includes(body.shotType as ShotType)
  ) {
    return `shotType must be one of: ${VALID_SHOT_TYPES.join(', ')}`
  }
  if (
    typeof body.cameraAngle !== 'string' ||
    !VALID_CAMERA_ANGLES.includes(body.cameraAngle as CameraAngle)
  ) {
    return `cameraAngle must be one of: ${VALID_CAMERA_ANGLES.join(', ')}`
  }
  if (body.speedFactor !== undefined) {
    if (typeof body.speedFactor !== 'number' || !isAllowedSpeedFactor(body.speedFactor)) {
      return 'speedFactor must be one of: 0.25, 0.333…, 0.5, 1, 2, 3, 4'
    }
  }
  if (body.durationSec !== undefined) {
    if (typeof body.durationSec !== 'number' || body.durationSec <= 0 || body.durationSec > 60) {
      return 'durationSec must be a positive number <= 60'
    }
  }
  if (body.nationality !== undefined && body.nationality !== null) {
    if (typeof body.nationality !== 'string' || body.nationality.length > 100) {
      return 'nationality must be a string under 100 chars'
    }
  }
  if (body.sourceLabel !== undefined && body.sourceLabel !== null) {
    if (typeof body.sourceLabel !== 'string' || body.sourceLabel.length > 200) {
      return 'sourceLabel must be a string under 200 chars'
    }
  }
  return null
}

export async function POST(request: NextRequest) {
  const guard = requireAdminAuth(request)
  if (guard) return guard

  let body: Body
  try {
    body = (await request.json()) as Body
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const err = validate(body)
  if (err) return NextResponse.json({ error: err }, { status: 400 })

  const blobUrl = body.blobUrl as string
  const rawProName = (body.proName as string).trim()
  const shotType = body.shotType as ShotType
  const cameraAngle = body.cameraAngle as CameraAngle
  const speedFactor = (body.speedFactor as number | undefined) ?? 1
  const durationSec = body.durationSec as number | undefined
  const nationality = (body.nationality as string | undefined)?.trim() || null
  const sourceLabel = (body.sourceLabel as string | undefined) ?? null
  const confirmNewPro = body.confirmNewPro === true

  // Reject pro names longer than 100 chars to avoid absurd DB rows.
  if (rawProName.length > 100) {
    return NextResponse.json({ error: 'proName too long' }, { status: 400 })
  }

  // Step 1: resolve existing pro by case-insensitive exact match, or flag
  // fuzzy duplicates so the UI can confirm before creating a new one.
  const { data: allPros, error: prosFetchErr } = await supabase
    .from('pros')
    .select('id, name')
  if (prosFetchErr) {
    console.error('[clips/save] Pros fetch error:', prosFetchErr.message)
    return NextResponse.json({ error: 'Failed to fetch pros' }, { status: 500 })
  }

  const exact = (allPros ?? []).find(
    (p) => p.name.trim().toLowerCase() === rawProName.toLowerCase(),
  )
  let resolvedPro: { id: string; name: string } | null = exact ?? null

  if (!resolvedPro && !confirmNewPro) {
    const threshold = fuzzyThreshold(rawProName)
    if (threshold > 0) {
      const suggestions = (allPros ?? [])
        .map((p) => ({
          name: p.name,
          dist: levenshtein(p.name.toLowerCase().trim(), rawProName.toLowerCase()),
        }))
        .filter((s) => s.dist > 0 && s.dist <= threshold)
        .sort((a, b) => a.dist - b.dist)
        .slice(0, 5)
        .map((s) => s.name)
      if (suggestions.length > 0) {
        return NextResponse.json(
          {
            error: `"${rawProName}" looks close to existing pro(s). Confirm this is a new pro or use an existing name.`,
            suggestions,
            code: 'fuzzy_match',
          },
          { status: 409 },
        )
      }
    }
  }

  if (!resolvedPro) {
    const { data: newPro, error: insertErr } = await supabase
      .from('pros')
      .insert({ name: rawProName, nationality })
      .select('id, name')
      .single()
    if (insertErr || !newPro) {
      console.error('[clips/save] Pro insert error:', insertErr?.message)
      return NextResponse.json({ error: 'Failed to create pro' }, { status: 500 })
    }
    resolvedPro = newPro
  }

  const durationMs = durationSec ? Math.round(durationSec * 1000) : null

  const { data: swing, error: swingErr } = await supabase
    .from('pro_swings')
    .insert({
      pro_id: resolvedPro.id,
      shot_type: shotType,
      video_url: blobUrl,
      keypoints_json: { fps_sampled: 30, frame_count: 0, frames: [] },
      fps: 30,
      duration_ms: durationMs,
      phase_labels: {},
      metadata: {
        camera_angle: cameraAngle,
        source: 'upload',
        speed_factor: speedFactor,
        source_label: sourceLabel,
        label: `${shotType}_${cameraAngle}`,
      },
    })
    .select('id, shot_type, video_url')
    .single()

  if (swingErr || !swing) {
    console.error('[clips/save] Swing insert error:', swingErr?.message)
    return NextResponse.json({ error: 'Failed to save swing' }, { status: 500 })
  }

  return NextResponse.json({
    success: true,
    pro: { id: resolvedPro.id, name: resolvedPro.name },
    swing: {
      id: swing.id,
      shot_type: swing.shot_type,
      video_url: swing.video_url,
    },
  })
}
