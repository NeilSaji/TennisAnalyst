import { NextRequest, NextResponse } from 'next/server'
import { supabaseAdmin as supabase } from '@/lib/supabase'
import { put } from '@vercel/blob'
import { execFileSync } from 'child_process'
import { mkdtempSync, unlinkSync, existsSync, readFileSync } from 'fs'
import { join } from 'path'
import os from 'os'
import { VALID_SHOT_TYPES, type ShotType } from '@/lib/shotTypeConfig'

const VALID_CAMERA_ANGLES = ['side', 'behind', 'front', 'court_level'] as const
const ALLOWED_SPEED_FACTORS = [1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4] as const

type CameraAngle = (typeof VALID_CAMERA_ANGLES)[number]

const YOUTUBE_URL_PATTERN = /^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\//

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

// Small helper: names are considered a "fuzzy duplicate" if they differ by
// at most `maxDist` characters (case-insensitive). Threshold scales down for
// short strings so "Tim" and "Tom" don't collide.
function fuzzyMatchDistance(a: string, b: string): number {
  return levenshtein(a.toLowerCase().trim(), b.toLowerCase().trim())
}

function fuzzyThreshold(name: string): number {
  const len = name.trim().length
  if (len <= 4) return 0
  if (len <= 8) return 1
  return 2
}

function sanitizeFilename(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')
}

function validateBody(body: Record<string, unknown>): string | null {
  const { youtubeUrl, startTime, endTime, proName, shotType, cameraAngle, speedFactor } =
    body

  if (!youtubeUrl || typeof youtubeUrl !== 'string') {
    return 'youtubeUrl is required and must be a string'
  }
  if (!YOUTUBE_URL_PATTERN.test(youtubeUrl)) {
    return 'youtubeUrl must be a valid YouTube URL'
  }

  if (typeof startTime !== 'number' || isNaN(startTime)) {
    return 'startTime is required and must be a number'
  }
  if (typeof endTime !== 'number' || isNaN(endTime)) {
    return 'endTime is required and must be a number'
  }
  if (endTime <= startTime) {
    return 'endTime must be greater than startTime'
  }

  if (!proName || typeof proName !== 'string') {
    return 'proName is required and must be a string'
  }

  if (
    !shotType ||
    typeof shotType !== 'string' ||
    !VALID_SHOT_TYPES.includes(shotType as ShotType)
  ) {
    return `shotType must be one of: ${VALID_SHOT_TYPES.join(', ')}`
  }

  if (
    !cameraAngle ||
    typeof cameraAngle !== 'string' ||
    !VALID_CAMERA_ANGLES.includes(cameraAngle as CameraAngle)
  ) {
    return `cameraAngle must be one of: ${VALID_CAMERA_ANGLES.join(', ')}`
  }

  if (speedFactor !== undefined) {
    if (typeof speedFactor !== 'number' || !isAllowedSpeedFactor(speedFactor)) {
      return 'speedFactor must be one of: 0.25, 0.333…, 0.5, 1, 2, 3, 4'
    }
  }

  return null
}

export async function POST(request: NextRequest) {
  if (process.env.NEXT_PUBLIC_ADMIN_ENABLED !== 'true') {
    return new NextResponse(null, { status: 404 })
  }

  let body: Record<string, unknown>
  try {
    body = await request.json()
  } catch {
    return NextResponse.json(
      { error: 'Invalid JSON in request body' },
      { status: 400 }
    )
  }

  const validationError = validateBody(body)
  if (validationError) {
    return NextResponse.json({ error: validationError }, { status: 400 })
  }

  const {
    youtubeUrl,
    startTime,
    endTime,
    proName,
    nationality,
    shotType,
    cameraAngle,
    speedFactor: rawSpeedFactor,
    confirmNewPro,
  } = body as {
    youtubeUrl: string
    startTime: number
    endTime: number
    proName: string
    nationality?: string
    shotType: ShotType
    cameraAngle: CameraAngle
    speedFactor?: number
    confirmNewPro?: boolean
  }

  const speedFactor = rawSpeedFactor ?? 1
  const sourceDuration = endTime - startTime
  const outputDuration = sourceDuration / speedFactor
  const trimmedProName = proName.trim()
  const projectRoot = process.cwd()
  const ffmpegBin = join(projectRoot, 'pro-videos', 'bin', 'ffmpeg')

  // Step 0: Resolve the pro row BEFORE downloading anything — saves the user
  // from waiting 30–60s on yt-dlp just to hit a name-typo rejection.
  const { data: allPros, error: prosFetchErr } = await supabase
    .from('pros')
    .select('id, name')
  if (prosFetchErr) {
    console.error('[tag-clip] Pros fetch error:', prosFetchErr.message)
    return NextResponse.json(
      { error: 'Failed to fetch pros' },
      { status: 500 }
    )
  }

  const exactMatch = (allPros ?? []).find(
    (p) => p.name.trim().toLowerCase() === trimmedProName.toLowerCase()
  )

  const resolvedPro: { id: string; name: string } | null = exactMatch ?? null

  if (!resolvedPro && !confirmNewPro) {
    const threshold = fuzzyThreshold(trimmedProName)
    if (threshold > 0) {
      const suggestions = (allPros ?? [])
        .map((p) => ({ name: p.name, dist: fuzzyMatchDistance(trimmedProName, p.name) }))
        .filter((s) => s.dist > 0 && s.dist <= threshold)
        .sort((a, b) => a.dist - b.dist)
        .slice(0, 5)
        .map((s) => s.name)
      if (suggestions.length > 0) {
        return NextResponse.json(
          {
            error: `"${trimmedProName}" looks close to existing pro(s). Confirm this is a new pro or use an existing name.`,
            suggestions,
            code: 'fuzzy_match',
          },
          { status: 409 }
        )
      }
    }
  }

  const tmpDir = mkdtempSync(join(os.tmpdir(), 'tag-clip-'))
  const tmpFullVideo = join(tmpDir, 'full.mp4')
  const sanitizedName = sanitizeFilename(resolvedPro?.name ?? trimmedProName)
  const filename = `${sanitizedName}_${shotType}_${cameraAngle}_${Date.now()}.mp4`
  const trimmedPath = join(tmpDir, 'trimmed.mp4')

  try {
    // Step 1: Download ONLY the needed segment with yt-dlp --download-sections
    // Tell yt-dlp where our local ffmpeg is so it can do partial downloads.
    const ffmpegDir = join(projectRoot, 'pro-videos', 'bin')
    const sectionArg = `*${startTime}-${endTime}`
    console.log('[tag-clip] Downloading segment:', sectionArg, 'from:', youtubeUrl)
    // Quality: sort by resolution first, then prefer H.264/MP4 when available.
    // This picks the highest-res stream YouTube offers (often 1080p60 VP9 or
    // 4K AV1) and only transcodes when the top-res stream isn't already H.264.
    // The old `[ext=mp4]` filter pinned us to a lower-res MP4 fallback.
    execFileSync(
      'yt-dlp',
      [
        '-f',
        'bestvideo+bestaudio/best',
        '-S',
        'res,ext:mp4:m4a,codec:avc,fps',
        '--merge-output-format',
        'mp4',
        '--no-playlist',
        '--download-sections',
        sectionArg,
        '--force-keyframes-at-cuts',
        '--ffmpeg-location',
        ffmpegDir,
        '-o',
        tmpFullVideo,
        youtubeUrl,
      ],
      { stdio: 'pipe', timeout: 300_000 }
    )

    // Step 2: Finalize clip.
    //  - speedFactor === 1: stream copy (lossless) + faststart
    //  - otherwise: re-encode with setpts filter, drop audio (sped/slowed audio is noise)
    if (speedFactor === 1) {
      console.log('[tag-clip] Stream-copying clip (no speed change)')
      execFileSync(
        ffmpegBin,
        [
          '-y',
          '-i',
          tmpFullVideo,
          '-c',
          'copy',
          '-movflags',
          '+faststart',
          trimmedPath,
        ],
        { stdio: 'pipe', timeout: 60_000 }
      )
    } else {
      console.log(`[tag-clip] Re-encoding clip at speedFactor=${speedFactor}`)
      execFileSync(
        ffmpegBin,
        [
          '-y',
          '-i',
          tmpFullVideo,
          '-filter:v',
          `setpts=${1 / speedFactor}*PTS`,
          '-an',
          '-c:v',
          'libx264',
          '-preset',
          'slow',
          '-crf',
          '18',
          '-movflags',
          '+faststart',
          trimmedPath,
        ],
        { stdio: 'pipe', timeout: 120_000 }
      )
    }

    // Step 3: Upload trimmed clip to Vercel Blob so it plays on any device.
    console.log('[tag-clip] Uploading to Vercel Blob:', filename)
    const clipBuffer = readFileSync(trimmedPath)
    const blob = await put(`pro-videos/${filename}`, clipBuffer, {
      access: 'public',
      contentType: 'video/mp4',
    })
    const videoUrl = blob.url

    // Step 4: Resolve or create the pro row.
    // Existing pros were matched in Step 0; here we only have to create new ones.
    let pro: { id: string; name: string }
    if (resolvedPro) {
      pro = resolvedPro
    } else {
      const { data: newPro, error: insertError } = await supabase
        .from('pros')
        .insert({ name: trimmedProName, nationality: nationality ?? null })
        .select('id, name')
        .single()

      if (insertError || !newPro) {
        console.error('[tag-clip] Pro insert error:', insertError?.message)
        throw new Error('Failed to create pro')
      }
      pro = newPro
    }

    // Step 5: Insert the pro_swing
    const durationMs = Math.round(outputDuration * 1000)

    const { data: swing, error: swingError } = await supabase
      .from('pro_swings')
      .insert({
        pro_id: pro.id,
        shot_type: shotType,
        video_url: videoUrl,
        keypoints_json: { fps_sampled: 30, frame_count: 0, frames: [] },
        fps: 30,
        duration_ms: durationMs,
        phase_labels: {},
        metadata: {
          camera_angle: cameraAngle,
          source: 'youtube',
          original_url: youtubeUrl,
          start_time_sec: startTime,
          end_time_sec: endTime,
          speed_factor: speedFactor,
          label: `${shotType}_${cameraAngle}`,
        },
      })
      .select('id, shot_type, video_url')
      .single()

    if (swingError || !swing) {
      console.error('[tag-clip] Swing insert error:', swingError?.message)
      throw new Error('Failed to create pro swing')
    }

    console.log('[tag-clip] Successfully created swing:', swing.id)

    // Step 6: Auto-extract keypoints from the local tmp clip (still on disk
    // until the finally block runs). Reading from blob would re-download.
    let frameCount = 0
    try {
      const scriptPath = join(projectRoot, 'railway-service', 'extract_clip_keypoints.py')
      if (existsSync(scriptPath) && existsSync(trimmedPath)) {
        console.log('[tag-clip] Extracting keypoints...')
        const kpOutput = execFileSync(
          'arch',
          ['-arm64', 'python3', scriptPath, trimmedPath],
          { timeout: 120_000, maxBuffer: 50 * 1024 * 1024 }
        )
        const keypoints = JSON.parse(kpOutput.toString())
        if (keypoints.frame_count > 0) {
          await supabase
            .from('pro_swings')
            .update({
              keypoints_json: keypoints,
              frame_count: keypoints.frame_count,
              duration_ms: keypoints.duration_ms,
              fps: keypoints.fps_sampled,
            })
            .eq('id', swing.id)
          frameCount = keypoints.frame_count
          console.log(`[tag-clip] Extracted ${frameCount} frames`)
        }
      }
    } catch (kpErr) {
      console.warn('[tag-clip] Keypoint extraction failed (clip still saved):', kpErr instanceof Error ? kpErr.message : kpErr)
    }

    return NextResponse.json({
      success: true,
      pro: { id: pro.id, name: pro.name },
      swing: {
        id: swing.id,
        shot_type: swing.shot_type,
        video_url: swing.video_url,
      },
      frameCount,
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    console.error('[tag-clip] Processing failed:', message)
    return NextResponse.json({ error: message }, { status: 500 })
  } finally {
    // Clean up temp files
    try {
      if (existsSync(tmpFullVideo)) unlinkSync(tmpFullVideo)
    } catch {
      // Temp file cleanup is best-effort
    }
    try {
      if (existsSync(trimmedPath)) unlinkSync(trimmedPath)
    } catch {
      // Temp file cleanup is best-effort
    }
    try {
      if (existsSync(tmpDir)) {
        execFileSync('rm', ['-rf', tmpDir], { stdio: 'pipe' })
      }
    } catch {
      // Temp dir cleanup is best-effort
    }
  }
}
