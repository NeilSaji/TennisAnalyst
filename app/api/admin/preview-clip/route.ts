import { NextRequest, NextResponse } from 'next/server'
import { execFileSync } from 'child_process'
import {
  mkdtempSync,
  existsSync,
  mkdirSync,
  renameSync,
  readdirSync,
  unlinkSync,
  statSync,
} from 'fs'
import { join } from 'path'
import os from 'os'
import { randomUUID } from 'crypto'

export const runtime = 'nodejs'

const ALLOWED_SPEED_FACTORS = [1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4] as const
const YOUTUBE_URL_PATTERN = /^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\//
const MAX_SOURCE_DURATION_SEC = 60
const PREVIEW_MAX_AGE_MS = 30 * 60 * 1000

function isAllowedSpeedFactor(n: number): boolean {
  return ALLOWED_SPEED_FACTORS.some((f) => Math.abs(n - f) < 1e-6)
}

function cleanOldPreviews(dir: string) {
  if (!existsSync(dir)) return
  const now = Date.now()
  for (const entry of readdirSync(dir)) {
    if (!entry.endsWith('.mp4')) continue
    const full = join(dir, entry)
    try {
      const st = statSync(full)
      if (now - st.mtimeMs > PREVIEW_MAX_AGE_MS) {
        unlinkSync(full)
      }
    } catch {
      // best-effort cleanup
    }
  }
}

export async function POST(request: NextRequest) {
  if (process.env.NEXT_PUBLIC_ADMIN_ENABLED !== 'true') {
    return new NextResponse(null, { status: 404 })
  }

  let body: Record<string, unknown>
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON in request body' }, { status: 400 })
  }

  const { youtubeUrl, startTime, endTime, speedFactor: rawSpeedFactor } = body as {
    youtubeUrl?: unknown
    startTime?: unknown
    endTime?: unknown
    speedFactor?: unknown
  }

  if (typeof youtubeUrl !== 'string' || !YOUTUBE_URL_PATTERN.test(youtubeUrl)) {
    return NextResponse.json({ error: 'youtubeUrl must be a valid YouTube URL' }, { status: 400 })
  }
  if (typeof startTime !== 'number' || isNaN(startTime) || startTime < 0) {
    return NextResponse.json({ error: 'startTime must be a non-negative number' }, { status: 400 })
  }
  if (typeof endTime !== 'number' || isNaN(endTime) || endTime <= startTime) {
    return NextResponse.json({ error: 'endTime must be greater than startTime' }, { status: 400 })
  }
  if (endTime - startTime > MAX_SOURCE_DURATION_SEC) {
    return NextResponse.json(
      { error: `Source segment cannot exceed ${MAX_SOURCE_DURATION_SEC}s` },
      { status: 400 }
    )
  }
  const speedFactor =
    rawSpeedFactor === undefined ? 1 : (rawSpeedFactor as number)
  if (typeof speedFactor !== 'number' || !isAllowedSpeedFactor(speedFactor)) {
    return NextResponse.json(
      { error: 'speedFactor must be one of: 0.25, 0.333…, 0.5, 1, 2, 3, 4' },
      { status: 400 }
    )
  }

  const projectRoot = process.cwd()
  const ffmpegDir = join(projectRoot, 'pro-videos', 'bin')
  const ffmpegBin = join(ffmpegDir, 'ffmpeg')
  const previewDir = join(projectRoot, 'public', 'clip-previews')
  if (!existsSync(previewDir)) mkdirSync(previewDir, { recursive: true })
  cleanOldPreviews(previewDir)

  const tmpDir = mkdtempSync(join(os.tmpdir(), 'preview-clip-'))
  const tmpFullVideo = join(tmpDir, 'full.mp4')
  const tmpProcessed = join(tmpDir, 'processed.mp4')
  const previewId = randomUUID()
  const previewFilename = `${previewId}.mp4`
  const previewPath = join(previewDir, previewFilename)

  try {
    const sectionArg = `*${startTime}-${endTime}`
    // Quality: sort by resolution first, then prefer H.264/MP4 when available.
    // Matches tag-clip/route.ts so the preview the user sees is exactly the
    // quality they'll save.
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

    if (speedFactor === 1) {
      execFileSync(
        ffmpegBin,
        ['-y', '-i', tmpFullVideo, '-c', 'copy', '-movflags', '+faststart', tmpProcessed],
        { stdio: 'pipe', timeout: 60_000 }
      )
    } else {
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
          tmpProcessed,
        ],
        { stdio: 'pipe', timeout: 120_000 }
      )
    }

    renameSync(tmpProcessed, previewPath)

    return NextResponse.json({
      previewId,
      videoUrl: `/clip-previews/${previewFilename}`,
      durationSec: (endTime - startTime) / speedFactor,
      speedFactor,
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    console.error('[preview-clip] Failed:', message)
    return NextResponse.json({ error: message }, { status: 500 })
  } finally {
    try {
      if (existsSync(tmpDir)) {
        execFileSync('rm', ['-rf', tmpDir], { stdio: 'pipe' })
      }
    } catch {
      // best-effort
    }
  }
}
