import { NextRequest, NextResponse } from 'next/server'
import { requireAdminAuth } from '@/lib/adminAuth'
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
import { join, dirname } from 'path'
import os from 'os'
import { randomUUID } from 'crypto'
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg'
import youtubeDl from 'youtube-dl-exec'

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
  const guard = requireAdminAuth(request)
  if (guard) return guard

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

  // Use binaries from npm packages so this works on Vercel (which doesn't
  // have yt-dlp or ffmpeg pre-installed). Falls back to nothing if the
  // package didn't ship the current platform; we check that below.
  const ffmpegBin = ffmpegInstaller.path
  if (!ffmpegBin || !existsSync(ffmpegBin)) {
    return NextResponse.json(
      { error: 'ffmpeg binary unavailable on this platform' },
      { status: 500 },
    )
  }
  const ffmpegDir = dirname(ffmpegBin)
  // Previews are written to a NON-public directory and streamed through an
  // auth-gated GET handler. Previously we dropped them under public/ where
  // anyone who knew the UUID could fetch them; moving them out of public/
  // means only admin-authenticated requests can read back the files.
  const previewDir = join(os.tmpdir(), 'tennisiq-clip-previews')
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
    // Use youtube-dl-exec's bundled yt-dlp binary so we don't depend on
    // a system-installed yt-dlp (absent on Vercel). Flag names map from
    // kebab-case CLI to camelCase options.
    await youtubeDl(youtubeUrl, {
      format: 'bestvideo+bestaudio/best',
      // Colon-delimited sort keys (ext:mp4:m4a, codec:avc) are valid yt-dlp
      // syntax but the Node wrapper's types only whitelist bare keys.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      formatSort: ['res', 'ext:mp4:m4a', 'codec:avc', 'fps'] as any,
      mergeOutputFormat: 'mp4',
      noPlaylist: true,
      downloadSections: sectionArg,
      forceKeyframesAtCuts: true,
      ffmpegLocation: ffmpegDir,
      output: tmpFullVideo,
    })

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
      // Client fetches this with the admin header, converts the response to
      // a blob URL, and feeds that into <video src>. We no longer expose a
      // directly-browsable path under /public/.
      videoUrl: `/api/admin/preview-clip/${previewId}`,
      durationSec: (endTime - startTime) / speedFactor,
      speedFactor,
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    console.error('[preview-clip] Failed:', message)
    return NextResponse.json({ error: 'Preview generation failed' }, { status: 500 })
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
