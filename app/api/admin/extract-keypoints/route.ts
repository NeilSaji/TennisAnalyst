import { NextRequest, NextResponse } from 'next/server'
import { execFile } from 'node:child_process'
import { existsSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { supabaseAdmin as supabase } from '@/lib/supabase'
import { requireAdminAuth } from '@/lib/adminAuth'

const EXTRACT_TIMEOUT_MS = 120_000
const MAX_VIDEO_BYTES = 100 * 1024 * 1024

export async function POST(request: NextRequest) {
  const guard = requireAdminAuth(request)
  if (guard) return guard

  let body
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const { swingId } = body
  if (!swingId || typeof swingId !== 'string') {
    return NextResponse.json(
      { error: 'Missing or invalid swingId' },
      { status: 400 }
    )
  }

  // 1. Fetch the swing from Supabase
  const { data: swing, error: fetchError } = await supabase
    .from('pro_swings')
    .select('id, video_url')
    .eq('id', swingId)
    .single()

  if (fetchError || !swing) {
    return NextResponse.json(
      { error: 'Swing not found' },
      { status: 404 }
    )
  }

  const videoUrl: string = swing.video_url
  if (!videoUrl) {
    return NextResponse.json(
      { error: 'Swing has no video_url' },
      { status: 400 }
    )
  }

  const scriptPath = path.join(
    process.cwd(),
    'railway-service',
    'extract_clip_keypoints.py'
  )
  if (!existsSync(scriptPath)) {
    return NextResponse.json(
      { error: 'Extraction script not found at railway-service/extract_clip_keypoints.py' },
      { status: 500 }
    )
  }

  // Resolve video to a local path. Remote blob URLs get downloaded to a tmp
  // file; legacy relative paths (e.g. /pro-videos/foo.mp4) resolve under public/.
  const isRemote = /^https?:\/\//i.test(videoUrl)
  let videoPath: string
  let tmpDir: string | null = null

  if (isRemote) {
    try {
      const res = await fetch(videoUrl)
      if (!res.ok) {
        return NextResponse.json(
          { error: `Failed to fetch video (${res.status})` },
          { status: 502 }
        )
      }
      const contentLength = Number(res.headers.get('content-length') ?? '0')
      if (contentLength && contentLength > MAX_VIDEO_BYTES) {
        return NextResponse.json(
          { error: 'Video exceeds max size' },
          { status: 413 }
        )
      }
      const buf = Buffer.from(await res.arrayBuffer())
      if (buf.length > MAX_VIDEO_BYTES) {
        return NextResponse.json(
          { error: 'Video exceeds max size' },
          { status: 413 }
        )
      }
      tmpDir = mkdtempSync(path.join(tmpdir(), 'extract-kp-'))
      videoPath = path.join(tmpDir, 'video.mp4')
      writeFileSync(videoPath, buf)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown download error'
      console.error('[/api/admin/extract-keypoints] Download error:', message)
      return NextResponse.json({ error: 'Failed to download video' }, { status: 502 })
    }
  } else {
    // Legacy code path: video_url might be a relative path like
    // /pro-videos/foo.mp4 that maps under public/. Resolve it explicitly and
    // verify it stays inside public/ so a row with video_url=../../etc/passwd
    // can't trick us into running the Python script on a system file.
    const publicRoot = path.resolve(process.cwd(), 'public')
    const resolved = path.resolve(publicRoot, videoUrl.replace(/^\/+/, ''))
    if (!resolved.startsWith(publicRoot + path.sep) && resolved !== publicRoot) {
      return NextResponse.json({ error: 'Invalid video path' }, { status: 400 })
    }
    if (!existsSync(resolved)) {
      return NextResponse.json(
        { error: 'Video file not found on disk' },
        { status: 404 }
      )
    }
    videoPath = resolved
  }

  let result: {
    fps_sampled: number
    frame_count: number
    frames: unknown[]
    duration_ms: number
  }

  try {
    const stdout = await new Promise<string>((resolve, reject) => {
      execFile(
        'arch',
        ['-arm64', 'python3', scriptPath, videoPath],
        { timeout: EXTRACT_TIMEOUT_MS, maxBuffer: 50 * 1024 * 1024 },
        (error, stdout, stderr) => {
          if (error) {
            const msg =
              error.code === 'ENOENT'
                ? 'python3 is not installed or not in PATH'
                : error.killed
                  ? `Extraction timed out after ${EXTRACT_TIMEOUT_MS / 1000}s`
                  : `Extraction failed: ${stderr || error.message}`
            reject(new Error(msg))
            return
          }
          resolve(stdout)
        }
      )
    })

    const parsed = JSON.parse(stdout)
    if (parsed.error) {
      return NextResponse.json({ error: parsed.error }, { status: 500 })
    }
    result = parsed
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown extraction error'
    console.error('[/api/admin/extract-keypoints] Extraction error:', message)
    return NextResponse.json({ error: 'Keypoint extraction failed' }, { status: 500 })
  } finally {
    if (tmpDir) {
      try {
        rmSync(tmpDir, { recursive: true, force: true })
      } catch {
        // best-effort cleanup
      }
    }
  }

  // 4. Update the Supabase pro_swings row
  const { error: updateError } = await supabase
    .from('pro_swings')
    .update({
      keypoints_json: result,
      frame_count: result.frame_count,
      duration_ms: result.duration_ms,
      fps: result.fps_sampled,
    })
    .eq('id', swingId)

  if (updateError) {
    console.error('[/api/admin/extract-keypoints] Supabase update error:', updateError.message)
    return NextResponse.json(
      { error: 'Failed to update swing data' },
      { status: 500 }
    )
  }

  // 5. Return success
  return NextResponse.json({
    success: true,
    frameCount: result.frame_count,
    durationMs: result.duration_ms,
  })
}
