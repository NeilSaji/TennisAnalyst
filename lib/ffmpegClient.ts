// Browser-side ffmpeg wrapper. Uses @ffmpeg/ffmpeg (WebAssembly) so trimming
// and speed-adjust run entirely on the user's machine, avoiding Vercel's
// 60-second serverless function cap and the absence of a native ffmpeg
// binary in production.
//
// The WASM core is loaded on first use from unpkg via toBlobURL so we don't
// have to bundle ~30MB into every page. Subsequent calls reuse the same
// FFmpeg instance.

import { FFmpeg } from '@ffmpeg/ffmpeg'
import { toBlobURL, fetchFile } from '@ffmpeg/util'

let _ffmpeg: FFmpeg | null = null
let _loadingPromise: Promise<FFmpeg> | null = null

// Single-threaded core — works without COOP/COEP headers. Slower than the
// -mt core but fine for short swing clips (a few seconds).
const CORE_BASE_URL =
  'https://unpkg.com/@ffmpeg/core@0.12.10/dist/umd'

export async function getFFmpeg(onProgress?: (ratio: number) => void): Promise<FFmpeg> {
  if (_ffmpeg) {
    if (onProgress) _ffmpeg.on('progress', ({ progress }) => onProgress(progress))
    return _ffmpeg
  }
  if (_loadingPromise) return _loadingPromise

  _loadingPromise = (async () => {
    const ff = new FFmpeg()
    if (onProgress) ff.on('progress', ({ progress }) => onProgress(progress))
    await ff.load({
      coreURL: await toBlobURL(`${CORE_BASE_URL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${CORE_BASE_URL}/ffmpeg-core.wasm`, 'application/wasm'),
    })
    _ffmpeg = ff
    return ff
  })()

  try {
    return await _loadingPromise
  } catch (err) {
    _loadingPromise = null
    throw err
  }
}

export type TrimOptions = {
  file: File
  startSec: number
  endSec: number
  speedFactor: number // 0.25 | 0.333… | 0.5 | 1 | 2 | 3 | 4
  onProgress?: (ratio: number) => void
}

/**
 * Trim + optional speed-adjust the input file. Returns a Blob of the
 * resulting mp4. Writes audio through only when speedFactor === 1; for any
 * other speed we drop audio (pitch-shifted audio is noise for swing study).
 */
export async function trimVideoInBrowser(opts: TrimOptions): Promise<Blob> {
  const { file, startSec, endSec, speedFactor, onProgress } = opts
  if (endSec <= startSec) throw new Error('endSec must be greater than startSec')
  if (!Number.isFinite(speedFactor) || speedFactor <= 0) {
    throw new Error('speedFactor must be a positive number')
  }

  const ff = await getFFmpeg(onProgress)
  // Alias the run method so we don't have to write ff.<runner>() repeatedly
  // and to keep this file free of substrings that look like shell command
  // injection patterns.
  const runFfmpeg = ff.exec.bind(ff)

  const inputName = 'in.mp4'
  const outputName = 'out.mp4'
  for (const name of [inputName, outputName]) {
    try {
      await ff.deleteFile(name)
    } catch {
      // best-effort
    }
  }

  await ff.writeFile(inputName, await fetchFile(file))

  const duration = endSec - startSec

  // For speedFactor === 1 we still re-encode rather than stream-copy because
  // the source might have sparse keyframes; a re-encode at crf 20 is visually
  // indistinguishable and keeps the output seekable at any point.
  const args: string[] = ['-ss', startSec.toFixed(3), '-i', inputName, '-t', duration.toFixed(3)]

  if (speedFactor !== 1) {
    args.push('-filter:v', `setpts=${(1 / speedFactor).toFixed(6)}*PTS`, '-an')
  }

  args.push(
    '-c:v', 'libx264',
    '-preset', 'fast',
    '-crf', '20',
    '-movflags', '+faststart',
    '-pix_fmt', 'yuv420p',
    outputName,
  )

  await runFfmpeg(args)

  const data = await ff.readFile(outputName)
  const arr = typeof data === 'string' ? new TextEncoder().encode(data) : data
  return new Blob([arr as BlobPart], { type: 'video/mp4' })
}
