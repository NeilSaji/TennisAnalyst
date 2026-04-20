// Standalone sanity check for the YouTube download path. Run with:
//   node scripts/smoke-test-yt.mjs
// Asserts we can pull a 3-second clip from a known public video without
// cookies. Exits 0 on success, 1 on failure.

import { execFileSync } from 'node:child_process'
import { existsSync, mkdtempSync, statSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import os from 'node:os'
import { runInThisContext } from 'node:vm'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)

const { Innertube, UniversalCache } = await import('youtubei.js')
const { BG } = await import('bgutils-js')
const { JSDOM } = await import('jsdom')
const ffmpegInstaller = require('@ffmpeg-installer/ffmpeg')

const TEST_VIDEO_ID = 'dQw4w9WgXcQ'
const START = 10
const END = 13

async function main() {
  console.log('[smoke] installing jsdom globals')
  const dom = new JSDOM()
  globalThis.window = dom.window
  globalThis.document = dom.window.document

  console.log('[smoke] probing innertube for visitorData')
  const probe = await Innertube.create({
    retrieve_player: false,
    cache: new UniversalCache(false),
  })
  const visitorData = probe.session.context.client.visitorData
  if (!visitorData) throw new Error('no visitorData')

  const bgConfig = {
    fetch: (url, opts) => fetch(url, opts),
    globalObj: globalThis,
    identifier: visitorData,
    requestKey: 'O43z0dpjhgX20SCx4KAo',
  }

  console.log('[smoke] creating BotGuard challenge')
  const challenge = await BG.Challenge.create(bgConfig)
  if (!challenge) throw new Error('challenge creation failed')

  const script =
    challenge.interpreterJavascript.privateDoNotAccessOrElseSafeScriptWrappedValue
  if (!script) throw new Error('no interpreter js')
  runInThisContext(script)

  console.log('[smoke] minting PO token')
  const poTokenResult = await BG.PoToken.generate({
    program: challenge.program,
    globalName: challenge.globalName,
    bgConfig,
  })

  console.log('[smoke] poToken minted (len=' + poTokenResult.poToken.length + ')')

  const yt = await Innertube.create({
    po_token: poTokenResult.poToken,
    visitor_data: visitorData,
    cache: new UniversalCache(false),
  })

  console.log('[smoke] fetching video info (client=IOS)')
  const info = await yt.getInfo(TEST_VIDEO_ID, { client: 'IOS' })
  console.log('[smoke] title:', info.basic_info.title)

  const videoFmt = info.chooseFormat({ type: 'video', quality: 'best' })
  const audioFmt = info.chooseFormat({ type: 'audio', quality: 'best' })
  console.log('[smoke] video itag=%s mime=%s', videoFmt.itag, videoFmt.mime_type)
  console.log('[smoke] audio itag=%s mime=%s', audioFmt.itag, audioFmt.mime_type)

  const videoUrl = await videoFmt.decipher(yt.session.player)
  const audioUrl = await audioFmt.decipher(yt.session.player)

  const tmpDir = mkdtempSync(join(os.tmpdir(), 'yt-smoke-'))
  const out = join(tmpDir, 'clip.mp4')

  console.log('[smoke] ffmpeg clipping [%ds, %ds)', START, END)
  execFileSync(
    ffmpegInstaller.path,
    [
      '-y',
      '-ss', String(START),
      '-i', videoUrl,
      '-ss', String(START),
      '-i', audioUrl,
      '-t', String(END - START),
      '-map', '0:v:0',
      '-map', '1:a:0',
      '-c', 'copy',
      '-movflags', '+faststart',
      out,
    ],
    { stdio: 'inherit', timeout: 180_000 },
  )

  if (!existsSync(out)) throw new Error('no output file')
  const sz = statSync(out).size
  console.log('[smoke] wrote', out, '(' + sz + ' bytes)')
  if (sz < 10_000) throw new Error('output suspiciously small')

  rmSync(tmpDir, { recursive: true, force: true })
  console.log('[smoke] PASS')
}

main().catch((err) => {
  console.error('[smoke] FAIL:', err?.message || err)
  if (err?.stack) console.error(err.stack)
  process.exit(1)
})
