import { execFileSync } from 'node:child_process'
import { runInThisContext } from 'node:vm'
import { Innertube, UniversalCache } from 'youtubei.js'
import { BG, type BgConfig } from 'bgutils-js'
import { JSDOM } from 'jsdom'

// YouTube's InnerTube now gates anonymous playback behind a Proof-of-Origin
// (PO) token minted by BotGuard. bgutils-js runs the BotGuard challenge
// locally, youtubei.js consumes the token. Result: direct googlevideo.com
// media URLs without cookies or a logged-in browser. Token is visitor-bound,
// not content-bound, so we can reuse it across videos.
const CACHE_TTL_MS = 6 * 60 * 60 * 1000

let _cachedClient: Innertube | null = null
let _cacheExpires = 0
let _mintInFlight: Promise<Innertube> | null = null

function installJsdomGlobals() {
  // BotGuard's descrambled interpreter expects a DOM-ish environment
  // (`window`, `document`). Node lacks both. Install once per process;
  // the VM eval'd below attaches itself to globalThis[globalName] and
  // reads these globals during snapshot generation.
  const g = globalThis as unknown as Record<string, unknown>
  if (!g.window) {
    const dom = new JSDOM()
    g.window = dom.window
    g.document = dom.window.document
  }
}

async function mintInnertubeClient(): Promise<Innertube> {
  if (_cachedClient && Date.now() < _cacheExpires) return _cachedClient
  if (_mintInFlight) return _mintInFlight

  _mintInFlight = (async () => {
    installJsdomGlobals()

    // Probe to get a fresh visitorData string; retrieve_player=false keeps
    // this round trip cheap since we don't need the player yet.
    const probe = await Innertube.create({
      retrieve_player: false,
      cache: new UniversalCache(false),
    })
    const visitorData = probe.session.context.client.visitorData
    if (!visitorData) throw new Error('youtubei.js returned no visitorData')

    const bgConfig: BgConfig = {
      fetch: (url, opts) => fetch(url as RequestInfo, opts),
      globalObj: globalThis as unknown as Record<string, unknown>,
      identifier: visitorData,
      // Constant for the web client request key (from the official LuanRT
      // bgutils-js example). Rotates rarely; update if BG starts 4xx-ing.
      requestKey: 'O43z0dpjhgX20SCx4KAo',
    }

    const challenge = await BG.Challenge.create(bgConfig)
    if (!challenge) throw new Error('BotGuard challenge creation failed')
    const script =
      challenge.interpreterJavascript.privateDoNotAccessOrElseSafeScriptWrappedValue
    if (!script) throw new Error('BotGuard returned no interpreter script')

    // The interpreter MUST run against our globalThis so it can register
    // the VM at globalThis[challenge.globalName]; BG.PoToken.generate()
    // looks it up there. Source is the BotGuard challenge script fetched
    // over TLS from youtube.com; a sandbox would block VM registration.
    runInThisContext(script)

    const poTokenResult = await BG.PoToken.generate({
      program: challenge.program,
      globalName: challenge.globalName,
      bgConfig,
    })

    const client = await Innertube.create({
      po_token: poTokenResult.poToken,
      visitor_data: visitorData,
      cache: new UniversalCache(false),
    })
    _cachedClient = client
    _cacheExpires = Date.now() + CACHE_TTL_MS
    return client
  })()

  try {
    return await _mintInFlight
  } finally {
    _mintInFlight = null
  }
}

export function extractYouTubeVideoId(url: string): string {
  const u = new URL(url)
  if (u.hostname === 'youtu.be') return u.pathname.slice(1).split('/')[0]
  const v = u.searchParams.get('v')
  if (v) return v
  const parts = u.pathname.split('/').filter(Boolean)
  const markerIdx = parts.findIndex(
    (p) => p === 'shorts' || p === 'embed' || p === 'live',
  )
  if (markerIdx !== -1 && parts[markerIdx + 1]) return parts[markerIdx + 1]
  throw new Error(`Could not parse YouTube video id from ${url}`)
}

export interface DownloadSectionOpts {
  youtubeUrl: string
  startSec: number
  endSec: number
  outputPath: string
  ffmpegBin: string
}

/**
 * Writes [startSec, endSec] of a YouTube video to outputPath as mp4, lossless
 * stream-copy from the best available adaptive video + audio streams.
 */
export async function downloadYouTubeSection(
  opts: DownloadSectionOpts,
): Promise<void> {
  const { youtubeUrl, startSec, endSec, outputPath, ffmpegBin } = opts
  const videoId = extractYouTubeVideoId(youtubeUrl)
  const yt = await mintInnertubeClient()
  // The IOS client returns pre-deciphered media URLs; the WEB client
  // returns signature_cipher blobs that require nsig deobfuscation (which
  // YouTube breaks every few weeks). IOS = stable, and supports up to 4K.
  const info = await yt.getInfo(videoId, { client: 'IOS' })

  const videoFmt = info.chooseFormat({ type: 'video', quality: 'best' })
  const audioFmt = info.chooseFormat({ type: 'audio', quality: 'best' })

  const [videoUrl, audioUrl] = await Promise.all([
    videoFmt.decipher(yt.session.player),
    audioFmt.decipher(yt.session.player),
  ])

  const duration = endSec - startSec
  // `-ss` BEFORE each `-i` makes ffmpeg issue HTTP Range requests against
  // the googlevideo CDN, so we only transfer the bytes in our window.
  // `-c copy` keeps the clip lossless.
  execFileSync(
    ffmpegBin,
    [
      '-y',
      '-ss',
      String(startSec),
      '-i',
      videoUrl,
      '-ss',
      String(startSec),
      '-i',
      audioUrl,
      '-t',
      String(duration),
      '-map',
      '0:v:0',
      '-map',
      '1:a:0',
      '-c',
      'copy',
      '-movflags',
      '+faststart',
      outputPath,
    ],
    { stdio: 'pipe', timeout: 180_000 },
  )
}
