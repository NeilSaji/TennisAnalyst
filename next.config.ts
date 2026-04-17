import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // Both packages do runtime dynamic require() to find their platform binary,
  // which Turbopack can't bundle. Mark them as server externals so Next.js
  // leaves the requires alone and loads them from node_modules on Vercel at
  // runtime instead of trying to analyze them at build time.
  serverExternalPackages: ['@ffmpeg-installer/ffmpeg', 'youtube-dl-exec'],

  // Force the platform-specific ffmpeg binary + the yt-dlp binary into the
  // admin function bundle. Next.js's default tracing only follows JS requires
  // and would miss the data files the binaries live in.
  outputFileTracingIncludes: {
    'app/api/admin/tag-clip/route': [
      './node_modules/@ffmpeg-installer/**/*',
      './node_modules/youtube-dl-exec/bin/**/*',
      './node_modules/youtube-dl-exec/python/**/*',
    ],
    'app/api/admin/preview-clip/route': [
      './node_modules/@ffmpeg-installer/**/*',
      './node_modules/youtube-dl-exec/bin/**/*',
      './node_modules/youtube-dl-exec/python/**/*',
    ],
  },
}

export default nextConfig
