import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // @ffmpeg-installer does a runtime dynamic require() to find its platform
  // binary, which Turbopack can't bundle. Mark it as a server external so
  // Next.js leaves the require alone and loads it from node_modules at
  // runtime. jsdom + youtubei.js are marked external too — they need Node
  // builtins and break when Turbopack tries to inline them.
  serverExternalPackages: [
    '@ffmpeg-installer/ffmpeg',
    'jsdom',
    'youtubei.js',
    'bgutils-js',
  ],

  // Force the platform-specific ffmpeg binary into the admin function
  // bundle. Next.js's default tracing only follows JS requires and would
  // miss the data files the binary lives in.
  outputFileTracingIncludes: {
    'app/api/admin/tag-clip/route': [
      './node_modules/@ffmpeg-installer/**/*',
    ],
    'app/api/admin/preview-clip/route': [
      './node_modules/@ffmpeg-installer/**/*',
    ],
  },
}

export default nextConfig
