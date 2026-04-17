import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // @ffmpeg-installer/ffmpeg and youtube-dl-exec ship native binaries as
  // data files next to their JS. Next.js's default output tracing follows
  // JS requires but may miss those binaries, so we force-include them in
  // the admin function bundles that actually shell out. Globs are relative
  // to the project root.
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
