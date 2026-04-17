import { handleUpload, type HandleUploadBody } from '@vercel/blob/client'
import { NextRequest, NextResponse } from 'next/server'
import { requireAdminAuth } from '@/lib/adminAuth'

// Token endpoint for the browser-side Clip Studio upload.
//
// The client (after trimming the video in-browser with ffmpeg.wasm) calls
// @vercel/blob/client.upload() with handleUploadUrl pointing here. That
// triggers two server round-trips:
//  1. onBeforeGenerateToken — where we authenticate the request and return
//     a short-lived blob token to the browser.
//  2. onUploadCompleted — a server-to-server webhook from Blob after the
//     browser finishes uploading. We don't insert the DB row here because
//     the client still needs to POST /api/clips/save afterwards with the
//     pro name / shot type / camera angle metadata.
//
// The first call (token generation) runs with the normal request, so we can
// inspect headers. The second call (completion webhook) is signed by Blob
// and arrives without our admin header — handleUpload verifies that
// signature internally, so it's still authenticated; we just can't re-run
// requireAdminAuth on it.
export async function POST(request: NextRequest): Promise<NextResponse> {
  let body: HandleUploadBody
  try {
    body = (await request.json()) as HandleUploadBody
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  // Blob's completion webhook is POSTed with a `type: 'blob.upload-completed'`
  // body. We only check the admin token on the TOKEN-GENERATION call so we
  // don't accidentally block the webhook.
  if (body.type === 'blob.generate-client-token') {
    const guard = requireAdminAuth(request)
    if (guard) return guard
  }

  try {
    const jsonResponse = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async () => ({
        allowedContentTypes: ['video/mp4'],
        maximumSizeInBytes: 200 * 1024 * 1024,
        addRandomSuffix: false,
        // pathname comes in via handleUpload; we don't override it here.
      }),
      onUploadCompleted: async () => {
        // Intentional no-op. The browser calls /api/clips/save with the
        // blob URL + metadata after uploading, which is where the DB row
        // gets inserted.
      },
    })
    return NextResponse.json(jsonResponse)
  } catch (error) {
    const msg = error instanceof Error ? error.message : 'Upload token failed'
    return NextResponse.json({ error: msg }, { status: 400 })
  }
}
