import { handleUpload, type HandleUploadBody } from '@vercel/blob/client'
import { NextRequest, NextResponse } from 'next/server'
import { isAdminTokenValid } from '@/lib/adminAuth'

// Token endpoint for the browser-side Clip Studio upload.
//
// Two kinds of requests arrive here:
//  1. `blob.generate-client-token` — the browser asks for a signed Blob token
//     before starting the upload. MUST be admin-authenticated. The admin
//     token rides inside clientPayload (JSON string) so it travels through
//     @vercel/blob's documented plumbing — not via an undocumented `headers`
//     option that a future library version might silently drop.
//  2. `blob.upload-completed` — server-to-server webhook from Vercel Blob
//     after the upload succeeds. Authenticated by Vercel's HMAC signature,
//     verified inside handleUpload(), NOT by our admin token.
//
// Default policy: require admin auth. Only the known completion-webhook
// event type is allowed through without our token, so a future Blob event
// we don't recognize won't silently become an unauthenticated endpoint.
export async function POST(request: NextRequest): Promise<NextResponse> {
  let body: HandleUploadBody
  try {
    body = (await request.json()) as HandleUploadBody
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  if (body.type !== 'blob.upload-completed') {
    let token: string | undefined
    try {
      const raw = (body as { payload?: { clientPayload?: string } }).payload?.clientPayload
      const parsed = raw ? JSON.parse(raw) : {}
      token = typeof parsed?.adminToken === 'string' ? parsed.adminToken : undefined
    } catch {
      // fall through — token remains undefined, auth will fail
    }
    if (!isAdminTokenValid(token)) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }
  }

  try {
    const jsonResponse = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async () => ({
        allowedContentTypes: ['video/mp4'],
        maximumSizeInBytes: 200 * 1024 * 1024,
        addRandomSuffix: false,
      }),
      onUploadCompleted: async () => {
        // No DB write here — the browser calls /api/clips/save after the
        // upload lands, which is where the pro_swings row is inserted.
      },
    })
    return NextResponse.json(jsonResponse)
  } catch (error) {
    console.error('[clips/upload] handleUpload failed:', error)
    return NextResponse.json({ error: 'Upload token failed' }, { status: 400 })
  }
}
