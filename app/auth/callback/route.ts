import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

// Completes the magic-link sign-in. Supabase sends a link of the form
//   /auth/callback?code=<pkce-code>&next=<redirect>
// We exchange the code for a session (which writes auth cookies via the
// server client's cookie adapter) and then redirect to `next`.
export async function GET(request: NextRequest) {
  const url = request.nextUrl
  const code = url.searchParams.get('code')
  const next = url.searchParams.get('next') ?? '/baseline'

  // Guard against open-redirect: only allow same-origin paths.
  const safeNext = next.startsWith('/') && !next.startsWith('//') ? next : '/baseline'

  if (!code) {
    return NextResponse.redirect(new URL('/login?error=missing_code', url))
  }

  const supabase = await createClient()
  const { error } = await supabase.auth.exchangeCodeForSession(code)

  if (error) {
    return NextResponse.redirect(
      new URL(`/login?error=${encodeURIComponent(error.message)}`, url)
    )
  }

  return NextResponse.redirect(new URL(safeNext, url))
}
