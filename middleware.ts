import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'
import { hasCompletedOnboardingFlow } from '@/lib/profile'

// Paths that must never be gated. Hitting the gate itself (or anything the
// user needs to complete the gate) would cause a redirect loop.
const PROFILE_GATE_EXEMPT_PREFIXES = [
  '/onboarding',
  '/login',
  '/auth',
  '/api/auth',
  '/api/profile',
]

function isExemptPath(pathname: string): boolean {
  return PROFILE_GATE_EXEMPT_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
  )
}

// Paths that require an authenticated session. Anonymous visitors to these
// routes get bounced to /login with a `next` param. Everything else
// (landing page, /login itself, static assets) passes through.
const REQUIRES_AUTH_PREFIXES = ['/analyze', '/baseline', '/live', '/profile']

function requiresAuth(pathname: string): boolean {
  return REQUIRES_AUTH_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
  )
}

// Refresh the Supabase session on every request. Without this, expired access
// tokens never refresh in Server Components / RSC paths and users silently
// drop out of their sessions. Per Supabase SSR docs: "Failing to do this will
// cause significant and difficult to debug authentication issues."
export async function middleware(request: NextRequest) {
  let response = NextResponse.next({ request })

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll()
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value)
          )
          response = NextResponse.next({ request })
          cookiesToSet.forEach(({ name, value, options }) =>
            response.cookies.set(name, value, options)
          )
        },
      },
    }
  )

  // This call triggers the cookie refresh if the access token has rotated.
  const { data: { user } } = await supabase.auth.getUser()

  const { pathname, search } = request.nextUrl
  const originalPath = `${pathname}${search}`

  // Auth gate: anonymous users hitting a protected route get pushed to /login
  // with the original destination preserved as ?next=... . Login completes by
  // replacing into `next`, which then re-enters the profile gate below on the
  // same request cycle (after sign-in).
  if (!user && requiresAuth(pathname)) {
    const loginUrl = request.nextUrl.clone()
    loginUrl.pathname = '/login'
    loginUrl.search = `?next=${encodeURIComponent(originalPath)}`
    return NextResponse.redirect(loginUrl)
  }

  // Profile gate: signed-in users without a completed onboarding record go to
  // /onboarding. We carry the `next` param so the onboarding flow can bounce
  // them back to their original destination afterward. Exempt paths skip the
  // gate to avoid a loop (/onboarding itself, /login, /auth callbacks, etc).
  if (user && !isExemptPath(pathname) && !hasCompletedOnboardingFlow(user.user_metadata)) {
    const onboardingUrl = request.nextUrl.clone()
    onboardingUrl.pathname = '/onboarding'
    onboardingUrl.search = `?next=${encodeURIComponent(originalPath)}`
    return NextResponse.redirect(onboardingUrl)
  }

  return response
}

export const config = {
  // Run on every path EXCEPT static assets, next internals, and API routes.
  // API routes that need Supabase auth call createClient() directly and don't
  // rely on middleware refreshing the session; excluding them saves a round
  // trip and keeps /api/upload (which only needs the Blob token, not auth)
  // from touching Supabase at all.
  matcher: [
    '/((?!api/|_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico|mp4|webm|woff2?)$).*)',
  ],
}
