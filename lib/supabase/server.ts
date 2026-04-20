import { createServerClient } from '@supabase/ssr'
import { cookies } from 'next/headers'

// Create a fresh Supabase server client for each request. Do not cache —
// the cookie jar is request-scoped.
//
// Reads the session from cookies. Writes refreshed cookies back when called
// from a route handler or server action; silently ignores cookie writes when
// called from a server component (middleware handles refresh in that case).
export async function createClient() {
  const cookieStore = await cookies()

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options)
            )
          } catch {
            // Called from a Server Component — cookies are read-only here.
            // Safe to ignore; middleware keeps the session fresh.
          }
        },
      },
    }
  )
}
