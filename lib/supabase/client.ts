'use client'

import { createBrowserClient } from '@supabase/ssr'

// Browser client — safe to call at module scope, returns a singleton.
// Uses NEXT_PUBLIC_* env vars so it's available on the client bundle.
export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  )
}
