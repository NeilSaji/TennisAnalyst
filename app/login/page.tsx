'use client'

import { useState, Suspense } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'

function LoginForm() {
  const params = useSearchParams()
  const next = params.get('next') ?? '/baseline'
  const [email, setEmail] = useState('')
  const [status, setStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle')
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!email.trim() || status === 'sending') return
    setStatus('sending')
    setError(null)

    const supabase = createClient()
    const origin = window.location.origin
    const callbackUrl = `${origin}/auth/callback?next=${encodeURIComponent(next)}`

    const { error: otpError } = await supabase.auth.signInWithOtp({
      email: email.trim(),
      options: {
        emailRedirectTo: callbackUrl,
      },
    })

    if (otpError) {
      setStatus('error')
      setError(otpError.message)
      return
    }
    setStatus('sent')
  }

  return (
    <div className="max-w-md mx-auto px-4 py-16">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-black text-white mb-2">Sign in</h1>
        <p className="text-white/50 text-sm">
          We&apos;ll email you a link. One click and you&apos;re in.
        </p>
      </div>

      {status === 'sent' ? (
        <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-6 text-center">
          <div className="text-4xl mb-3">📬</div>
          <p className="text-white font-semibold mb-2">Check your inbox</p>
          <p className="text-white/60 text-sm">
            We sent a sign-in link to <span className="text-emerald-300">{email}</span>. Click it
            to finish signing in.
          </p>
          <button
            onClick={() => {
              setStatus('idle')
              setEmail('')
            }}
            className="mt-4 text-xs text-white/50 hover:text-white underline underline-offset-2"
          >
            Use a different email
          </button>
        </div>
      ) : (
        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="block text-sm text-white/60 mb-2" htmlFor="email">
              Email
            </label>
            <input
              id="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:border-emerald-500/50"
              disabled={status === 'sending'}
            />
          </div>

          {error && (
            <p className="text-red-400 text-sm">{error}</p>
          )}

          <button
            type="submit"
            disabled={status === 'sending' || !email.trim()}
            className="w-full px-4 py-3 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status === 'sending' ? 'Sending link...' : 'Email me a link'}
          </button>

          <p className="text-center text-white/40 text-xs pt-2">
            You can still{' '}
            <Link href="/analyze" className="text-white/60 hover:text-white underline">
              analyze a swing without an account
            </Link>
            . Your data just won&apos;t persist.
          </p>
        </form>
      )}
    </div>
  )
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="max-w-md mx-auto px-4 py-16 text-white/50">Loading...</div>}>
      <LoginForm />
    </Suspense>
  )
}
