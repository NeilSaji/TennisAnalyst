'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useUser } from '@/hooks/useUser'

export default function Nav() {
  const { user, loading } = useUser()

  return (
    <nav className="border-b border-white/5 bg-black/30 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-bold text-white text-lg">
          <span className="text-2xl">🎾</span>
          <span>TennisIQ</span>
        </Link>

        <div className="flex items-center gap-1">
          <NavLink href="/analyze">Analyze</NavLink>
          <NavLink href="/baseline">Baselines</NavLink>
          <NavLink href="/compare">Compare</NavLink>
          <NavLink href="/pros">Pro Library</NavLink>
          <NavLink href="/admin/tag-clips">Clip Studio</NavLink>

          <div className="ml-2 pl-2 border-l border-white/10">
            {loading ? (
              <div className="w-8 h-6" aria-hidden />
            ) : user ? (
              <AccountMenu email={user.email ?? 'Account'} />
            ) : (
              <Link
                href="/login"
                className="px-3 py-1.5 rounded-lg text-sm text-emerald-400 hover:text-emerald-300 hover:bg-emerald-500/10 transition-colors"
              >
                Sign in
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}

function AccountMenu({ email }: { email: string }) {
  // Lightweight dropdown — pure CSS details/summary to avoid click-outside plumbing.
  const initial = email[0]?.toUpperCase() ?? '?'
  return (
    <details className="relative group">
      <summary className="list-none cursor-pointer w-8 h-8 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-300 text-sm font-semibold hover:bg-emerald-500/30">
        {initial}
      </summary>
      <div className="absolute right-0 mt-2 w-56 rounded-xl border border-white/10 bg-[#0a0a0f] shadow-xl overflow-hidden">
        <div className="px-3 py-2 border-b border-white/5">
          <p className="text-white/50 text-xs">Signed in as</p>
          <p className="text-white text-sm truncate">{email}</p>
        </div>
        <form action="/auth/signout" method="post">
          <button
            type="submit"
            className="w-full text-left px-3 py-2 text-sm text-white/70 hover:bg-white/5 hover:text-white transition-colors"
          >
            Sign out
          </button>
        </form>
      </div>
    </details>
  )
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  const pathname = usePathname()
  const isActive = pathname != null && (pathname === href || pathname.startsWith(href + '/'))

  return (
    <Link
      href={href}
      className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
        isActive
          ? 'text-white bg-white/10'
          : 'text-white/60 hover:text-white hover:bg-white/5'
      }`}
    >
      {children}
    </Link>
  )
}
