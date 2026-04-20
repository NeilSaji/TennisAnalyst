'use client'

import { Suspense, useEffect, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import { useProfile } from '@/hooks/useProfile'
import {
  SKILL_TIER_LABELS,
  PRIMARY_GOAL_LABELS,
  type SkillTier,
  type DominantHand,
  type BackhandStyle,
  type PrimaryGoal,
} from '@/lib/profile'

// Guard against open-redirect: same-origin paths only.
function safeNext(raw: string | null, fallback: string): string {
  if (!raw) return fallback
  return raw.startsWith('/') && !raw.startsWith('//') ? raw : fallback
}

const SKILL_DESCRIPTIONS: Record<SkillTier, string> = {
  beginner: 'Just picking up a racket.',
  intermediate: 'Rally consistently, play casually.',
  competitive: 'Matches, tournaments, strong club level.',
  advanced: 'Top club, college, or pro level.',
}

const GOAL_ORDER: PrimaryGoal[] = ['power', 'consistency', 'topspin', 'slice', 'learning']

function OnboardingForm() {
  const router = useRouter()
  const params = useSearchParams()
  const nextUrl = safeNext(params.get('next'), '/analyze')

  const { loading: profileLoading, isOnboarded, skipped, skipOnboarding } = useProfile()

  const [skillTier, setSkillTier] = useState<SkillTier | null>(null)
  const [dominantHand, setDominantHand] = useState<DominantHand | null>(null)
  const [backhandStyle, setBackhandStyle] = useState<BackhandStyle | null>(null)
  const [primaryGoal, setPrimaryGoal] = useState<PrimaryGoal | null>(null)
  const [goalNote, setGoalNote] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [skipping, setSkipping] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [authChecked, setAuthChecked] = useState(false)

  // Single initial-mount gate: decide once whether to redirect the user away
  // (signed out → /login, already-onboarded → next) or show the form. We
  // deliberately do NOT re-run on every `profile` change — a mid-session
  // auth flip shouldn't wipe the user's in-flight form state.
  const initialGateDoneRef = useRef(false)
  useEffect(() => {
    if (initialGateDoneRef.current) return
    if (profileLoading) return
    initialGateDoneRef.current = true
    const supabase = createClient()
    supabase.auth
      .getUser()
      .then(({ data }) => {
        if (!data.user) {
          router.replace(`/login?next=${encodeURIComponent(`/onboarding?next=${encodeURIComponent(nextUrl)}`)}`)
          return
        }
        // Either completing onboarding or explicitly skipping counts as
        // "done with the form" — bounce to the intended destination.
        if (isOnboarded || skipped) {
          router.replace(nextUrl)
          return
        }
        setAuthChecked(true)
      })
      .catch(() => {
        // Fail safe: if we can't confirm auth, bounce to /login rather than
        // leave the user stuck on "Loading...".
        router.replace(`/login?next=${encodeURIComponent(`/onboarding?next=${encodeURIComponent(nextUrl)}`)}`)
      })
  }, [profileLoading, isOnboarded, skipped, nextUrl, router])

  const needsNote = primaryGoal === 'other'
  const noteOk = !needsNote || goalNote.trim().length > 0
  const complete =
    skillTier !== null &&
    dominantHand !== null &&
    backhandStyle !== null &&
    primaryGoal !== null &&
    noteOk

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!complete || submitting) return
    setSubmitting(true)
    setError(null)

    const supabase = createClient()
    const { error: updateErr } = await supabase.auth.updateUser({
      data: {
        skill_tier: skillTier,
        dominant_hand: dominantHand,
        backhand_style: backhandStyle,
        primary_goal: primaryGoal,
        primary_goal_note: needsNote ? goalNote.trim().slice(0, 120) : null,
        onboarded_at: new Date().toISOString(),
      },
    })

    if (updateErr) {
      setError(updateErr.message)
      setSubmitting(false)
      return
    }

    router.refresh()
    router.replace(nextUrl)
  }

  const skip = async () => {
    if (submitting || skipping) return
    setSkipping(true)
    setError(null)
    try {
      await skipOnboarding()
      router.refresh()
      router.replace(nextUrl)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to skip.')
      setSkipping(false)
    }
  }

  if (!authChecked) {
    return <div className="max-w-2xl mx-auto px-4 py-16 text-white/50">Loading...</div>
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-12">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-black text-white mb-2">Let&apos;s tune the coaching to you</h1>
        <p className="text-white/50 text-sm">This takes 30 seconds.</p>
      </div>

      <form onSubmit={submit} className="space-y-8">
        {/* Skill tier */}
        <section>
          <h2 className="text-sm font-semibold text-white/80 mb-3">How would you rate yourself?</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {(Object.keys(SKILL_TIER_LABELS) as SkillTier[]).map((tier) => {
              const active = skillTier === tier
              return (
                <button
                  key={tier}
                  type="button"
                  onClick={() => setSkillTier(tier)}
                  className={`text-left p-4 rounded-xl border transition-colors ${
                    active
                      ? 'border-emerald-500/60 bg-emerald-500/10'
                      : 'border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/5'
                  }`}
                >
                  <p className={`font-semibold ${active ? 'text-emerald-300' : 'text-white'}`}>
                    {SKILL_TIER_LABELS[tier]}
                  </p>
                  <p className="text-xs text-white/50 mt-1">{SKILL_DESCRIPTIONS[tier]}</p>
                </button>
              )
            })}
          </div>
        </section>

        {/* Handedness + backhand */}
        <section>
          <h2 className="text-sm font-semibold text-white/80 mb-3">Your grip</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-white/50 mb-2">Dominant hand</p>
              <div className="flex gap-2 p-1 bg-white/5 rounded-xl">
                {(['right', 'left'] as DominantHand[]).map((h) => {
                  const active = dominantHand === h
                  return (
                    <button
                      key={h}
                      type="button"
                      onClick={() => setDominantHand(h)}
                      className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                        active ? 'bg-white text-black' : 'text-white/60 hover:text-white'
                      }`}
                    >
                      {h === 'right' ? 'Right' : 'Left'}
                    </button>
                  )
                })}
              </div>
            </div>
            <div>
              <p className="text-xs text-white/50 mb-2">Backhand</p>
              <div className="flex gap-2 p-1 bg-white/5 rounded-xl">
                {(['one_handed', 'two_handed'] as BackhandStyle[]).map((b) => {
                  const active = backhandStyle === b
                  return (
                    <button
                      key={b}
                      type="button"
                      onClick={() => setBackhandStyle(b)}
                      className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                        active ? 'bg-white text-black' : 'text-white/60 hover:text-white'
                      }`}
                    >
                      {b === 'one_handed' ? 'One-handed' : 'Two-handed'}
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
        </section>

        {/* Primary goal */}
        <section>
          <h2 className="text-sm font-semibold text-white/80 mb-3">What do you want to work on?</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {GOAL_ORDER.map((g) => {
              const active = primaryGoal === g
              return (
                <button
                  key={g}
                  type="button"
                  onClick={() => setPrimaryGoal(g)}
                  className={`p-3 rounded-xl border text-sm font-medium transition-colors ${
                    active
                      ? 'border-emerald-500/60 bg-emerald-500/10 text-emerald-300'
                      : 'border-white/10 bg-white/[0.02] text-white/70 hover:border-white/20 hover:bg-white/5'
                  }`}
                >
                  {PRIMARY_GOAL_LABELS[g]}
                </button>
              )
            })}
            <button
              type="button"
              onClick={() => setPrimaryGoal('other')}
              className={`p-3 rounded-xl border text-sm font-medium transition-colors ${
                primaryGoal === 'other'
                  ? 'border-emerald-500/60 bg-emerald-500/10 text-emerald-300'
                  : 'border-white/10 bg-white/[0.02] text-white/70 hover:border-white/20 hover:bg-white/5'
              }`}
            >
              {PRIMARY_GOAL_LABELS.other}
            </button>
          </div>
          {needsNote && (
            <div className="mt-3">
              <label className="block text-xs text-white/50 mb-2" htmlFor="goal-note">
                Tell us what you&apos;re working on ({goalNote.length}/120)
              </label>
              <input
                id="goal-note"
                type="text"
                maxLength={120}
                value={goalNote}
                onChange={(e) => setGoalNote(e.target.value)}
                placeholder="e.g. clean up my serve toss"
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:border-emerald-500/50"
                disabled={submitting}
              />
            </div>
          )}
        </section>

        {error && <p className="text-red-400 text-sm">{error}</p>}

        <button
          type="submit"
          disabled={!complete || submitting || skipping}
          className="w-full px-4 py-3 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {submitting ? 'Saving...' : 'Continue'}
        </button>

        {/* Escape hatch — intentionally tertiary styling so it isn't the
            happy path. Users who skip get generic coaching until they
            complete their profile from /profile. */}
        <div className="text-center">
          <button
            type="button"
            onClick={skip}
            disabled={submitting || skipping}
            className="text-xs text-white/40 hover:text-white/70 transition-colors disabled:cursor-not-allowed"
          >
            {skipping ? 'Saving…' : 'Skip for now →'}
          </button>
        </div>
      </form>
    </div>
  )
}

export default function OnboardingPage() {
  return (
    <Suspense
      fallback={<div className="max-w-2xl mx-auto px-4 py-16 text-white/50">Loading...</div>}
    >
      <OnboardingForm />
    </Suspense>
  )
}
