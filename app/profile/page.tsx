'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import { useProfile } from '@/hooks/useProfile'
import {
  SKILL_TIER_LABELS,
  PRIMARY_GOAL_LABELS,
  type SkillTier,
  type DominantHand,
  type BackhandStyle,
  type PrimaryGoal,
  type UserProfile,
} from '@/lib/profile'

const SKILL_DESCRIPTIONS: Record<SkillTier, string> = {
  beginner: 'Just picking up a racket.',
  intermediate: 'Rally consistently, play casually.',
  competitive: 'Matches, tournaments, strong club level.',
  advanced: 'Top club, college, or pro level.',
}

const GOAL_ORDER: PrimaryGoal[] = ['power', 'consistency', 'topspin', 'slice', 'learning']

export default function ProfilePage() {
  const router = useRouter()
  const { profile, skipped, loading: profileLoading, updateProfile } = useProfile()

  const [skillTier, setSkillTier] = useState<SkillTier | null>(null)
  const [dominantHand, setDominantHand] = useState<DominantHand | null>(null)
  const [backhandStyle, setBackhandStyle] = useState<BackhandStyle | null>(null)
  const [primaryGoal, setPrimaryGoal] = useState<PrimaryGoal | null>(null)
  const [goalNote, setGoalNote] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [savedAt, setSavedAt] = useState<number | null>(null)
  const [authChecked, setAuthChecked] = useState(false)

  // Prime-once guard: a token refresh or auth state change swaps the
  // `profile` reference, and without this the prefill effect would stomp
  // the user's in-progress edits back to the persisted values.
  const primedRef = useRef(false)

  // Auth gate — unauthed users go to /login with a return path.
  useEffect(() => {
    const supabase = createClient()
    supabase.auth.getUser().then(({ data }) => {
      if (!data.user) {
        router.replace('/login?next=/profile')
        return
      }
      setAuthChecked(true)
    })
  }, [router])

  // Pre-fill form only on the first non-null profile load. Subsequent
  // profile reference changes (token refresh, onAuthStateChange) must
  // not re-prime, or they'd stomp in-progress edits.
  useEffect(() => {
    if (!profile || primedRef.current) return
    setSkillTier(profile.skill_tier)
    setDominantHand(profile.dominant_hand)
    setBackhandStyle(profile.backhand_style)
    setPrimaryGoal(profile.primary_goal)
    setGoalNote(profile.primary_goal_note ?? '')
    primedRef.current = true
  }, [profile])

  // Auto-dismiss the "Saved" toast after a few seconds.
  useEffect(() => {
    if (savedAt === null) return
    const t = setTimeout(() => setSavedAt(null), 2500)
    return () => clearTimeout(t)
  }, [savedAt])

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

    try {
      // First-time save after skipping needs `onboarded_at` to flip the user
      // from skipped → onboarded; parseProfile requires it. Editing an
      // existing profile leaves the original timestamp untouched.
      const payload: Partial<UserProfile> = {
        skill_tier: skillTier!,
        dominant_hand: dominantHand!,
        backhand_style: backhandStyle!,
        primary_goal: primaryGoal!,
        primary_goal_note: needsNote ? goalNote.trim().slice(0, 120) : null,
      }
      if (!profile) {
        payload.onboarded_at = new Date().toISOString()
      }
      await updateProfile(payload)
      setSavedAt(Date.now())
      router.refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save.')
    } finally {
      setSubmitting(false)
    }
  }

  if (!authChecked || profileLoading) {
    return <div className="max-w-2xl mx-auto px-4 py-16 text-white/50">Loading...</div>
  }

  // Signed in but metadata missing AND user didn't opt-out — they're
  // mid-flow. Send them back to onboarding. Skipped users fall through
  // and render the empty form below so they can finish setting up.
  if (!profile && !skipped && authChecked) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16">
        <p className="text-white/60 mb-4">You haven&apos;t completed onboarding yet.</p>
        <Link
          href="/onboarding?next=/profile"
          className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl transition-colors"
        >
          Start onboarding
        </Link>
      </div>
    )
  }

  // Skipped-but-not-onboarded users are graduating from skipped → onboarded
  // here, so reframe the copy from "edit" to "finish setting up".
  const finishingAfterSkip = !profile && skipped

  return (
    <div className="max-w-2xl mx-auto px-4 py-12">
      <div className="mb-6 flex items-center justify-between">
        <Link href="/analyze" className="text-sm text-white/50 hover:text-white">
          ← Back to analyze
        </Link>
        {savedAt !== null && (
          <span className="text-emerald-400 text-sm font-medium" role="status">
            Saved
          </span>
        )}
      </div>

      <div className="mb-8">
        <h1 className="text-3xl font-black text-white mb-2">
          {finishingAfterSkip ? 'Finish setting up' : 'Your coaching profile'}
        </h1>
        <p className="text-white/50 text-sm">
          {finishingAfterSkip
            ? 'Tell us a bit about your game for tailored coaching.'
            : 'Edit anytime — changes take effect on your next analysis.'}
        </p>
      </div>

      <form onSubmit={submit} className="space-y-8">
        <section>
          <h2 className="text-sm font-semibold text-white/80 mb-3">Skill level</h2>
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

        <section>
          <h2 className="text-sm font-semibold text-white/80 mb-3">Grip</h2>
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

        <section>
          <h2 className="text-sm font-semibold text-white/80 mb-3">Primary goal</h2>
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
          disabled={!complete || submitting}
          className="w-full px-4 py-3 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {submitting ? 'Saving...' : 'Save changes'}
        </button>
      </form>
    </div>
  )
}
