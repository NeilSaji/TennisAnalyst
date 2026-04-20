'use client'

import { useJointStore, useComparisonStore } from '@/store'
import type { JointGroup } from '@/lib/jointAngles'
import { getRecommendedVisibility, getShotTypeConfig } from '@/lib/shotTypeConfig'

const JOINT_META: { group: JointGroup; label: string; emoji: string; color: string }[] = [
  { group: 'shoulders', label: 'Shoulders', emoji: '🟡', color: 'bg-amber-500' },
  { group: 'elbows', label: 'Elbows', emoji: '🔴', color: 'bg-red-500' },
  { group: 'wrists', label: 'Wrists / Racket', emoji: '🟢', color: 'bg-emerald-500' },
  { group: 'hips', label: 'Hips', emoji: '🟣', color: 'bg-violet-500' },
  { group: 'knees', label: 'Knees', emoji: '🔵', color: 'bg-blue-500' },
  { group: 'ankles', label: 'Ankles', emoji: '🩷', color: 'bg-pink-500' },
]

export default function JointTogglePanel() {
  const {
    visible,
    showSkeleton,
    showTrail,
    showRacket,
    toggleJoint,
    toggleSkeleton,
    toggleTrail,
    toggleRacket,
    setAllVisible,
    setVisibility,
  } = useJointStore()
  const { activeProSwing } = useComparisonStore()

  const allOn = Object.values(visible).every(Boolean)
  const shotType = activeProSwing?.shot_type ?? null
  const shotConfig = shotType ? getShotTypeConfig(shotType) : null

  const handleShotFocus = () => {
    if (!shotType) return
    setVisibility(getRecommendedVisibility(shotType))
  }

  return (
    <div className="rounded-xl bg-white/5 border border-white/10 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">Joint Visibility</h3>
        <div className="flex items-center gap-2">
          {shotConfig && (
            <button
              onClick={handleShotFocus}
              className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
              title={`Show joints most relevant for ${shotConfig.label}`}
            >
              {shotConfig.label} focus
            </button>
          )}
          <button
            onClick={() => setAllVisible(!allOn)}
            className="text-xs text-white/50 hover:text-white transition-colors"
          >
            {allOn ? 'Hide all' : 'Show all'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {JOINT_META.map(({ group, label, color }) => (
          <button
            key={group}
            onClick={() => toggleJoint(group)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-medium transition-all ${
              visible[group]
                ? 'border-white/20 bg-white/10 text-white'
                : 'border-white/5 bg-white/0 text-white/30'
            }`}
          >
            <span
              className={`w-3 h-3 rounded-full flex-shrink-0 ${color} ${
                visible[group] ? 'opacity-100' : 'opacity-20'
              }`}
            />
            {label}
          </button>
        ))}
      </div>

      <div className="border-t border-white/10 pt-3 space-y-2">
        <button
          onClick={toggleSkeleton}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-medium transition-all ${
            showSkeleton
              ? 'border-white/20 bg-white/10 text-white'
              : 'border-white/5 bg-white/0 text-white/30'
          }`}
        >
          <span className="w-3 h-3 rounded-sm bg-white/60 flex-shrink-0" />
          Skeleton Lines
        </button>

        <button
          onClick={toggleTrail}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-medium transition-all ${
            showTrail
              ? 'border-white/20 bg-white/10 text-white'
              : 'border-white/5 bg-white/0 text-white/30'
          }`}
        >
          <span className="w-3 h-3 rounded-full bg-emerald-500 flex-shrink-0" />
          Swing Path Trail
        </button>

        <button
          onClick={toggleRacket}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-medium transition-all ${
            showRacket
              ? 'border-white/20 bg-white/10 text-white'
              : 'border-white/5 bg-white/0 text-white/30'
          }`}
        >
          <span className="w-3 h-3 rounded-full bg-amber-400 flex-shrink-0" />
          Racket
        </button>
      </div>
    </div>
  )
}
