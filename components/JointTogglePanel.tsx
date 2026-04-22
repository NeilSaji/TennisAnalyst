'use client'

import { useJointStore } from '@/store'
import { usePoseStore } from '@/store'
import type { JointGroup } from '@/lib/jointAngles'
import { getRecommendedVisibility, getShotTypeConfig } from '@/lib/shotTypeConfig'

// Wrist joint-dots were retired as visual noise — the racket-path trail is
// the signal the user cares about on the wrist side. 'wrists' stays in the
// underlying JointGroup enum so existing types and shot-config references
// still compile.
const JOINT_META: { group: JointGroup; label: string; emoji: string; color: string }[] = [
  { group: 'shoulders', label: 'Shoulders', emoji: '🟡', color: 'bg-amber-500' },
  { group: 'elbows', label: 'Elbows', emoji: '🔴', color: 'bg-red-500' },
  { group: 'hips', label: 'Hips', emoji: '🟣', color: 'bg-violet-500' },
  { group: 'knees', label: 'Knees', emoji: '🔵', color: 'bg-blue-500' },
  { group: 'ankles', label: 'Ankles', emoji: '🩷', color: 'bg-pink-500' },
]

export default function JointTogglePanel() {
  const {
    visible,
    showSkeleton,
    showRacket,
    toggleJoint,
    toggleSkeleton,
    toggleRacket,
    setAllVisible,
    setVisibility,
  } = useJointStore()
  // Drive shot-specific focus recommendations off the user's current clip
  // instead of a pro reference (pros are gone). Falls back to null if the
  // uploaded clip has no classified shot type yet.
  const { shotType: userShotType } = usePoseStore()
  const shotConfig = userShotType ? getShotTypeConfig(userShotType) : null
  const allOn = Object.values(visible).every(Boolean)

  const handleShotFocus = () => {
    if (!userShotType) return
    setVisibility(getRecommendedVisibility(userShotType))
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
          onClick={toggleRacket}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-medium transition-all ${
            showRacket
              ? 'border-white/20 bg-white/10 text-white'
              : 'border-white/5 bg-white/0 text-white/30'
          }`}
        >
          <span className="w-3 h-3 rounded-full bg-amber-400 flex-shrink-0" />
          Racket Path (center)
        </button>
      </div>
    </div>
  )
}
