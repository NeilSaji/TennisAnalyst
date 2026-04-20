'use client'

import { useMemo } from 'react'

export type ChainTiming = {
  joint: string
  peakMs: number
}

interface KineticChainTimingBarProps {
  userTimings: ChainTiming[]
  proTimings: ChainTiming[]
  /**
   * Total duration of the user swing in milliseconds. Used as the bar's
   * right edge so the user ticks are normalized to [0, swingDurationMs].
   */
  swingDurationMs: number
  /**
   * Total duration of the pro swing in milliseconds. Defaults to
   * swingDurationMs when not provided (assumes both bars share a duration).
   */
  proSwingDurationMs?: number
}

const GREEN_WINDOW_MS = 20
const AMBER_WINDOW_MS = 50

const PRETTY_JOINT_LABELS: Record<string, string> = {
  hip_rotation: 'Hips',
  trunk_rotation: 'Trunk',
  right_shoulder: 'R Shoulder',
  left_shoulder: 'L Shoulder',
  right_elbow: 'R Elbow',
  left_elbow: 'L Elbow',
  right_wrist: 'R Wrist',
  left_wrist: 'L Wrist',
  right_knee: 'R Knee',
  left_knee: 'L Knee',
}

function prettyLabel(joint: string): string {
  return PRETTY_JOINT_LABELS[joint] ?? joint
}

/**
 * Map each user tick to a color based on how well its inter-link delay
 * matches the pro's same-joint inter-link delay.
 *
 * For each consecutive pair (prev -> curr) of user ticks we compute
 *   userDelay = curr.peakMs - prev.peakMs
 *   proDelay  = proCurr.peakMs - proPrev.peakMs  (matched by joint name)
 * and classify |userDelay - proDelay|.
 *
 * The first tick has no "previous" so it always displays neutral (white).
 */
function classifyUserTicks(
  userTimings: ChainTiming[],
  proTimings: ChainTiming[]
): ('neutral' | 'green' | 'amber' | 'red')[] {
  const proByJoint = new Map(proTimings.map((t) => [t.joint, t.peakMs]))
  const out: ('neutral' | 'green' | 'amber' | 'red')[] = []

  for (let i = 0; i < userTimings.length; i++) {
    if (i === 0) {
      out.push('neutral')
      continue
    }
    const prev = userTimings[i - 1]
    const curr = userTimings[i]
    const userDelay = curr.peakMs - prev.peakMs

    const proPrev = proByJoint.get(prev.joint)
    const proCurr = proByJoint.get(curr.joint)
    if (proPrev == null || proCurr == null) {
      out.push('neutral')
      continue
    }
    const proDelay = proCurr - proPrev
    const err = Math.abs(userDelay - proDelay)
    if (err <= GREEN_WINDOW_MS) out.push('green')
    else if (err <= AMBER_WINDOW_MS) out.push('amber')
    else out.push('red')
  }
  return out
}

const COLOR_STYLES: Record<
  'neutral' | 'green' | 'amber' | 'red',
  { dot: string; label: string }
> = {
  neutral: { dot: 'bg-white/70', label: 'text-white/60' },
  green: { dot: 'bg-emerald-400', label: 'text-emerald-200' },
  amber: { dot: 'bg-amber-400', label: 'text-amber-200' },
  red: { dot: 'bg-red-500', label: 'text-red-200' },
}

export default function KineticChainTimingBar({
  userTimings,
  proTimings,
  swingDurationMs,
  proSwingDurationMs,
}: KineticChainTimingBarProps) {
  const userDur = swingDurationMs > 0 ? swingDurationMs : 1
  const proDur =
    proSwingDurationMs && proSwingDurationMs > 0 ? proSwingDurationMs : userDur

  const userClasses = useMemo(
    () => classifyUserTicks(userTimings, proTimings),
    [userTimings, proTimings]
  )

  if (userTimings.length === 0 && proTimings.length === 0) {
    return (
      <p className="text-xs text-white/40">
        Not enough data to build the chain timeline.
      </p>
    )
  }

  return (
    <div className="space-y-3">
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-semibold text-white/70 uppercase tracking-wider">
            You
          </span>
          <span className="text-[10px] text-white/40 tabular-nums">
            {Math.round(userDur)} ms
          </span>
        </div>
        <BarRow
          ticks={userTimings}
          durationMs={userDur}
          classes={userClasses}
        />
      </div>

      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-semibold text-white/70 uppercase tracking-wider">
            Pro
          </span>
          <span className="text-[10px] text-white/40 tabular-nums">
            {Math.round(proDur)} ms
          </span>
        </div>
        <BarRow
          ticks={proTimings}
          durationMs={proDur}
          classes={proTimings.map(() => 'neutral')}
        />
      </div>

      <div className="flex items-center gap-3 text-[10px] text-white/40">
        <LegendDot color="bg-emerald-400" label="±20 ms" />
        <LegendDot color="bg-amber-400" label="±50 ms" />
        <LegendDot color="bg-red-500" label="> 50 ms" />
      </div>
    </div>
  )
}

interface BarRowProps {
  ticks: ChainTiming[]
  durationMs: number
  classes: ('neutral' | 'green' | 'amber' | 'red')[]
}

function BarRow({ ticks, durationMs, classes }: BarRowProps) {
  return (
    <div className="relative h-10 rounded-md bg-white/5 border border-white/10">
      {ticks.map((t, i) => {
        const pct = Math.max(
          0,
          Math.min(100, (t.peakMs / durationMs) * 100)
        )
        const cls = classes[i] ?? 'neutral'
        const style = COLOR_STYLES[cls]
        return (
          <div
            key={`${t.joint}-${i}`}
            className="absolute top-0 bottom-0 flex flex-col items-center"
            style={{ left: `${pct}%`, transform: 'translateX(-50%)' }}
          >
            <span className={`w-[2px] flex-1 ${style.dot}`} />
            <span
              className={`mt-0.5 text-[10px] whitespace-nowrap ${style.label}`}
            >
              {prettyLabel(t.joint)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className={`inline-block w-2 h-2 rounded-full ${color}`} />
      <span>{label}</span>
    </span>
  )
}
