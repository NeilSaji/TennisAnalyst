'use client'

import { useMemo, useState } from 'react'
import SegmentCard from '@/components/SegmentCard'
import type { SegmentCardSaveOverride } from '@/components/SegmentCard'
import type { VideoSegment } from '@/lib/supabase'

// Filter chips at the top of the grid. Ordering mirrors the baseline
// allowlist; 'all' is the default. Baseline shot types + two
// classifier-only labels we want users to be able to filter by.
const FILTER_OPTIONS = [
  'all',
  'forehand',
  'backhand',
  'serve',
  'volley',
  'slice',
] as const
type FilterOption = (typeof FILTER_OPTIONS)[number]

export interface SegmentPickerGridProps {
  sessionId: string
  segments: VideoSegment[]
  blobUrl: string
  onSaveAsBaseline: (segmentId: string, override: SegmentCardSaveOverride) => void | Promise<void>
  signedIn?: boolean
  // Per-segment saving/saved/error state is owned by the parent so
  // optimistic UI stays consistent across remounts.
  savingSegmentId?: string | null
  savedSegmentIds?: ReadonlySet<string>
  errorBySegmentId?: Readonly<Record<string, string | null>>
}

// Grid of per-segment cards for multi-shot videos. Filter chips
// narrow the visible set by shot_type; 'all' shows everything. The
// grid does not fetch segments itself -- the parent (analyze page)
// is responsible for that so the fetch can be shared with other
// consumers if needed.
export default function SegmentPickerGrid({
  sessionId: _sessionId,
  segments,
  blobUrl,
  onSaveAsBaseline,
  signedIn = true,
  savingSegmentId = null,
  savedSegmentIds,
  errorBySegmentId,
}: SegmentPickerGridProps) {
  const [filter, setFilter] = useState<FilterOption>('all')

  const filtered = useMemo(() => {
    if (filter === 'all') return segments
    return segments.filter((s) => s.shot_type === filter)
  }, [filter, segments])

  // Count of each shot type so the chips can surface how much is
  // under each filter without making the user guess.
  const counts = useMemo(() => {
    const map: Record<string, number> = { all: segments.length }
    for (const s of segments) {
      map[s.shot_type] = (map[s.shot_type] ?? 0) + 1
    }
    return map
  }, [segments])

  if (segments.length === 0) return null

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-white">
            {segments.length} segments detected
          </h3>
          <p className="text-xs text-white/40">
            Save any of them as a baseline to track drift on that shot.
          </p>
        </div>
      </div>

      <div className="flex gap-2 flex-wrap mb-4" role="group" aria-label="Shot type filter">
        {FILTER_OPTIONS.map((opt) => {
          const count = counts[opt] ?? 0
          const disabled = opt !== 'all' && count === 0
          const active = filter === opt
          return (
            <button
              key={opt}
              type="button"
              onClick={() => setFilter(opt)}
              disabled={disabled}
              className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                active
                  ? 'bg-emerald-500 border-emerald-400 text-white'
                  : disabled
                    ? 'bg-white/0 border-white/5 text-white/20 cursor-not-allowed'
                    : 'bg-white/5 border-white/10 text-white/70 hover:bg-white/10 hover:text-white'
              }`}
              aria-pressed={active}
            >
              {opt}
              {count > 0 && <span className="ml-1 text-white/40">({count})</span>}
            </button>
          )
        })}
      </div>

      {filtered.length === 0 ? (
        <p className="text-xs text-white/40 py-6 text-center">
          No segments match this filter.
        </p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {filtered.map((segment) => (
            <SegmentCard
              key={segment.id}
              segment={segment}
              blobUrl={blobUrl}
              onSave={onSaveAsBaseline}
              saving={savingSegmentId === segment.id}
              saved={savedSegmentIds?.has(segment.id) ?? false}
              errorMessage={errorBySegmentId?.[segment.id] ?? null}
              signedIn={signedIn}
            />
          ))}
        </div>
      )}
    </div>
  )
}
