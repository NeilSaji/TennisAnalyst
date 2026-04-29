'use client'

import { useState } from 'react'
import Link from 'next/link'
import type { Baseline } from '@/lib/supabase'

interface BaselineCardProps {
  baseline: Baseline
  onSetActive?: (id: string) => Promise<void> | void
  onRename?: (id: string, label: string) => Promise<void> | void
  onDelete?: (id: string) => Promise<void> | void
}

function relativeDate(iso: string): string {
  const t = new Date(iso).getTime()
  const days = Math.floor((Date.now() - t) / (1000 * 60 * 60 * 24))
  if (days === 0) return 'today'
  if (days === 1) return 'yesterday'
  if (days < 30) return `${days}d ago`
  if (days < 365) return `${Math.floor(days / 30)}mo ago`
  return `${Math.floor(days / 365)}y ago`
}

export default function BaselineCard({ baseline, onSetActive, onRename, onDelete }: BaselineCardProps) {
  const [editing, setEditing] = useState(false)
  const [labelDraft, setLabelDraft] = useState(baseline.label)
  const [busy, setBusy] = useState(false)

  const submitRename = async () => {
    const next = labelDraft.trim()
    if (!next || next === baseline.label) {
      setEditing(false)
      setLabelDraft(baseline.label)
      return
    }
    setBusy(true)
    try {
      await onRename?.(baseline.id, next)
    } finally {
      setBusy(false)
      setEditing(false)
    }
  }

  return (
    <div
      className={`rounded-xl border p-4 flex gap-4 items-center transition-colors ${
        baseline.is_active
          ? 'border-emerald-500/40 bg-emerald-500/5'
          : 'border-white/10 bg-white/[0.02]'
      }`}
    >
      <Link
        href={`/baseline/${baseline.id}`}
        className="shrink-0 group relative"
        title="Watch with pose overlay"
      >
        <video
          src={baseline.blob_url}
          className="w-24 h-24 rounded-lg object-cover bg-black"
          muted
          playsInline
          preload="metadata"
        />
        <div className="absolute inset-0 rounded-lg bg-black/0 group-hover:bg-black/40 flex items-center justify-center transition-colors">
          <svg viewBox="0 0 24 24" className="w-6 h-6 fill-white opacity-0 group-hover:opacity-100 transition-opacity">
            <polygon points="5,3 19,12 5,21" />
          </svg>
        </div>
      </Link>

      <div className="flex-1 min-w-0">
        {editing ? (
          <input
            autoFocus
            value={labelDraft}
            onChange={(e) => setLabelDraft(e.target.value)}
            onBlur={submitRename}
            onKeyDown={(e) => {
              if (e.key === 'Enter') submitRename()
              if (e.key === 'Escape') {
                setEditing(false)
                setLabelDraft(baseline.label)
              }
            }}
            className="w-full bg-white/10 border border-white/20 rounded px-2 py-1 text-white text-sm"
          />
        ) : (
          <button
            onClick={() => setEditing(true)}
            className="text-left text-white font-semibold hover:text-emerald-300 transition-colors truncate block max-w-full"
            title="Click to rename"
          >
            {baseline.label}
          </button>
        )}

        <div className="flex gap-2 items-center text-xs text-white/50 mt-1">
          <span className="capitalize">{baseline.shot_type}</span>
          <span>·</span>
          <span>{baseline.keypoints_json?.frame_count ?? 0} frames</span>
          <span>·</span>
          <span>{relativeDate(baseline.created_at)}</span>
          {baseline.is_active && (
            <>
              <span>·</span>
              <span className="text-emerald-400 font-medium">Active</span>
            </>
          )}
        </div>
      </div>

      <div className="flex gap-2 shrink-0">
        <Link
          href={`/baseline/${baseline.id}`}
          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 hover:bg-white/10 text-white/70 hover:text-white transition-colors"
        >
          Watch
        </Link>
        {!baseline.is_active && onSetActive && (
          <button
            onClick={async () => {
              setBusy(true)
              try {
                await onSetActive(baseline.id)
              } finally {
                setBusy(false)
              }
            }}
            disabled={busy}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-500 hover:bg-emerald-400 text-white disabled:opacity-50"
          >
            Set active
          </button>
        )}
        {onDelete && (
          <button
            onClick={async () => {
              if (!confirm(`Delete baseline "${baseline.label}"?`)) return
              setBusy(true)
              try {
                await onDelete(baseline.id)
              } finally {
                setBusy(false)
              }
            }}
            disabled={busy}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 hover:bg-red-500/20 text-white/60 hover:text-red-300 disabled:opacity-50"
          >
            Delete
          </button>
        )}
      </div>
    </div>
  )
}
