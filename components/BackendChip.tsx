'use client'

import type { ExtractorBackend } from '@/lib/poseExtraction'

// Diagnostic chip surfacing which backend produced the keypoints.
// When tracing looks bad in the browser the user can immediately see
// whether they're on rtmpose-Railway (the path tested locally), server
// MediaPipe (a config issue: POSE_BACKEND env not set on Railway), or
// browser MediaPipe (Railway failed silently and fell back).
const LABELS: Record<ExtractorBackend, { text: string; classes: string }> = {
  'rtmpose-railway': {
    text: 'RTMPose · Railway',
    classes: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  },
  'mediapipe-railway': {
    text: 'MediaPipe · Railway',
    classes: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  },
  'mediapipe-browser': {
    text: 'MediaPipe · Browser',
    classes: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  },
  'mediapipe-browser-fallback': {
    text: 'MediaPipe · Browser (fallback)',
    classes: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
  },
}

export default function BackendChip({
  backend,
  className = '',
}: {
  backend: ExtractorBackend | null
  className?: string
}) {
  if (!backend) return null
  const { text, classes } = LABELS[backend]
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium border ${classes} ${className}`}
      title="Which extractor produced these keypoints. Surfaced for debugging when tracing looks off."
    >
      {text}
    </span>
  )
}
