'use client'

import dynamic from 'next/dynamic'

// MediaPipe + MediaRecorder require a real browser, so these must not SSR.
const LiveCapturePanel = dynamic(() => import('@/components/LiveCapturePanel'), { ssr: false })
const LiveCoachingTranscript = dynamic(() => import('@/components/LiveCoachingTranscript'), { ssr: false })

export default function LivePage() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold text-white">Live Feedback</h1>
        <p className="text-white/60">
          Tap Start, play, and get coached every few swings — no upload, no review. Your session saves afterward like any other clip.
        </p>
      </header>

      <LiveCapturePanel />
      <LiveCoachingTranscript />
    </div>
  )
}
