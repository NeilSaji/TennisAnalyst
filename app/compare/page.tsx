'use client'

import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useState, useRef } from 'react'
import JointTogglePanel from '@/components/JointTogglePanel'
import ProSelector from '@/components/ProSelector'
import { usePoseStore, useJointStore, useComparisonStore } from '@/store'
import { usePoseExtractor } from '@/hooks/usePoseExtractor'
import { getProVideoUrl } from '@/lib/proVideoUrl'

const ComparisonLayout = dynamic(() => import('@/components/ComparisonLayout'), { ssr: false })
const MetricsComparison = dynamic(() => import('@/components/MetricsComparison'), { ssr: false })
const LLMCoachingPanel = dynamic(() => import('@/components/LLMCoachingPanel'), { ssr: false })

export default function ComparePage() {
  const { framesData, blobUrl, localVideoUrl } = usePoseStore()
  const { visible, showSkeleton, showTrail } = useJointStore()
  const { activeProSwing, secondaryBlobUrl, secondaryFramesData, setSecondaryBlobUrl, setSecondaryFramesData } =
    useComparisonStore()

  const [compareMode, setCompareMode] = useState<'pro' | 'custom'>('pro')
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { extract, progress, isProcessing: processing } = usePoseExtractor()

  const userVideoSrc = localVideoUrl || blobUrl
  const hasUserVideo = userVideoSrc && framesData.length > 0
  const hasProData = activeProSwing && (activeProSwing.keypoints_json?.frames?.length ?? 0) > 0
  const hasSecondary = secondaryBlobUrl && secondaryFramesData.length > 0

  const canCompare =
    hasUserVideo &&
    ((compareMode === 'pro' && hasProData) || (compareMode === 'custom' && hasSecondary))

  const handleSecondaryFile = (file: File) => {
    if (!file.type.startsWith('video/')) return
    processSecondaryVideo(file)
  }

  const handleSecondaryDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleSecondaryFile(file)
  }

  const processSecondaryVideo = async (file: File) => {
    // Capture the previous URL so we can revoke it after the new one is ready
    const previousBlobUrl = secondaryBlobUrl

    const result = await extract(file)
    if (!result) return // aborted or no pose detected; hook manages state

    // Use the returned objectUrl for playback (blob URL -> secondary video)
    setSecondaryBlobUrl(result.objectUrl)
    setSecondaryFramesData(result.frames)

    // Revoke previous URL now that the new one is set
    if (previousBlobUrl) URL.revokeObjectURL(previousBlobUrl)
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-white mb-2">Compare Swings</h1>
        <p className="text-white/50">
          Overlay or side-by-side compare your swing against a pro or another video.
        </p>
      </div>

      {!hasUserVideo && (
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-12 text-center mb-8">
          <div className="text-4xl mb-3">🎾</div>
          <p className="text-white font-medium mb-2">No video analyzed yet</p>
          <p className="text-white/40 text-sm mb-6">
            Go to the Analyze page first to upload and process your swing.
          </p>
          <Link
            href="/analyze"
            className="inline-block px-6 py-3 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl transition-colors"
          >
            Analyze My Swing
          </Link>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main comparison area */}
        <div className="lg:col-span-2 space-y-6">
          {/* Compare mode toggle */}
          <div className="flex gap-2 p-1 bg-white/5 rounded-xl w-fit">
            <button
              onClick={() => setCompareMode('pro')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                compareMode === 'pro' ? 'bg-white text-black' : 'text-white/50 hover:text-white'
              }`}
            >
              vs Pro Player
            </button>
            <button
              onClick={() => setCompareMode('custom')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                compareMode === 'custom' ? 'bg-white text-black' : 'text-white/50 hover:text-white'
              }`}
            >
              vs Custom Video
            </button>
          </div>

          {/* Custom video upload */}
          {compareMode === 'custom' && !hasSecondary && (
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleSecondaryDrop}
              onClick={() => !processing && fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
                dragging
                  ? 'border-emerald-400 bg-emerald-500/10'
                  : 'border-white/20 hover:border-white/40 hover:bg-white/5'
              } ${processing ? 'cursor-not-allowed opacity-80' : ''}`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) processSecondaryVideo(file)
                }}
              />
              {processing ? (
                <div className="space-y-3">
                  <p className="text-white">Processing second video... {progress}%</p>
                  <div className="w-full bg-white/10 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="h-full bg-emerald-400 rounded-full transition-all"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              ) : (
                <div>
                  <div className="text-4xl mb-3">🎬</div>
                  <p className="text-white font-medium">Upload second video to compare</p>
                  <p className="text-white/40 text-sm mt-1">MP4, MOV, WebM</p>
                </div>
              )}
            </div>
          )}

          {/* Comparison view */}
          {canCompare && hasUserVideo && (
            <div className="rounded-2xl border border-white/10 overflow-hidden bg-black">
              <div className="p-3 border-b border-white/5">
                <p className="text-sm text-white/60">
                  {compareMode === 'pro'
                    ? `Your swing vs ${activeProSwing?.pros?.name ?? 'Pro'} · ${activeProSwing?.shot_type}`
                    : 'Side by Side Comparison'}
                </p>
              </div>
              <div className="p-3">
                <ComparisonLayout
                  userBlobUrl={userVideoSrc!}
                  userFrames={framesData}
                  proFrames={
                    compareMode === 'pro'
                      ? (activeProSwing?.keypoints_json?.frames ?? [])
                      : secondaryFramesData
                  }
                  proVideoUrl={
                    compareMode === 'pro'
                      ? (getProVideoUrl(activeProSwing) ?? '')
                      : (secondaryBlobUrl ?? '')
                  }
                  proName={
                    compareMode === 'pro'
                      ? (activeProSwing?.pros?.name ?? 'Pro')
                      : 'Video 2'
                  }
                />
              </div>
            </div>
          )}

          {/* Key metrics (kinetic chain, mistakes, coaching cues) */}
          {canCompare && hasUserVideo && (
            <MetricsComparison
              userFrames={framesData}
              proFrames={
                compareMode === 'pro'
                  ? (activeProSwing?.keypoints_json?.frames ?? [])
                  : secondaryFramesData
              }
              shotType={
                compareMode === 'pro'
                  ? (activeProSwing?.shot_type ?? null)
                  : null
              }
            />
          )}

          {/* AI Coach chat */}
          {hasUserVideo && (
            <LLMCoachingPanel
              proSwing={compareMode === 'pro' ? activeProSwing : null}
              compareMode={compareMode}
              compareFrames={compareMode === 'custom' ? secondaryFramesData : undefined}
            />
          )}
        </div>

        {/* Right sidebar */}
        <div className="space-y-4">
          <JointTogglePanel />

          {compareMode === 'pro' && (
            <div className="rounded-xl bg-white/5 border border-white/10 p-4">
              <h3 className="text-sm font-semibold text-white mb-3">Select Pro</h3>
              <ProSelector />
            </div>
          )}
        </div>
      </div>

    </div>
  )
}
