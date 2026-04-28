import { create } from 'zustand'
import type { PoseFrame } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'
import type { ExtractorBackend } from '@/lib/poseExtraction'

// Video playback state
interface VideoStore {
  currentTime: number
  playing: boolean
  duration: number
  setCurrentTime: (t: number) => void
  setPlaying: (p: boolean) => void
  setDuration: (d: number) => void
}

export const useVideoStore = create<VideoStore>((set) => ({
  currentTime: 0,
  playing: false,
  duration: 0,
  setCurrentTime: (currentTime) => set({ currentTime }),
  setPlaying: (playing) => set({ playing }),
  setDuration: (duration) => set({ duration }),
}))

// Extracted pose data for the user's uploaded video
interface PoseStore {
  framesData: PoseFrame[]
  blobUrl: string | null
  localVideoUrl: string | null // object URL for browser playback
  sessionId: string | null
  shotType: string | null
  isProcessing: boolean
  progress: number // 0–100
  // Which extractor produced framesData. Surfaced in the UI as a
  // diagnostic chip so we can tell at a glance which path the user is
  // on when tracing looks wrong. Null until the first extraction lands.
  extractorBackend: ExtractorBackend | null
  setFramesData: (frames: PoseFrame[]) => void
  setBlobUrl: (url: string | null) => void
  setLocalVideoUrl: (url: string | null) => void
  setSessionId: (id: string | null) => void
  setShotType: (type: string | null) => void
  setProcessing: (v: boolean) => void
  setProgress: (p: number) => void
  setExtractorBackend: (backend: ExtractorBackend | null) => void
  reset: () => void
}

export const usePoseStore = create<PoseStore>((set, get) => ({
  framesData: [],
  blobUrl: null,
  localVideoUrl: null,
  sessionId: null,
  shotType: null,
  isProcessing: false,
  progress: 0,
  extractorBackend: null,
  setFramesData: (framesData) => set({ framesData }),
  setBlobUrl: (blobUrl) => set({ blobUrl }),
  setLocalVideoUrl: (url) => {
    const prev = get().localVideoUrl
    if (prev) URL.revokeObjectURL(prev)
    set({ localVideoUrl: url })
  },
  setSessionId: (sessionId) => set({ sessionId }),
  setShotType: (shotType) => set({ shotType }),
  setProcessing: (isProcessing) => set({ isProcessing }),
  setProgress: (progress) => set({ progress }),
  setExtractorBackend: (extractorBackend) => set({ extractorBackend }),
  reset: () => {
    const prev = get().localVideoUrl
    if (prev) URL.revokeObjectURL(prev)
    set({
      framesData: [],
      blobUrl: null,
      localVideoUrl: null,
      sessionId: null,
      shotType: null,
      isProcessing: false,
      progress: 0,
      extractorBackend: null,
    })
  },
}))

// Joint visibility toggles
type VisibilityMap = Record<JointGroup, boolean>

interface JointStore {
  visible: VisibilityMap
  showSkeleton: boolean
  showTrail: boolean
  // Racket-head trail (schema_version 2 clips only). Kept separate from the
  // JointGroup visibility map so existing `visible: Record<JointGroup, boolean>`
  // consumers stay untouched.
  showRacket: boolean
  // On-canvas joint-angle labels (elbow/knee degree badges). On by
  // default — they're the single biggest "looks like a coaching tool,
  // not a hobby project" visual cue, and they surface data we already
  // compute per-frame in jointAngles.ts but weren't drawing.
  showAngles: boolean
  toggleJoint: (group: JointGroup) => void
  toggleSkeleton: () => void
  toggleTrail: () => void
  toggleRacket: () => void
  toggleAngles: () => void
  setAllVisible: (v: boolean) => void
  setVisibility: (map: VisibilityMap) => void
}

const defaultVisible: VisibilityMap = {
  shoulders: true,
  elbows: true,
  // Wrist joint-dots retired as visual clutter; racket-path trail is the
  // signal we show on the wrist side now. Kept on the enum for type
  // compatibility but off by default and not surfaced in the toggle UI.
  wrists: false,
  hips: true,
  knees: true,
  ankles: true,
}

export const useJointStore = create<JointStore>((set) => ({
  visible: { ...defaultVisible },
  showSkeleton: true,
  // Wrist-trail overlay was retired as visual noise once the racket-path
  // overlay became the primary signal. Field + toggle kept on the store
  // so downstream consumers still compile; no UI surfaces it.
  showTrail: false,
  // Default OFF. The racket-path tracer was retired from the demo —
  // without server-side YOLO it was rendering the wrist position (the
  // "forearm fallback") and labelling it "racket trail", which misled
  // viewers and added visual noise on top of the hitting-arm joint
  // dot that already marks the same point. Kept on the store +
  // toggleable so the Railway-backed code path can re-enable it per-
  // clip once real detection is wired end-to-end.
  showRacket: false,
  showAngles: true,
  toggleJoint: (group) =>
    set((state) => ({
      visible: { ...state.visible, [group]: !state.visible[group] },
    })),
  toggleSkeleton: () =>
    set((state) => ({ showSkeleton: !state.showSkeleton })),
  toggleTrail: () => set((state) => ({ showTrail: !state.showTrail })),
  toggleRacket: () => set((state) => ({ showRacket: !state.showRacket })),
  toggleAngles: () => set((state) => ({ showAngles: !state.showAngles })),
  setAllVisible: (v) =>
    set({
      visible: Object.fromEntries(
        Object.keys(defaultVisible).map((k) => [k, v])
      ) as VisibilityMap,
    }),
  setVisibility: (map) => set({ visible: { ...map } }),
}))

// Comparison mode for user-vs-user (baseline) comparison views.
type ComparisonMode = 'side-by-side' | 'overlay'

interface ComparisonStore {
  mode: ComparisonMode
  secondaryBlobUrl: string | null
  secondaryFramesData: PoseFrame[]
  setMode: (mode: ComparisonMode) => void
  setSecondaryBlobUrl: (url: string | null) => void
  setSecondaryFramesData: (frames: PoseFrame[]) => void
}

export const useComparisonStore = create<ComparisonStore>((set, get) => ({
  mode: 'side-by-side',
  secondaryBlobUrl: null,
  secondaryFramesData: [],
  setMode: (mode) => set({ mode }),
  setSecondaryBlobUrl: (url) => {
    const prev = get().secondaryBlobUrl
    if (prev) URL.revokeObjectURL(prev)
    set({ secondaryBlobUrl: url })
  },
  setSecondaryFramesData: (secondaryFramesData) => set({ secondaryFramesData }),
}))

// LLM analysis results
interface AnalysisStore {
  feedback: string
  scores: Record<string, number>
  loading: boolean
  setFeedback: (f: string) => void
  appendFeedback: (chunk: string) => void
  setScores: (s: Record<string, number>) => void
  setLoading: (l: boolean) => void
  reset: () => void
}

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  feedback: '',
  scores: {},
  loading: false,
  setFeedback: (feedback) => set({ feedback }),
  appendFeedback: (chunk) =>
    set((state) => ({ feedback: state.feedback + chunk })),
  setScores: (scores) => set({ scores }),
  setLoading: (loading) => set({ loading }),
  reset: () => set({ feedback: '', scores: {}, loading: false }),
}))

// Synchronized playback time for comparison views
interface SyncStore {
  syncedTime: number
  setSyncedTime: (t: number) => void
  timeMapping: ((userTimeMs: number) => number) | null
  setTimeMapping: (fn: ((userTimeMs: number) => number) | null) => void
  proPlaybackRate: number
  setProPlaybackRate: (rate: number) => void
  isPlaying: boolean
  setIsPlaying: (p: boolean) => void
}

export const useSyncStore = create<SyncStore>((set) => ({
  syncedTime: 0,
  setSyncedTime: (syncedTime) => set({ syncedTime }),
  timeMapping: null,
  setTimeMapping: (timeMapping) => set({ timeMapping }),
  proPlaybackRate: 1,
  setProPlaybackRate: (proPlaybackRate) => set({ proPlaybackRate }),
  isPlaying: false,
  setIsPlaying: (isPlaying) => set({ isPlaying }),
}))

