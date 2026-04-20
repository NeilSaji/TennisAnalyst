import { create } from 'zustand'
import type { PoseFrame, ProSwing, Pro } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'

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
  setFramesData: (frames: PoseFrame[]) => void
  setBlobUrl: (url: string | null) => void
  setLocalVideoUrl: (url: string | null) => void
  setSessionId: (id: string | null) => void
  setShotType: (type: string | null) => void
  setProcessing: (v: boolean) => void
  setProgress: (p: number) => void
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
  toggleJoint: (group: JointGroup) => void
  toggleSkeleton: () => void
  toggleTrail: () => void
  toggleRacket: () => void
  setAllVisible: (v: boolean) => void
  setVisibility: (map: VisibilityMap) => void
}

const defaultVisible: VisibilityMap = {
  shoulders: true,
  elbows: true,
  wrists: true,
  hips: true,
  knees: true,
  ankles: true,
}

export const useJointStore = create<JointStore>((set) => ({
  visible: { ...defaultVisible },
  showSkeleton: true,
  showTrail: true,
  // On by default: racket tracking is the headline user-facing feature. The
  // tracer silently no-ops on legacy clips where `racket_head` is absent.
  showRacket: true,
  toggleJoint: (group) =>
    set((state) => ({
      visible: { ...state.visible, [group]: !state.visible[group] },
    })),
  toggleSkeleton: () =>
    set((state) => ({ showSkeleton: !state.showSkeleton })),
  toggleTrail: () => set((state) => ({ showTrail: !state.showTrail })),
  toggleRacket: () => set((state) => ({ showRacket: !state.showRacket })),
  setAllVisible: (v) =>
    set({
      visible: Object.fromEntries(
        Object.keys(defaultVisible).map((k) => [k, v])
      ) as VisibilityMap,
    }),
  setVisibility: (map) => set({ visible: { ...map } }),
}))

// Comparison mode and pro selection
type ComparisonMode = 'side-by-side' | 'overlay'

interface ComparisonStore {
  mode: ComparisonMode
  activePro: Pro | null
  activeProSwing: ProSwing | null
  secondaryBlobUrl: string | null // for user-vs-user comparison
  secondaryFramesData: PoseFrame[]
  setMode: (mode: ComparisonMode) => void
  setActivePro: (pro: Pro | null) => void
  setActiveProSwing: (swing: ProSwing | null) => void
  setSecondaryBlobUrl: (url: string | null) => void
  setSecondaryFramesData: (frames: PoseFrame[]) => void
}

export const useComparisonStore = create<ComparisonStore>((set, get) => ({
  mode: 'side-by-side',
  activePro: null,
  activeProSwing: null,
  secondaryBlobUrl: null,
  secondaryFramesData: [],
  setMode: (mode) => set({ mode }),
  setActivePro: (activePro) => set({ activePro }),
  setActiveProSwing: (activeProSwing) => set({ activeProSwing }),
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

export { useProLibraryStore, type ChatMessage } from './proLibrary'
