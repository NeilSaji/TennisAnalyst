import { create } from 'zustand'

export type LiveStatus =
  | 'idle'
  | 'preflight'
  | 'recording'
  | 'uploading'
  | 'complete'
  | 'error'

export type LivePermissionState = 'unknown' | 'granted' | 'denied'

export type LiveTranscriptEntry = {
  id: string
  text: string
  // Time (ms since recording start) when this line was produced by the LLM
  sessionMs: number
  // X-Analysis-Event-Id from /api/live-coach — used at stop time to backfill
  // the session_id / segment_id on the corresponding analysis_events row
  eventId: string | null
  // How many swings this line summarized
  swingCount: number
}

type LiveShotType = 'forehand' | 'backhand' | 'serve' | 'volley'

interface LiveStore {
  status: LiveStatus
  errorMessage: string | null

  shotType: LiveShotType
  cameraPermission: LivePermissionState
  micPermission: LivePermissionState

  swingCount: number
  lastBatchAtMs: number | null
  sessionStartedAtMs: number | null

  transcript: LiveTranscriptEntry[]
  ttsEnabled: boolean
  ttsAvailable: boolean

  setStatus: (status: LiveStatus) => void
  setErrorMessage: (msg: string | null) => void
  setShotType: (t: LiveShotType) => void
  setCameraPermission: (p: LivePermissionState) => void
  setMicPermission: (p: LivePermissionState) => void
  setSwingCount: (n: number) => void
  setLastBatchAtMs: (ms: number | null) => void
  setSessionStartedAtMs: (ms: number | null) => void
  appendTranscriptEntry: (entry: LiveTranscriptEntry) => void
  clearTranscript: () => void
  setTtsEnabled: (v: boolean) => void
  setTtsAvailable: (v: boolean) => void
  resetSession: () => void
}

const INITIAL: Pick<
  LiveStore,
  | 'status'
  | 'errorMessage'
  | 'shotType'
  | 'cameraPermission'
  | 'micPermission'
  | 'swingCount'
  | 'lastBatchAtMs'
  | 'sessionStartedAtMs'
  | 'transcript'
  | 'ttsEnabled'
  | 'ttsAvailable'
> = {
  status: 'idle',
  errorMessage: null,
  shotType: 'forehand',
  cameraPermission: 'unknown',
  micPermission: 'unknown',
  swingCount: 0,
  lastBatchAtMs: null,
  sessionStartedAtMs: null,
  transcript: [],
  ttsEnabled: true,
  ttsAvailable: false,
}

export const useLiveStore = create<LiveStore>((set) => ({
  ...INITIAL,
  setStatus: (status) => set({ status }),
  setErrorMessage: (errorMessage) => set({ errorMessage }),
  setShotType: (shotType) => set({ shotType }),
  setCameraPermission: (cameraPermission) => set({ cameraPermission }),
  setMicPermission: (micPermission) => set({ micPermission }),
  setSwingCount: (swingCount) => set({ swingCount }),
  setLastBatchAtMs: (lastBatchAtMs) => set({ lastBatchAtMs }),
  setSessionStartedAtMs: (sessionStartedAtMs) => set({ sessionStartedAtMs }),
  appendTranscriptEntry: (entry) =>
    set((state) => ({ transcript: [...state.transcript, entry] })),
  clearTranscript: () => set({ transcript: [] }),
  setTtsEnabled: (ttsEnabled) => set({ ttsEnabled }),
  setTtsAvailable: (ttsAvailable) => set({ ttsAvailable }),
  resetSession: () =>
    set({
      status: 'idle',
      errorMessage: null,
      swingCount: 0,
      lastBatchAtMs: null,
      sessionStartedAtMs: null,
      transcript: [],
    }),
}))
