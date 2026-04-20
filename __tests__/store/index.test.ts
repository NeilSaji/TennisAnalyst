import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  useSyncStore,
  useComparisonStore,
  usePoseStore,
  useVideoStore,
  useJointStore,
  useAnalysisStore,
} from '@/store/index'
import type { PoseFrame, Pro, ProSwing } from '@/lib/supabase'
import { makeFrame, makeStandingPose } from '../helpers'

// ---------------------------------------------------------------------------
// useSyncStore
// ---------------------------------------------------------------------------
describe('useSyncStore', () => {
  beforeEach(() => {
    useSyncStore.setState({ syncedTime: 0, proPlaybackRate: 1, isPlaying: false })
  })

  it('has initial syncedTime of 0', () => {
    expect(useSyncStore.getState().syncedTime).toBe(0)
  })

  it('setSyncedTime updates the value', () => {
    useSyncStore.getState().setSyncedTime(12.5)
    expect(useSyncStore.getState().syncedTime).toBe(12.5)
  })

  it('setSyncedTime can be called multiple times', () => {
    useSyncStore.getState().setSyncedTime(1)
    useSyncStore.getState().setSyncedTime(2)
    useSyncStore.getState().setSyncedTime(3.3)
    expect(useSyncStore.getState().syncedTime).toBe(3.3)
  })

  it('handles zero and negative values', () => {
    useSyncStore.getState().setSyncedTime(-1)
    expect(useSyncStore.getState().syncedTime).toBe(-1)
    useSyncStore.getState().setSyncedTime(0)
    expect(useSyncStore.getState().syncedTime).toBe(0)
  })

  it('has initial proPlaybackRate of 1', () => {
    expect(useSyncStore.getState().proPlaybackRate).toBe(1)
  })

  it('setProPlaybackRate updates the value', () => {
    useSyncStore.getState().setProPlaybackRate(2.5)
    expect(useSyncStore.getState().proPlaybackRate).toBe(2.5)
  })

  it('setProPlaybackRate handles fractional rates', () => {
    useSyncStore.getState().setProPlaybackRate(0.5)
    expect(useSyncStore.getState().proPlaybackRate).toBe(0.5)
  })

  it('setProPlaybackRate can be set to 1 (normal speed)', () => {
    useSyncStore.getState().setProPlaybackRate(8)
    useSyncStore.getState().setProPlaybackRate(1)
    expect(useSyncStore.getState().proPlaybackRate).toBe(1)
  })

  it('setProPlaybackRate accepts high values for fast-forward', () => {
    useSyncStore.getState().setProPlaybackRate(16)
    expect(useSyncStore.getState().proPlaybackRate).toBe(16)
  })

  it('setProPlaybackRate accepts low values for slow motion', () => {
    useSyncStore.getState().setProPlaybackRate(0.25)
    expect(useSyncStore.getState().proPlaybackRate).toBe(0.25)
  })

  it('has initial isPlaying of false', () => {
    expect(useSyncStore.getState().isPlaying).toBe(false)
  })

  it('setIsPlaying(true) updates isPlaying', () => {
    useSyncStore.getState().setIsPlaying(true)
    expect(useSyncStore.getState().isPlaying).toBe(true)
  })

  it('setIsPlaying(false) updates isPlaying back', () => {
    useSyncStore.getState().setIsPlaying(true)
    useSyncStore.getState().setIsPlaying(false)
    expect(useSyncStore.getState().isPlaying).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// useComparisonStore
// ---------------------------------------------------------------------------
describe('useComparisonStore', () => {
  beforeEach(() => {
    useComparisonStore.setState({
      mode: 'side-by-side',
      activePro: null,
      activeProSwing: null,
      secondaryBlobUrl: null,
      secondaryFramesData: [],
    })
  })

  it('has correct initial state', () => {
    const state = useComparisonStore.getState()
    expect(state.mode).toBe('side-by-side')
    expect(state.activePro).toBeNull()
    expect(state.activeProSwing).toBeNull()
    expect(state.secondaryBlobUrl).toBeNull()
    expect(state.secondaryFramesData).toEqual([])
  })

  it('setMode switches to overlay', () => {
    useComparisonStore.getState().setMode('overlay')
    expect(useComparisonStore.getState().mode).toBe('overlay')
  })

  it('setMode switches back to side-by-side', () => {
    useComparisonStore.getState().setMode('overlay')
    useComparisonStore.getState().setMode('side-by-side')
    expect(useComparisonStore.getState().mode).toBe('side-by-side')
  })

  it('setActivePro stores a pro object', () => {
    const pro: Pro = {
      id: 'pro-1',
      name: 'Roger Federer',
      nationality: 'Swiss',
      ranking: 3,
      bio: 'Tennis legend',
      profile_image_url: null,
      created_at: '2024-01-01T00:00:00Z',
    }
    useComparisonStore.getState().setActivePro(pro)
    expect(useComparisonStore.getState().activePro).toEqual(pro)
  })

  it('setActivePro can be set to null', () => {
    const pro: Pro = {
      id: 'pro-1',
      name: 'Federer',
      nationality: null,
      ranking: null,
      bio: null,
      profile_image_url: null,
      created_at: '2024-01-01T00:00:00Z',
    }
    useComparisonStore.getState().setActivePro(pro)
    useComparisonStore.getState().setActivePro(null)
    expect(useComparisonStore.getState().activePro).toBeNull()
  })

  it('setActiveProSwing stores a swing object', () => {
    const swing: ProSwing = {
      id: 'swing-1',
      pro_id: 'pro-1',
      shot_type: 'forehand',
      video_url: 'https://example.com/video.mp4',
      thumbnail_url: null,
      keypoints_json: { fps_sampled: 30, frame_count: 10, frames: [] },
      fps: 30,
      frame_count: 10,
      duration_ms: 333,
      phase_labels: { contact: 5 },
      metadata: {},
      created_at: '2024-01-01T00:00:00Z',
    }
    useComparisonStore.getState().setActiveProSwing(swing)
    expect(useComparisonStore.getState().activeProSwing).toEqual(swing)
  })

  it('setSecondaryBlobUrl updates the URL', () => {
    useComparisonStore.getState().setSecondaryBlobUrl('https://blob.example.com/video2')
    expect(useComparisonStore.getState().secondaryBlobUrl).toBe(
      'https://blob.example.com/video2'
    )
  })

  it('setSecondaryFramesData stores frames', () => {
    const frames: PoseFrame[] = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 90 }),
      makeFrame(1, 33, makeStandingPose(), { right_elbow: 100 }),
    ]
    useComparisonStore.getState().setSecondaryFramesData(frames)
    expect(useComparisonStore.getState().secondaryFramesData).toHaveLength(2)
    expect(useComparisonStore.getState().secondaryFramesData[0].joint_angles.right_elbow).toBe(90)
  })
})

// ---------------------------------------------------------------------------
// usePoseStore
// ---------------------------------------------------------------------------
describe('usePoseStore', () => {
  const mockRevokeObjectURL = vi.fn()

  beforeEach(() => {
    // Reset store
    usePoseStore.setState({
      framesData: [],
      blobUrl: null,
      localVideoUrl: null,
      sessionId: null,
      shotType: null,
      isProcessing: false,
      progress: 0,
    })
    mockRevokeObjectURL.mockClear()
    // Mock URL.revokeObjectURL
    globalThis.URL.revokeObjectURL = mockRevokeObjectURL
  })

  it('has correct initial state', () => {
    const state = usePoseStore.getState()
    expect(state.framesData).toEqual([])
    expect(state.blobUrl).toBeNull()
    expect(state.localVideoUrl).toBeNull()
    expect(state.sessionId).toBeNull()
    expect(state.isProcessing).toBe(false)
    expect(state.progress).toBe(0)
  })

  it('setFramesData stores frames', () => {
    const frames: PoseFrame[] = [
      makeFrame(0, 0, makeStandingPose()),
      makeFrame(1, 33, makeStandingPose()),
      makeFrame(2, 66, makeStandingPose()),
    ]
    usePoseStore.getState().setFramesData(frames)
    expect(usePoseStore.getState().framesData).toHaveLength(3)
    expect(usePoseStore.getState().framesData[2].frame_index).toBe(2)
  })

  it('setLocalVideoUrl revokes the previous URL', () => {
    // Set an initial URL
    usePoseStore.setState({ localVideoUrl: 'blob:old-url' })

    // Set a new URL - should revoke the old one
    usePoseStore.getState().setLocalVideoUrl('blob:new-url')

    expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:old-url')
    expect(usePoseStore.getState().localVideoUrl).toBe('blob:new-url')
  })

  it('setLocalVideoUrl does not revoke if no previous URL', () => {
    usePoseStore.getState().setLocalVideoUrl('blob:first-url')
    expect(mockRevokeObjectURL).not.toHaveBeenCalled()
    expect(usePoseStore.getState().localVideoUrl).toBe('blob:first-url')
  })

  it('setLocalVideoUrl handles null', () => {
    usePoseStore.setState({ localVideoUrl: 'blob:some-url' })
    usePoseStore.getState().setLocalVideoUrl(null)
    expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:some-url')
    expect(usePoseStore.getState().localVideoUrl).toBeNull()
  })

  it('reset() clears everything and revokes localVideoUrl', () => {
    usePoseStore.setState({
      framesData: [makeFrame(0, 0, [])],
      blobUrl: 'https://blob.example.com/vid',
      localVideoUrl: 'blob:to-revoke',
      sessionId: 'session-123',
      isProcessing: true,
      progress: 75,
    })

    usePoseStore.getState().reset()

    expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:to-revoke')
    const state = usePoseStore.getState()
    expect(state.framesData).toEqual([])
    expect(state.blobUrl).toBeNull()
    expect(state.localVideoUrl).toBeNull()
    expect(state.sessionId).toBeNull()
    expect(state.isProcessing).toBe(false)
    expect(state.progress).toBe(0)
  })

  it('reset() does not revoke if localVideoUrl is null', () => {
    usePoseStore.setState({
      framesData: [makeFrame(0, 0, [])],
      blobUrl: 'https://blob.example.com/vid',
      localVideoUrl: null,
      sessionId: 'session-123',
      isProcessing: true,
      progress: 75,
    })

    usePoseStore.getState().reset()
    expect(mockRevokeObjectURL).not.toHaveBeenCalled()
  })

  it('setProcessing and setProgress work', () => {
    usePoseStore.getState().setProcessing(true)
    expect(usePoseStore.getState().isProcessing).toBe(true)

    usePoseStore.getState().setProgress(50)
    expect(usePoseStore.getState().progress).toBe(50)

    usePoseStore.getState().setProgress(100)
    expect(usePoseStore.getState().progress).toBe(100)

    usePoseStore.getState().setProcessing(false)
    expect(usePoseStore.getState().isProcessing).toBe(false)
  })

  it('setBlobUrl updates the blob URL', () => {
    usePoseStore.getState().setBlobUrl('https://blob.example.com/my-video')
    expect(usePoseStore.getState().blobUrl).toBe('https://blob.example.com/my-video')
  })

  it('setSessionId updates the session ID', () => {
    usePoseStore.getState().setSessionId('abc-123')
    expect(usePoseStore.getState().sessionId).toBe('abc-123')
  })

  it('has initial shotType of null', () => {
    expect(usePoseStore.getState().shotType).toBeNull()
  })

  it('setShotType updates the shot type', () => {
    usePoseStore.getState().setShotType('forehand')
    expect(usePoseStore.getState().shotType).toBe('forehand')
  })

  it('setShotType can be set to different values', () => {
    usePoseStore.getState().setShotType('forehand')
    usePoseStore.getState().setShotType('backhand')
    expect(usePoseStore.getState().shotType).toBe('backhand')
  })

  it('setShotType can be set back to null', () => {
    usePoseStore.getState().setShotType('serve')
    usePoseStore.getState().setShotType(null)
    expect(usePoseStore.getState().shotType).toBeNull()
  })

  it('reset() clears shotType back to null', () => {
    usePoseStore.setState({ shotType: 'forehand' })
    usePoseStore.getState().reset()
    expect(usePoseStore.getState().shotType).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// useVideoStore
// ---------------------------------------------------------------------------
describe('useVideoStore', () => {
  beforeEach(() => {
    useVideoStore.setState({ currentTime: 0, playing: false, duration: 0 })
  })

  it('has correct initial state', () => {
    const state = useVideoStore.getState()
    expect(state.currentTime).toBe(0)
    expect(state.playing).toBe(false)
    expect(state.duration).toBe(0)
  })

  it('setCurrentTime updates the time', () => {
    useVideoStore.getState().setCurrentTime(5.5)
    expect(useVideoStore.getState().currentTime).toBe(5.5)
  })

  it('setPlaying toggles playback state', () => {
    useVideoStore.getState().setPlaying(true)
    expect(useVideoStore.getState().playing).toBe(true)

    useVideoStore.getState().setPlaying(false)
    expect(useVideoStore.getState().playing).toBe(false)
  })

  it('setDuration sets the video duration', () => {
    useVideoStore.getState().setDuration(120.5)
    expect(useVideoStore.getState().duration).toBe(120.5)
  })
})

// ---------------------------------------------------------------------------
// useJointStore
// ---------------------------------------------------------------------------
describe('useJointStore', () => {
  beforeEach(() => {
    useJointStore.setState({
      visible: {
        shoulders: true,
        elbows: true,
        wrists: true,
        hips: true,
        knees: true,
        ankles: true,
      },
      showSkeleton: true,
      showTrail: true,
      showRacket: true,
    })
  })

  it('has all joints visible by default', () => {
    const { visible } = useJointStore.getState()
    expect(visible.shoulders).toBe(true)
    expect(visible.elbows).toBe(true)
    expect(visible.wrists).toBe(true)
    expect(visible.hips).toBe(true)
    expect(visible.knees).toBe(true)
    expect(visible.ankles).toBe(true)
  })

  it('showSkeleton and showTrail are true by default', () => {
    expect(useJointStore.getState().showSkeleton).toBe(true)
    expect(useJointStore.getState().showTrail).toBe(true)
  })

  it('toggleJoint flips one group', () => {
    useJointStore.getState().toggleJoint('elbows')
    expect(useJointStore.getState().visible.elbows).toBe(false)
    // Other groups should remain unchanged
    expect(useJointStore.getState().visible.shoulders).toBe(true)
    expect(useJointStore.getState().visible.wrists).toBe(true)
  })

  it('toggleJoint flips back when called twice', () => {
    useJointStore.getState().toggleJoint('knees')
    expect(useJointStore.getState().visible.knees).toBe(false)
    useJointStore.getState().toggleJoint('knees')
    expect(useJointStore.getState().visible.knees).toBe(true)
  })

  it('toggleSkeleton flips the boolean', () => {
    useJointStore.getState().toggleSkeleton()
    expect(useJointStore.getState().showSkeleton).toBe(false)
    useJointStore.getState().toggleSkeleton()
    expect(useJointStore.getState().showSkeleton).toBe(true)
  })

  it('toggleTrail flips the boolean', () => {
    useJointStore.getState().toggleTrail()
    expect(useJointStore.getState().showTrail).toBe(false)
    useJointStore.getState().toggleTrail()
    expect(useJointStore.getState().showTrail).toBe(true)
  })

  it('showRacket is true by default and toggleRacket flips it', () => {
    expect(useJointStore.getState().showRacket).toBe(true)
    useJointStore.getState().toggleRacket()
    expect(useJointStore.getState().showRacket).toBe(false)
    useJointStore.getState().toggleRacket()
    expect(useJointStore.getState().showRacket).toBe(true)
  })

  it('setAllVisible(false) turns off everything', () => {
    useJointStore.getState().setAllVisible(false)
    const { visible } = useJointStore.getState()
    expect(visible.shoulders).toBe(false)
    expect(visible.elbows).toBe(false)
    expect(visible.wrists).toBe(false)
    expect(visible.hips).toBe(false)
    expect(visible.knees).toBe(false)
    expect(visible.ankles).toBe(false)
  })

  it('setAllVisible(true) turns on everything', () => {
    // First turn everything off
    useJointStore.getState().setAllVisible(false)
    // Then turn everything on
    useJointStore.getState().setAllVisible(true)
    const { visible } = useJointStore.getState()
    expect(visible.shoulders).toBe(true)
    expect(visible.elbows).toBe(true)
    expect(visible.wrists).toBe(true)
    expect(visible.hips).toBe(true)
    expect(visible.knees).toBe(true)
    expect(visible.ankles).toBe(true)
  })

  it('setAllVisible(false) then toggleJoint enables just one group', () => {
    useJointStore.getState().setAllVisible(false)
    useJointStore.getState().toggleJoint('wrists')
    const { visible } = useJointStore.getState()
    expect(visible.wrists).toBe(true)
    expect(visible.shoulders).toBe(false)
    expect(visible.elbows).toBe(false)
    expect(visible.hips).toBe(false)
    expect(visible.knees).toBe(false)
    expect(visible.ankles).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// useAnalysisStore
// ---------------------------------------------------------------------------
describe('useAnalysisStore', () => {
  beforeEach(() => {
    useAnalysisStore.setState({ feedback: '', scores: {}, loading: false })
  })

  it('has correct initial state', () => {
    const state = useAnalysisStore.getState()
    expect(state.feedback).toBe('')
    expect(state.scores).toEqual({})
    expect(state.loading).toBe(false)
  })

  it('setFeedback sets the feedback string', () => {
    useAnalysisStore.getState().setFeedback('Good form!')
    expect(useAnalysisStore.getState().feedback).toBe('Good form!')
  })

  it('appendFeedback concatenates to existing feedback', () => {
    useAnalysisStore.getState().setFeedback('Part 1. ')
    useAnalysisStore.getState().appendFeedback('Part 2. ')
    useAnalysisStore.getState().appendFeedback('Part 3.')
    expect(useAnalysisStore.getState().feedback).toBe('Part 1. Part 2. Part 3.')
  })

  it('appendFeedback works from empty state', () => {
    useAnalysisStore.getState().appendFeedback('First chunk')
    expect(useAnalysisStore.getState().feedback).toBe('First chunk')
  })

  it('setScores stores score data', () => {
    const scores = { form: 85, power: 72, timing: 90 }
    useAnalysisStore.getState().setScores(scores)
    expect(useAnalysisStore.getState().scores).toEqual(scores)
  })

  it('setLoading toggles loading state', () => {
    useAnalysisStore.getState().setLoading(true)
    expect(useAnalysisStore.getState().loading).toBe(true)
    useAnalysisStore.getState().setLoading(false)
    expect(useAnalysisStore.getState().loading).toBe(false)
  })

  it('reset clears feedback, scores, and loading', () => {
    useAnalysisStore.setState({
      feedback: 'Some analysis',
      scores: { accuracy: 95 },
      loading: true,
    })
    useAnalysisStore.getState().reset()
    const state = useAnalysisStore.getState()
    expect(state.feedback).toBe('')
    expect(state.scores).toEqual({})
    expect(state.loading).toBe(false)
  })
})
