import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import SegmentCard from '@/components/SegmentCard'
import type { VideoSegment } from '@/lib/supabase'

function makeSegment(overrides: Partial<VideoSegment> = {}): VideoSegment {
  return {
    id: 'seg-1',
    session_id: 'sess-1',
    segment_index: 0,
    shot_type: 'backhand',
    start_frame: 0,
    end_frame: 30,
    start_ms: 0,
    end_ms: 1000,
    confidence: 0.83,
    label: null,
    keypoints_json: null,
    analysis_result: null,
    created_at: new Date().toISOString(),
    ...overrides,
  }
}

describe('SegmentCard', () => {
  const BLOB = 'https://abc.public.blob.vercel-storage.com/v.mp4'

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the classifier confidence as a rounded percentage', () => {
    const seg = makeSegment({ confidence: 0.67 })
    render(<SegmentCard segment={seg} blobUrl={BLOB} onSave={vi.fn()} />)
    expect(screen.getByText(/67% conf/)).toBeInTheDocument()
  })

  it('defaults the override dropdown to the classifier shot_type', () => {
    const seg = makeSegment({ shot_type: 'serve' })
    render(<SegmentCard segment={seg} blobUrl={BLOB} onSave={vi.fn()} />)
    const select = screen.getByLabelText(/shot type override/i) as HTMLSelectElement
    expect(select.value).toBe('serve')
  })

  it('defaults the override to forehand when classifier label is unknown', () => {
    const seg = makeSegment({ shot_type: 'unknown' })
    render(<SegmentCard segment={seg} blobUrl={BLOB} onSave={vi.fn()} />)
    const select = screen.getByLabelText(/shot type override/i) as HTMLSelectElement
    // unknown is not a valid baseline shot, so the dropdown starts on forehand
    expect(select.value).toBe('forehand')
  })

  it('shows a "classified as X, saving as Y" hint when the user overrides', () => {
    const seg = makeSegment({ shot_type: 'backhand' })
    render(<SegmentCard segment={seg} blobUrl={BLOB} onSave={vi.fn()} />)
    const select = screen.getByLabelText(/shot type override/i) as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'forehand' } })
    expect(screen.getByText(/Classified as backhand, saving as forehand/i)).toBeInTheDocument()
  })

  it('save button calls onSave with the overridden shot type', async () => {
    const onSave = vi.fn()
    const seg = makeSegment({ id: 'seg-xyz', shot_type: 'backhand' })
    render(<SegmentCard segment={seg} blobUrl={BLOB} onSave={onSave} />)

    const select = screen.getByLabelText(/shot type override/i) as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'forehand' } })

    fireEvent.click(screen.getByRole('button', { name: /save as baseline/i }))

    expect(onSave).toHaveBeenCalledTimes(1)
    const [segmentId, override] = onSave.mock.calls[0]
    expect(segmentId).toBe('seg-xyz')
    expect(override.shotType).toBe('forehand')
  })

  it('save button is disabled and shows "Saving..." while saving is true', () => {
    const onSave = vi.fn()
    const seg = makeSegment()
    render(<SegmentCard segment={seg} blobUrl={BLOB} onSave={onSave} saving />)

    const btn = screen.getByRole('button', { name: /saving/i }) as HTMLButtonElement
    expect(btn.disabled).toBe(true)
  })

  it('shows a sign-in prompt instead of the save button when signedIn=false', () => {
    render(
      <SegmentCard
        segment={makeSegment()}
        blobUrl={BLOB}
        onSave={vi.fn()}
        signedIn={false}
      />,
    )
    expect(screen.queryByRole('button', { name: /save as baseline/i })).toBeNull()
    expect(screen.getByText(/sign in to save/i)).toBeInTheDocument()
  })

  it('renders a "Play this segment" button that triggers playback', () => {
    const seg = makeSegment({ start_ms: 400, end_ms: 2000 })
    render(<SegmentCard segment={seg} blobUrl={BLOB} onSave={vi.fn()} />)
    const btn = screen.getByRole('button', { name: /play this segment/i })
    expect(btn).toBeInTheDocument()
    // jsdom's HTMLMediaElement is a stub; we don't assert on playback state,
    // but we do verify the click doesn't throw.
    fireEvent.click(btn)
  })

  it('shows saved message when saved=true', () => {
    render(
      <SegmentCard
        segment={makeSegment()}
        blobUrl={BLOB}
        onSave={vi.fn()}
        saved
      />,
    )
    expect(screen.getByText(/saved as baseline/i)).toBeInTheDocument()
  })

  it('shows error message when errorMessage is present', () => {
    render(
      <SegmentCard
        segment={makeSegment()}
        blobUrl={BLOB}
        onSave={vi.fn()}
        errorMessage="trim failed"
      />,
    )
    expect(screen.getByText(/trim failed/i)).toBeInTheDocument()
  })

  it('propagates a user-edited label through onSave', () => {
    const onSave = vi.fn()
    render(<SegmentCard segment={makeSegment()} blobUrl={BLOB} onSave={onSave} />)
    const input = screen.getByPlaceholderText(/label/i) as HTMLInputElement
    fireEvent.change(input, { target: { value: 'my great swing' } })
    fireEvent.click(screen.getByRole('button', { name: /save as baseline/i }))

    expect(onSave.mock.calls[0][1].label).toBe('my great swing')
  })
})
