import type { PoseFrame } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'
import {
  JOINT_GROUPS,
  SKELETON_CONNECTIONS,
  LANDMARK_INDICES,
} from '@/lib/jointAngles'

export type RenderOptions = {
  visible: Record<JointGroup, boolean>
  showSkeleton: boolean
  color?: string           // joint dot color
  skeletonColor?: string   // skeleton line color
  scale?: number           // 0–1 alpha multiplier for ghost/overlay mode
}

const JOINT_GROUP_COLORS: Record<JointGroup, string> = {
  shoulders: '#f59e0b',  // amber
  elbows: '#ef4444',     // red
  wrists: '#10b981',     // emerald (racket hand)
  hips: '#8b5cf6',       // violet
  knees: '#3b82f6',      // blue
  ankles: '#ec4899',     // pink
}

export function renderPose(
  ctx: CanvasRenderingContext2D,
  frame: PoseFrame,
  canvasWidth: number,
  canvasHeight: number,
  options: RenderOptions
): void {
  const { visible, showSkeleton, color, skeletonColor, scale = 1 } = options

  // Build a map of landmark id -> pixel coords + visibility (for opacity).
  // Skip entirely below 0.3; above, caller fades by (vis - 0.3) / 0.7.
  const pixelMap = new Map<
    number,
    { x: number; y: number; visibility: number }
  >()
  for (const lm of frame.landmarks) {
    if (lm.visibility < 0.3) continue
    pixelMap.set(lm.id, {
      x: lm.x * canvasWidth,
      y: lm.y * canvasHeight,
      visibility: lm.visibility,
    })
  }

  // Draw skeleton bones with halo (dark outline first, then colored stroke
  // on top). Makes bones readable on both bright and dark backgrounds.
  if (showSkeleton) {
    ctx.save()
    const baseWidth = 2
    const strokeColor = skeletonColor ?? 'rgba(255,255,255,0.7)'

    for (const [fromId, toId] of SKELETON_CONNECTIONS) {
      const from = pixelMap.get(fromId)
      const to = pixelMap.get(toId)
      if (!from || !to) continue

      // Halo pass
      ctx.globalAlpha = 0.6 * scale
      ctx.strokeStyle = 'rgba(0,0,0,0.4)'
      ctx.lineWidth = baseWidth + 2
      ctx.beginPath()
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(to.x, to.y)
      ctx.stroke()

      // Colored pass on top
      ctx.globalAlpha = 0.6 * scale
      ctx.strokeStyle = strokeColor
      ctx.lineWidth = baseWidth
      ctx.beginPath()
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(to.x, to.y)
      ctx.stroke()
    }
    ctx.restore()
  }

  // Draw joint dots per group with continuous opacity by visibility.
  for (const [group, ids] of Object.entries(JOINT_GROUPS) as unknown as [JointGroup, number[]][]) {
    if (!visible[group]) continue

    const dotColor = color ?? JOINT_GROUP_COLORS[group]
    ctx.save()

    for (const id of ids) {
      const pos = pixelMap.get(id)
      if (!pos) continue

      // Fade joints smoothly through 0.3 -> 1.0 visibility rather than popping.
      const vAlpha = Math.max(0, Math.min(1, (pos.visibility - 0.3) / 0.7))
      ctx.globalAlpha = 0.9 * scale * vAlpha

      // Outer ring
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(0,0,0,0.5)'
      ctx.fill()

      // Colored dot
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 6, 0, Math.PI * 2)
      ctx.fillStyle = dotColor
      ctx.fill()

      // White center
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 2, 0, Math.PI * 2)
      ctx.fillStyle = 'white'
      ctx.fill()
    }
    ctx.globalAlpha = 1
    ctx.restore()
  }
}

// Compute the bounding box of visible landmarks in normalized [0,1] space.
// Returns { minX, minY, maxX, maxY } with padding.
export function computeLandmarkBounds(
  frame: PoseFrame,
  padding = 0.05
): { minX: number; minY: number; maxX: number; maxY: number } | null {
  const visible = frame.landmarks.filter((lm) => lm.visibility >= 0.3)
  if (visible.length === 0) return null

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const lm of visible) {
    if (lm.x < minX) minX = lm.x
    if (lm.y < minY) minY = lm.y
    if (lm.x > maxX) maxX = lm.x
    if (lm.y > maxY) maxY = lm.y
  }

  // Add padding
  const w = maxX - minX
  const h = maxY - minY
  minX = Math.max(0, minX - w * padding)
  minY = Math.max(0, minY - h * padding)
  maxX = Math.min(1, maxX + w * padding)
  maxY = Math.min(1, maxY + h * padding)

  return { minX, minY, maxX, maxY }
}

// Render pose with landmark coordinates zoomed/cropped to a bounding box.
// The bounds define a sub-region of the [0,1] coordinate space that should
// fill the entire canvas.
export function renderPoseZoomed(
  ctx: CanvasRenderingContext2D,
  frame: PoseFrame,
  canvasWidth: number,
  canvasHeight: number,
  bounds: { minX: number; minY: number; maxX: number; maxY: number },
  options: RenderOptions
): void {
  const { visible, showSkeleton, color, skeletonColor, scale = 1 } = options
  const bw = bounds.maxX - bounds.minX
  const bh = bounds.maxY - bounds.minY
  if (bw <= 0 || bh <= 0) return

  // Map landmark [0,1] coords into the cropped canvas space
  const toPixel = (x: number, y: number) => ({
    x: ((x - bounds.minX) / bw) * canvasWidth,
    y: ((y - bounds.minY) / bh) * canvasHeight,
  })

  // Build pixel map keyed by id, with visibility preserved for opacity fade.
  const pixelMap = new Map<
    number,
    { x: number; y: number; visibility: number }
  >()
  for (const lm of frame.landmarks) {
    if (lm.visibility < 0.3) continue
    const p = toPixel(lm.x, lm.y)
    pixelMap.set(lm.id, { x: p.x, y: p.y, visibility: lm.visibility })
  }

  // Draw skeleton bones with halo outline.
  if (showSkeleton) {
    ctx.save()
    const baseWidth = 3
    const strokeColor = skeletonColor ?? 'rgba(255,255,255,0.7)'

    for (const [fromId, toId] of SKELETON_CONNECTIONS) {
      const from = pixelMap.get(fromId)
      const to = pixelMap.get(toId)
      if (!from || !to) continue

      ctx.globalAlpha = 0.6 * scale
      ctx.strokeStyle = 'rgba(0,0,0,0.4)'
      ctx.lineWidth = baseWidth + 2
      ctx.beginPath()
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(to.x, to.y)
      ctx.stroke()

      ctx.globalAlpha = 0.6 * scale
      ctx.strokeStyle = strokeColor
      ctx.lineWidth = baseWidth
      ctx.beginPath()
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(to.x, to.y)
      ctx.stroke()
    }
    ctx.restore()
  }

  // Draw joint dots with confidence-modulated opacity.
  for (const [group, ids] of Object.entries(JOINT_GROUPS) as unknown as [JointGroup, number[]][]) {
    if (!visible[group]) continue

    const dotColor = color ?? JOINT_GROUP_COLORS[group]
    ctx.save()

    for (const id of ids) {
      const pos = pixelMap.get(id)
      if (!pos) continue

      const vAlpha = Math.max(0, Math.min(1, (pos.visibility - 0.3) / 0.7))
      ctx.globalAlpha = 0.9 * scale * vAlpha

      // Outer ring
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 10, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(0,0,0,0.5)'
      ctx.fill()

      // Colored dot
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2)
      ctx.fillStyle = dotColor
      ctx.fill()

      // White center
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 3, 0, Math.PI * 2)
      ctx.fillStyle = 'white'
      ctx.fill()
    }
    ctx.globalAlpha = 1
    ctx.restore()
  }
}

// Normalize landmarks to a common scale for overlay comparison
// Anchors at mid-hip, scales by hip width
export function normalizeLandmarks(frame: PoseFrame): PoseFrame {
  const lHip = frame.landmarks.find((l) => l.id === LANDMARK_INDICES.LEFT_HIP)
  const rHip = frame.landmarks.find((l) => l.id === LANDMARK_INDICES.RIGHT_HIP)

  if (!lHip || !rHip) return frame

  const hipMidX = (lHip.x + rHip.x) / 2
  const hipMidY = (lHip.y + rHip.y) / 2
  const hipWidth = Math.sqrt((rHip.x - lHip.x) ** 2 + (rHip.y - lHip.y) ** 2)
  // Guard against degenerate detections (hips nearly co-located) to prevent Infinity scale
  if (hipWidth < 0.01) return frame
  const scale = 1 / hipWidth

  const normalized = frame.landmarks.map((lm) => ({
    ...lm,
    x: (lm.x - hipMidX) * scale + 0.5,
    y: (lm.y - hipMidY) * scale + 0.5,
  }))

  return { ...frame, landmarks: normalized }
}
