import { LANDMARK_INDICES } from '@/lib/jointAngles'
import type { PoseFrame } from '@/lib/supabase'

const TRAIL_LENGTH = 40 // number of frames to keep in trail buffer

// Velocity ceiling (px/sec) used to map segment speed -> hue. A 1080p tennis
// swing typically peaks around 1500 px/s at the wrist; tune empirically.
const MAX_TRAIL_VELOCITY = 1500

export type TrailPoint = {
  x: number
  y: number
  timestamp: number
}

export type SwingPathRenderOptions = {
  rightColor?: string
  leftColor?: string
  racketColor?: string
  showRacketTrail?: boolean
  showWristTrails?: boolean
}

// Maintains a rolling buffer of wrist (and racket-head) positions for swing
// path tracing.
export class SwingPathTracer {
  private rightWristTrail: TrailPoint[] = []
  private leftWristTrail: TrailPoint[] = []
  private racketTrail: TrailPoint[] = []

  push(frame: PoseFrame, canvasWidth: number, canvasHeight: number): void {
    const rWrist = frame.landmarks.find(
      (l) => l.id === LANDMARK_INDICES.RIGHT_WRIST && l.visibility > 0.5
    )
    const lWrist = frame.landmarks.find(
      (l) => l.id === LANDMARK_INDICES.LEFT_WRIST && l.visibility > 0.5
    )

    if (rWrist) {
      this.rightWristTrail.push({
        x: rWrist.x * canvasWidth,
        y: rWrist.y * canvasHeight,
        timestamp: frame.timestamp_ms,
      })
      if (this.rightWristTrail.length > TRAIL_LENGTH) {
        this.rightWristTrail.shift()
      }
    }

    if (lWrist) {
      this.leftWristTrail.push({
        x: lWrist.x * canvasWidth,
        y: lWrist.y * canvasHeight,
        timestamp: frame.timestamp_ms,
      })
      if (this.leftWristTrail.length > TRAIL_LENGTH) {
        this.leftWristTrail.shift()
      }
    }

    // Racket-head: optional, only present on schema_version 2 frames. May be
    // explicitly null (no detection) on those, or absent entirely on legacy.
    const racket = frame.racket_head
    if (racket && racket.confidence >= 0.4) {
      this.racketTrail.push({
        x: racket.x * canvasWidth,
        y: racket.y * canvasHeight,
        timestamp: frame.timestamp_ms,
      })
      if (this.racketTrail.length > TRAIL_LENGTH) {
        this.racketTrail.shift()
      }
    }
  }

  render(
    ctx: CanvasRenderingContext2D,
    rightColorOrOptions: string | SwingPathRenderOptions = '#10b981',
    leftColor = '#60a5fa'
  ): void {
    // Back-compat: render(ctx, rightColor, leftColor) still works.
    let opts: Required<SwingPathRenderOptions>
    if (typeof rightColorOrOptions === 'string') {
      opts = {
        rightColor: rightColorOrOptions,
        leftColor,
        racketColor: '#fbbf24',
        showRacketTrail: true,
        showWristTrails: true,
      }
    } else {
      opts = {
        rightColor: rightColorOrOptions.rightColor ?? '#10b981',
        leftColor: rightColorOrOptions.leftColor ?? '#60a5fa',
        racketColor: rightColorOrOptions.racketColor ?? '#fbbf24',
        showRacketTrail: rightColorOrOptions.showRacketTrail ?? true,
        showWristTrails: rightColorOrOptions.showWristTrails ?? true,
      }
    }

    if (opts.showWristTrails) {
      this.drawTrail(ctx, this.rightWristTrail, opts.rightColor, {
        tipRadius: 5,
      })
      this.drawTrail(ctx, this.leftWristTrail, opts.leftColor, {
        tipRadius: 5,
      })
    }
    if (opts.showRacketTrail) {
      this.drawTrail(ctx, this.racketTrail, opts.racketColor, {
        tipRadius: 8,
        baseLineWidth: 3,
      })
    }
  }

  private drawTrail(
    ctx: CanvasRenderingContext2D,
    trail: TrailPoint[],
    color: string,
    {
      tipRadius = 5,
      baseLineWidth = 2,
    }: { tipRadius?: number; baseLineWidth?: number } = {}
  ): void {
    if (trail.length < 2) return

    ctx.save()
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    // Draw segments with tapering alpha/width and velocity-encoded hue.
    for (let i = 1; i < trail.length; i++) {
      const progress = i / trail.length // 0 (oldest) → 1 (newest)
      const alpha = progress * 0.85
      const lineWidth = baseLineWidth + progress * 4

      // px/sec between this and previous point; graceful on zero dt.
      const prevPt = trail[i - 1]
      const currPt = trail[i]
      const dx = currPt.x - prevPt.x
      const dy = currPt.y - prevPt.y
      const pxDist = Math.sqrt(dx * dx + dy * dy)
      const dtSec = (currPt.timestamp - prevPt.timestamp) / 1000
      const velocity = dtSec > 0 ? pxDist / dtSec : 0
      const hue = Math.max(
        0,
        Math.min(240, 240 - (velocity / MAX_TRAIL_VELOCITY) * 240)
      )
      const saturation = 90
      const lightness = 55

      ctx.globalAlpha = alpha
      ctx.strokeStyle = `hsl(${hue.toFixed(1)}, ${saturation}%, ${lightness}%)`
      ctx.lineWidth = lineWidth

      ctx.beginPath()

      if (i === 1) {
        ctx.moveTo(trail[0].x, trail[0].y)
        ctx.lineTo(trail[1].x, trail[1].y)
      } else {
        // Smooth quadratic curve: start at midpoint of previous segment so
        // each segment begins exactly where the last one ended (no overlap)
        const prev = trail[i - 1]
        const curr = trail[i]
        const startMidX = (trail[i - 2].x + prev.x) / 2
        const startMidY = (trail[i - 2].y + prev.y) / 2
        const endMidX = (prev.x + curr.x) / 2
        const endMidY = (prev.y + curr.y) / 2
        ctx.moveTo(startMidX, startMidY)
        ctx.quadraticCurveTo(prev.x, prev.y, endMidX, endMidY)
      }

      ctx.stroke()
    }

    // Draw a bright dot at the tip (current position) in the trail's base color
    // so each trail stays visually distinguishable when velocities are similar.
    const tip = trail[trail.length - 1]
    ctx.globalAlpha = 1
    ctx.beginPath()
    ctx.arc(tip.x, tip.y, tipRadius, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
    ctx.strokeStyle = 'white'
    ctx.lineWidth = 1.5
    ctx.stroke()

    ctx.restore()
  }

  reset(): void {
    this.rightWristTrail = []
    this.leftWristTrail = []
    this.racketTrail = []
  }

  // Build full trail from all frames (for reviewing a complete swing)
  buildFromFrames(
    frames: PoseFrame[],
    canvasWidth: number,
    canvasHeight: number
  ): void {
    this.reset()
    for (const frame of frames) {
      this.push(frame, canvasWidth, canvasHeight)
    }
  }
}
