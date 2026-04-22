import { createClient, SupabaseClient } from '@supabase/supabase-js'

// Lazy client — instantiated only at runtime (not during Next.js build)
let _client: SupabaseClient | null = null
let _adminClient: SupabaseClient | null = null

export function getSupabase(): SupabaseClient {
  if (!_client) {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL
    const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
    if (!url || !key) {
      throw new Error(
        'Missing NEXT_PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_ANON_KEY env vars'
      )
    }
    _client = createClient(url, key)
  }
  return _client
}

// Service-role client for server-side admin operations (bypasses RLS).
// Only use in API routes that run exclusively on the server.
//
// Supports both the new sb_secret_... key format and the legacy service-role
// JWT, checked in that order. Accepts either SUPABASE_SECRETKEY or
// SUPABASE_SECRET_KEY for the new-format key so users don't get tripped up
// by the underscore.
export function getSupabaseAdmin(): SupabaseClient {
  if (!_adminClient) {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL
    const key =
      process.env.SUPABASE_SECRETKEY ??
      process.env.SUPABASE_SECRET_KEY ??
      process.env.SUPABASE_SERVICE_KEY
    if (!url || !key) {
      throw new Error(
        'Missing NEXT_PUBLIC_SUPABASE_URL or a server key (SUPABASE_SECRETKEY / SUPABASE_SECRET_KEY / SUPABASE_SERVICE_KEY)'
      )
    }
    _adminClient = createClient(url, key)
  }
  return _adminClient
}

// Convenience export for server-side API routes and RSC.
// Binds `this` so method chaining (e.g. supabase.from(...).select(...)) works correctly.
export const supabase = new Proxy({} as SupabaseClient, {
  get(_target, prop) {
    const client = getSupabase()
    const value = client[prop as keyof SupabaseClient]
    return typeof value === 'function' ? (value as Function).bind(client) : value
  },
})

// Service-role convenience export for admin API routes.
export const supabaseAdmin = new Proxy({} as SupabaseClient, {
  get(_target, prop) {
    const client = getSupabaseAdmin()
    const value = client[prop as keyof SupabaseClient]
    return typeof value === 'function' ? (value as Function).bind(client) : value
  },
})

export type JointAngles = {
  right_elbow?: number
  left_elbow?: number
  right_shoulder?: number
  left_shoulder?: number
  right_knee?: number
  left_knee?: number
  right_hip?: number
  left_hip?: number
  // Wrist flexion (elbow → wrist → index_finger). Added in schema_version 2;
  // missing on legacy clips.
  right_wrist?: number
  left_wrist?: number
  hip_rotation?: number
  trunk_rotation?: number
}

export type Landmark = {
  id: number
  name: string
  x: number
  y: number
  z: number
  visibility: number
}

// Detected racket head position in normalized [0,1] coords (same space as
// Landmark.x/y). Null when no racket was detected with sufficient confidence
// on this frame. Added in schema_version 2; absent on legacy clips.
export type RacketHead = {
  x: number
  y: number
  confidence: number
} | null

export type PoseFrame = {
  frame_index: number
  timestamp_ms: number
  landmarks: Landmark[]
  joint_angles: JointAngles
  racket_head?: RacketHead
}

export type KeypointsJson = {
  fps_sampled: number
  frame_count: number
  frames: PoseFrame[]
  // Schema versions:
  //   undefined / 1 = legacy MediaPipe (no racket_head, no wrist angles).
  //   2             = MediaPipe Heavy with racket_head + wrist-flexion angle.
  //   3             = RTMPose-m via Railway. COCO-17 backbone remapped to
  //                   BlazePose-33 ids: face / hands / heels / foot_index
  //                   carry visibility=0; wrist-flexion joint angles
  //                   (right_wrist / left_wrist) are absent because the
  //                   index-finger landmarks aren't filled. Renderer and
  //                   joint-angle code handle this implicitly because they
  //                   gate on visibility>=cutoff and on landmark presence.
  schema_version?: 1 | 2 | 3
}

export type PhaseLabels = {
  preparation?: number
  loading?: number
  contact?: number
  follow_through?: number
  finish?: number
}

export type UserSession = {
  id: string
  blob_url: string
  shot_type: string | null
  keypoints_json: KeypointsJson | null
  analysis_result: Record<string, unknown> | null
  status: 'uploaded' | 'extracting' | 'analyzing' | 'complete' | 'error'
  error_message: string | null
  is_multi_shot: boolean
  segment_count: number
  created_at: string
  expires_at: string
}

export type Baseline = {
  id: string
  user_id: string
  label: string
  shot_type: 'forehand' | 'backhand' | 'serve' | 'volley' | 'slice'
  blob_url: string
  keypoints_json: KeypointsJson
  source_session_id: string | null
  is_active: boolean
  created_at: string
  replaced_at: string | null
}

// Telemetry row written by /api/analyze and /api/segments/.../analyze. Backs
// the post-launch "is the Djokovic downgrade a one-off or a pattern" SQL
// investigation. See lib/db/007_analysis_events.sql.
export type AnalysisEvent = {
  id: string
  user_id: string | null
  session_id: string | null
  segment_id: string | null
  created_at: string

  self_reported_tier: 'beginner' | 'intermediate' | 'competitive' | 'advanced' | null
  was_skipped: boolean
  handedness: 'right' | 'left' | null
  backhand_style: 'one_handed' | 'two_handed' | null
  primary_goal: string | null

  shot_type: string | null
  blob_url: string | null

  composite_metrics: Record<string, unknown> | null

  llm_assessed_tier: 'beginner' | 'intermediate' | 'competitive' | 'advanced' | null
  llm_coached_tier: 'beginner' | 'intermediate' | 'competitive' | 'advanced' | null
  llm_tier_downgrade: boolean

  capture_quality_flag:
    | 'green_side'
    | 'yellow_oblique'
    | 'red_front_or_back'
    | 'unknown'
    | null

  user_correction: 'correct' | 'too_easy' | 'too_hard' | null
  user_correction_note: string | null

  // new 2026-04: output shape telemetry for tier-rule rewrite validation
  response_token_count: number | null
  // new 2026-04: output shape telemetry for tier-rule rewrite validation
  response_tip_count: number | null
  // new 2026-04: output shape telemetry for tier-rule rewrite validation
  response_char_count: number | null
  // new 2026-04: output shape telemetry for tier-rule rewrite validation
  used_baseline_template: boolean
}

export type UserCorrection = NonNullable<AnalysisEvent['user_correction']>

export type VideoSegment = {
  id: string
  session_id: string
  segment_index: number
  shot_type: 'forehand' | 'backhand' | 'serve' | 'volley' | 'slice' | 'unknown' | 'idle'
  start_frame: number
  end_frame: number
  start_ms: number
  end_ms: number
  confidence: number
  label: string | null
  keypoints_json: KeypointsJson | null
  analysis_result: Record<string, unknown> | null
  created_at: string
}
