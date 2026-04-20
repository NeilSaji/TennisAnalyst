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

export type PoseFrame = {
  frame_index: number
  timestamp_ms: number
  landmarks: Landmark[]
  joint_angles: JointAngles
}

export type KeypointsJson = {
  fps_sampled: number
  frame_count: number
  frames: PoseFrame[]
}

export type PhaseLabels = {
  preparation?: number
  loading?: number
  contact?: number
  follow_through?: number
  finish?: number
}

export type Pro = {
  id: string
  name: string
  nationality: string | null
  ranking: number | null
  bio: string | null
  profile_image_url: string | null
  created_at: string
}

export type ProSwing = {
  id: string
  pro_id: string
  shot_type: 'forehand' | 'backhand' | 'serve' | 'volley' | 'slice'
  video_url: string
  thumbnail_url: string | null
  keypoints_json: KeypointsJson
  fps: number
  frame_count: number | null
  duration_ms: number | null
  phase_labels: PhaseLabels
  metadata: Record<string, unknown>
  created_at: string
  pros?: Pro
}

export type UserSession = {
  id: string
  blob_url: string
  shot_type: string | null
  keypoints_json: KeypointsJson | null
  analysis_result: Record<string, unknown> | null
  similarity_scores: Record<string, unknown> | null
  matched_pro_swing_id: string | null
  status: 'uploaded' | 'extracting' | 'analyzing' | 'complete' | 'error'
  error_message: string | null
  is_multi_shot: boolean
  segment_count: number
  created_at: string
  expires_at: string
}

export type Baseline = {
  id: string
  device_id: string
  user_id: string | null
  label: string
  shot_type: 'forehand' | 'backhand' | 'serve' | 'volley' | 'slice'
  blob_url: string
  keypoints_json: KeypointsJson
  source_session_id: string | null
  is_active: boolean
  created_at: string
  replaced_at: string | null
}

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
  matched_pro_swing_id: string | null
  created_at: string
}
