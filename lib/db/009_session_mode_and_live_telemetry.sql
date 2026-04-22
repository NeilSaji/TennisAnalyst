-- Adds support for the /live capture mode.
--
-- PREREQUISITE: migration 008 must be applied before this one. 009 does not
-- touch any columns 008 added, but the /api/live-coach telemetry inserts
-- populate composite_metrics.live_batch and rely on 008's response_*
-- columns being present on analysis_events for the backfill UPDATE.
--
-- Schema changes:
--   1. user_sessions gains session_mode ('upload' | 'live'). Default 'upload'
--      preserves every existing row as-is and every upload-flow session going
--      forward; only the new POST /api/sessions/live route sets 'live'.
--   2. A convenience view v_live_batch_calibration exposes the per-batch
--      telemetry stored under analysis_events.composite_metrics.live_batch.
--
-- Live-batch shape written by /api/live-coach:
--   composite_metrics = {
--     "live_batch": {
--       "batch_index":        int,
--       "swing_count":        int,
--       "session_duration_ms": int,
--       "baseline_present":   bool
--     }
--   }

ALTER TABLE user_sessions
  ADD COLUMN IF NOT EXISTS session_mode text NOT NULL DEFAULT 'upload'
    CHECK (session_mode IN ('upload','live'));

CREATE INDEX IF NOT EXISTS user_sessions_mode_idx
  ON user_sessions(session_mode, created_at DESC);

CREATE OR REPLACE VIEW v_live_batch_calibration AS
SELECT
  (composite_metrics->'live_batch'->>'batch_index')::int          AS batch_index,
  (composite_metrics->'live_batch'->>'swing_count')::int          AS swing_count,
  (composite_metrics->'live_batch'->>'session_duration_ms')::int  AS session_duration_ms,
  COALESCE(composite_metrics->'live_batch'->>'baseline_present','false')::boolean
                                                                  AS baseline_present,
  response_token_count,
  response_char_count,
  llm_coached_tier,
  shot_type,
  created_at,
  user_id
FROM analysis_events
WHERE composite_metrics ? 'live_batch'
ORDER BY created_at DESC;
