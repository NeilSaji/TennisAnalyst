-- Telemetry for every analyze-route call. This is the "read the data" substrate
-- for the Djokovic-failure-mode investigation: we need to know how often the
-- LLM disagrees with the user's self-reported tier, and whether that
-- correlates with capture quality, shot type, or anything else.
--
-- Writes happen from /api/analyze and /api/segments/.../analyze. Reads are
-- owner-scoped via RLS; the founder will SQL straight against this table in
-- the Supabase SQL editor to answer "is Djokovic a one-off or a real mode?"

CREATE TABLE IF NOT EXISTS analysis_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  session_id uuid,
  segment_id uuid,
  created_at timestamptz DEFAULT now(),

  -- Captured from user_metadata at the moment of analysis so we can replay
  -- history even after the user edits their profile.
  self_reported_tier text,        -- 'beginner' | 'intermediate' | 'competitive' | 'advanced' | NULL if skipped
  was_skipped boolean DEFAULT false,
  handedness text,                -- 'right' | 'left'
  backhand_style text,            -- 'one_handed' | 'two_handed'
  primary_goal text,

  -- Clip metadata
  shot_type text,
  blob_url text,

  -- Raw per-analysis signals. jsonb so we can evolve the feature set without a
  -- migration — the classifier build in month 2 will read historical rows.
  composite_metrics jsonb,

  -- LLM bookkeeping. llm_assessed_tier is what the model THOUGHT the tier was
  -- (emitted via a structured trailer and stripped from user output), so we
  -- can see disagreement even when the defanged RECONCILE_RULE prevents the
  -- model from acting on it.
  llm_assessed_tier text,         -- NULL when model didn't emit one
  llm_coached_tier text,          -- tier actually coached to in the stream
  llm_tier_downgrade boolean DEFAULT false,

  -- Populated by a later phase: railway camera_classifier plumbs this in as
  -- telemetry-only. NULL until that lands.
  capture_quality_flag text,      -- 'green_side' | 'yellow_oblique' | 'red_front_or_back' | 'unknown'

  -- Thumbs button on the analyze panel. 'correct' = coach was right,
  -- 'too_easy' / 'too_hard' = tier mismatch in user's opinion.
  user_correction text,
  user_correction_note text
);

-- Most common queries: recent events per user, disagreement-only filter.
CREATE INDEX IF NOT EXISTS analysis_events_user_idx ON analysis_events(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS analysis_events_downgrade_idx
  ON analysis_events(created_at DESC)
  WHERE llm_tier_downgrade = true;

ALTER TABLE analysis_events ENABLE ROW LEVEL SECURITY;

-- Users can read their own events (for future "your analysis history" UI).
CREATE POLICY "analysis_events_select_own"
  ON analysis_events FOR SELECT
  USING (auth.uid() = user_id);

-- Users can UPDATE only their own user_correction fields via the feedback API
-- (RLS enforces ownership; the column list is enforced at the API layer).
CREATE POLICY "analysis_events_update_own_correction"
  ON analysis_events FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Inserts go through the service-role client from the analyze routes, so no
-- INSERT policy is needed here — anon/authed roles cannot insert.

-- Convenience view for the founder's SQL spelunking.
CREATE OR REPLACE VIEW v_tier_disagreement AS
  SELECT
    created_at,
    user_id,
    self_reported_tier,
    llm_assessed_tier,
    llm_coached_tier,
    shot_type,
    capture_quality_flag,
    user_correction
  FROM analysis_events
  WHERE llm_tier_downgrade = true
     OR (self_reported_tier IS NOT NULL
         AND llm_assessed_tier IS NOT NULL
         AND self_reported_tier <> llm_assessed_tier)
  ORDER BY created_at DESC;
