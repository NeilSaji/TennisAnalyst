-- User baselines: "best day" swings pinned for long-term comparison.
-- Keyed by device_id (capability token in cookie + localStorage) until auth lands.
-- Follows the permissive-RLS + enforce-ownership-in-API pattern from 001_rls_policies.sql.

CREATE TABLE IF NOT EXISTS user_baselines (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  device_id text NOT NULL,
  user_id uuid,  -- reserved for future Supabase Auth migration; nullable today
  label text NOT NULL DEFAULT 'My baseline',
  shot_type text NOT NULL CHECK (shot_type IN ('forehand','backhand','serve','volley','slice')),
  blob_url text NOT NULL,
  keypoints_json jsonb NOT NULL,
  source_session_id uuid REFERENCES user_sessions(id) ON DELETE SET NULL,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  replaced_at timestamptz
);

CREATE INDEX IF NOT EXISTS user_baselines_device_idx ON user_baselines(device_id);
CREATE INDEX IF NOT EXISTS user_baselines_active_idx ON user_baselines(device_id, is_active)
  WHERE is_active = true;

-- "One active baseline per (device, shot_type)" is enforced in the API (atomic
-- swap on replace), not as a unique constraint — keeps replacement history queryable.

ALTER TABLE user_baselines ENABLE ROW LEVEL SECURITY;

CREATE POLICY "user_baselines_select_public"
  ON user_baselines FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "user_baselines_insert_public"
  ON user_baselines FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "user_baselines_update_public"
  ON user_baselines FOR UPDATE
  TO anon, authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "user_baselines_delete_public"
  ON user_baselines FOR DELETE
  TO anon, authenticated
  USING (true);

CREATE POLICY "user_baselines_all_service"
  ON user_baselines FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- No expires_at, no pg_cron purge — baselines are long-lived by design.
