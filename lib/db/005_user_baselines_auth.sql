-- Morph user_baselines from device-id capability tokens to Supabase Auth ownership.
-- Precondition: 004_user_baselines.sql already applied, table currently empty.
-- If you have any real rows in user_baselines, delete or migrate them first —
-- the NOT NULL constraint on user_id will reject pre-existing NULLs.

-- 1. Drop the permissive RLS policies from 004. New policies enforce auth.uid().
DROP POLICY IF EXISTS "user_baselines_select_public" ON user_baselines;
DROP POLICY IF EXISTS "user_baselines_insert_public" ON user_baselines;
DROP POLICY IF EXISTS "user_baselines_update_public" ON user_baselines;
DROP POLICY IF EXISTS "user_baselines_delete_public" ON user_baselines;
-- Service-role policy stays (admin pathways still need it).

-- 2. Drop device-id column + its indexes. Auth-based ownership replaces them.
DROP INDEX IF EXISTS user_baselines_device_idx;
DROP INDEX IF EXISTS user_baselines_active_idx;
ALTER TABLE user_baselines DROP COLUMN IF EXISTS device_id;

-- 3. user_id becomes the ownership key. FK cascades so deleting an auth user
--    nukes their baselines.
ALTER TABLE user_baselines
  ALTER COLUMN user_id SET NOT NULL,
  ADD CONSTRAINT user_baselines_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS user_baselines_user_idx ON user_baselines(user_id);
CREATE INDEX IF NOT EXISTS user_baselines_user_active_idx
  ON user_baselines(user_id, is_active)
  WHERE is_active = true;

-- 4. New RLS: each row is visible/mutable only by its owner. Proper ownership
--    at the DB layer, not enforced in the API.
CREATE POLICY "user_baselines_select_own"
  ON user_baselines FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "user_baselines_insert_own"
  ON user_baselines FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "user_baselines_update_own"
  ON user_baselines FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "user_baselines_delete_own"
  ON user_baselines FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- Note: no anon policies. Anonymous visitors can still use /analyze (which
-- touches user_sessions, not user_baselines). The moment they try to read or
-- write a baseline without a session, PostgREST returns 401/empty.
