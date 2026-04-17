#!/usr/bin/env node
// One-off utility: merge all pro_swings from one pro name onto another, then
// delete the old pro row. Uses the service-role key from .env so it bypasses
// the RLS policies that lock writes on pros/pro_swings.
//
//   node scripts/merge-pro-names.mjs "Novak Djokavic" "Novak Djokovic"
//
// First arg = bad name (will be removed), second arg = good name (target).
// Name match is case-sensitive and exact. Dry-run by default; pass --apply to commit.

import { createClient } from '@supabase/supabase-js'
import { readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'

function loadEnvFile(path) {
  if (!existsSync(path)) return
  const raw = readFileSync(path, 'utf8')
  for (const line of raw.split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/)
    if (!m) continue
    const key = m[1]
    let val = m[2]
    if (val.startsWith('"') && val.endsWith('"')) val = val.slice(1, -1)
    process.env[key] = val
  }
}

loadEnvFile(join(process.cwd(), '.env'))

const args = process.argv.slice(2).filter((a) => a !== '--apply')
const apply = process.argv.includes('--apply')
const [fromName, toName] = args

if (!fromName || !toName) {
  console.error('Usage: node scripts/merge-pro-names.mjs "Bad Name" "Good Name" [--apply]')
  process.exit(1)
}

const url = process.env.NEXT_PUBLIC_SUPABASE_URL
const key = process.env.SUPABASE_SERVICE_KEY
if (!url || !key) {
  console.error('Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_KEY in .env')
  process.exit(1)
}

const supabase = createClient(url, key)

async function main() {
  console.log(`\nMerging "${fromName}" → "${toName}"  (${apply ? 'APPLY' : 'dry run'})\n`)

  const { data: fromPros, error: fromErr } = await supabase
    .from('pros')
    .select('id, name')
    .eq('name', fromName)
  if (fromErr) throw fromErr

  const { data: toPros, error: toErr } = await supabase
    .from('pros')
    .select('id, name')
    .eq('name', toName)
  if (toErr) throw toErr

  console.log(`"${fromName}" rows: ${fromPros.length}`)
  console.log(`"${toName}" rows:  ${toPros.length}`)

  if (fromPros.length === 0) {
    console.log('Nothing to merge — bad-name rows not found.')
    return
  }

  let targetProId
  if (toPros.length > 0) {
    targetProId = toPros[0].id
  } else {
    console.log(`No "${toName}" row yet — will rename one "${fromName}" row to "${toName}".`)
    if (apply) {
      const { data: renamed, error: renameErr } = await supabase
        .from('pros')
        .update({ name: toName })
        .eq('id', fromPros[0].id)
        .select('id')
        .single()
      if (renameErr) throw renameErr
      targetProId = renamed.id
      fromPros.shift()
    } else {
      targetProId = '<renamed-row-id>'
    }
  }

  const fromIds = fromPros.map((p) => p.id)
  if (fromIds.length === 0) {
    console.log('Done — only rename was needed.')
    return
  }

  const { data: swings, error: swingErr } = await supabase
    .from('pro_swings')
    .select('id, pro_id, shot_type, video_url')
    .in('pro_id', fromIds)
  if (swingErr) throw swingErr

  console.log(`\nSwings to re-point: ${swings.length}`)
  for (const s of swings) console.log(`  ${s.id}  ${s.shot_type}  ${s.video_url}`)

  if (!apply) {
    console.log(`\nDry run — would:`)
    console.log(`  UPDATE pro_swings SET pro_id='${targetProId}' WHERE pro_id IN (${fromIds.map((x) => `'${x}'`).join(', ')})`)
    console.log(`  DELETE FROM pros WHERE id IN (${fromIds.map((x) => `'${x}'`).join(', ')})`)
    console.log(`\nRe-run with --apply to commit.`)
    return
  }

  const { error: updErr } = await supabase
    .from('pro_swings')
    .update({ pro_id: targetProId })
    .in('pro_id', fromIds)
  if (updErr) throw updErr
  console.log(`Updated ${swings.length} swings.`)

  const { error: delErr } = await supabase.from('pros').delete().in('id', fromIds)
  if (delErr) throw delErr
  console.log(`Deleted ${fromIds.length} stale pro row(s).`)

  console.log('\nDone.')
}

main().catch((err) => {
  console.error('\nFailed:', err.message ?? err)
  process.exit(1)
})
