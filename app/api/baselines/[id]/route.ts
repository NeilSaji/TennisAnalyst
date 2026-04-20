import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import type { SupabaseClient } from '@supabase/supabase-js'

type RouteCtx = { params: Promise<{ id: string }> }

async function authOrDeny() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) {
    return { error: NextResponse.json({ error: 'Not signed in' }, { status: 401 }) }
  }
  return { supabase, user }
}

async function loadOwned(supabase: SupabaseClient, id: string) {
  const { data, error } = await supabase
    .from('user_baselines')
    .select('*')
    .eq('id', id)
    .single()
  if (error || !data) {
    return { error: NextResponse.json({ error: 'Baseline not found' }, { status: 404 }) }
  }
  return { baseline: data }
}

export async function GET(_request: NextRequest, ctx: RouteCtx) {
  const { id } = await ctx.params
  const auth = await authOrDeny()
  if ('error' in auth) return auth.error
  const loaded = await loadOwned(auth.supabase, id)
  if ('error' in loaded) return loaded.error
  return NextResponse.json({ baseline: loaded.baseline })
}

// PUT — body { label?: string, isActive?: true }
export async function PUT(request: NextRequest, ctx: RouteCtx) {
  const { id } = await ctx.params
  const auth = await authOrDeny()
  if ('error' in auth) return auth.error
  const { supabase } = auth

  const loaded = await loadOwned(supabase, id)
  if ('error' in loaded) return loaded.error
  const { baseline } = loaded

  let body
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const patch: Record<string, unknown> = {}
  if (typeof body?.label === 'string' && body.label.trim()) {
    patch.label = body.label.trim().slice(0, 120)
  }

  if (body?.isActive === true) {
    const now = new Date().toISOString()
    // Deactivate siblings (same shot_type, not this row). RLS scopes to user.
    const { error: sibErr } = await supabase
      .from('user_baselines')
      .update({ is_active: false, replaced_at: now })
      .eq('shot_type', baseline.shot_type)
      .eq('is_active', true)
      .neq('id', id)
    if (sibErr) {
      return NextResponse.json({ error: sibErr.message }, { status: 500 })
    }
    patch.is_active = true
    patch.replaced_at = null
  }

  if (Object.keys(patch).length === 0) {
    return NextResponse.json({ baseline })
  }

  const { data, error } = await supabase
    .from('user_baselines')
    .update(patch)
    .eq('id', id)
    .select()
    .single()

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  return NextResponse.json({ baseline: data })
}

export async function DELETE(_request: NextRequest, ctx: RouteCtx) {
  const { id } = await ctx.params
  const auth = await authOrDeny()
  if ('error' in auth) return auth.error

  const { error } = await auth.supabase
    .from('user_baselines')
    .delete()
    .eq('id', id)

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  return NextResponse.json({ ok: true })
}
