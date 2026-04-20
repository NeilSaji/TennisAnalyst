import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import { sanitizePromptInput } from '@/lib/sanitize'
import type { UserCorrection } from '@/lib/supabase'

// PATCH /api/analysis-events/:id/feedback — user's thumbs-up/down on a
// coaching response. Writes go through the RLS-scoped server client (not the
// service-role admin client); the `analysis_events_update_own_correction`
// policy enforces that users can only patch rows they own.
//
// Response shape is deliberately minimal: { ok: true } on success, a plain
// error JSON on failure. We return 404 whenever RLS refuses the update so
// non-owners can't probe whether a given event id exists.

const ALLOWED_CORRECTIONS: readonly UserCorrection[] = ['correct', 'too_easy', 'too_hard']

type RouteCtx = { params: Promise<{ id: string }> }

export async function PATCH(request: NextRequest, ctx: RouteCtx) {
  const { id } = await ctx.params

  const supabase = await createClient()
  const { data: userData, error: userError } = await supabase.auth.getUser()
  if (userError || !userData?.user) {
    return NextResponse.json({ error: 'Not signed in' }, { status: 401 })
  }

  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  if (!body || typeof body !== 'object') {
    return NextResponse.json({ error: 'Invalid body' }, { status: 400 })
  }
  const { correction, note } = body as { correction?: unknown; note?: unknown }

  if (
    typeof correction !== 'string' ||
    !ALLOWED_CORRECTIONS.includes(correction as UserCorrection)
  ) {
    return NextResponse.json(
      { error: "correction must be one of 'correct', 'too_easy', 'too_hard'" },
      { status: 400 },
    )
  }

  // Note is optional. When present, we run it through the same sanitizer used
  // on LLM-prompt inputs so control chars / RTL overrides can't get persisted.
  let sanitizedNote: string | null = null
  if (note !== undefined && note !== null) {
    if (typeof note !== 'string') {
      return NextResponse.json({ error: 'note must be a string' }, { status: 400 })
    }
    sanitizedNote = sanitizePromptInput(note, 200)
  }

  // Use RLS-scoped update. The select().single() after the update turns an
  // unauthorized-or-missing row into a null data / row-not-found error, which
  // we surface as 404 (avoids leaking existence of other users' events).
  const { data, error } = await supabase
    .from('analysis_events')
    .update({
      user_correction: correction,
      user_correction_note: sanitizedNote,
    })
    .eq('id', id)
    .select('id')
    .single()

  if (error || !data) {
    return NextResponse.json({ error: 'Not found' }, { status: 404 })
  }

  return NextResponse.json({ ok: true })
}
