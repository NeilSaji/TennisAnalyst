// Thin streaming adapter over Gemini's OpenAI-compatible endpoint.
// Docs: https://ai.google.dev/gemini-api/docs/openai
//
// We don't pull in the openai SDK — `fetch` + SSE parsing is ~30 lines and
// avoids an extra dependency for two routes.

const GEMINI_BASE_URL =
  'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'

export type ChatMessage = {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export type StreamOptions = {
  model?: string
  systemPrompt?: string
  messages: ChatMessage[]
  maxTokens?: number
  apiKey?: string
}

/**
 * Streams text deltas from Gemini Flash.
 * Throws synchronously on config errors; mid-stream network errors propagate
 * through the async iterator and should be caught by the caller.
 */
export async function* streamGemini(opts: StreamOptions): AsyncGenerator<string> {
  const apiKey = opts.apiKey ?? process.env.GEMINI_API_KEY
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not set')
  }

  const model = opts.model ?? 'gemini-2.5-flash'
  const msgs: ChatMessage[] = []
  if (opts.systemPrompt) {
    msgs.push({ role: 'system', content: opts.systemPrompt })
  }
  msgs.push(...opts.messages)

  // Use Gemini's native `x-goog-api-key` header instead of `Authorization: Bearer`.
  // Vercel's runtime was injecting an extra Authorization-equivalent credential
  // somewhere in the chain, causing Gemini to reject with "Multiple authentication
  // credentials received". x-goog-api-key sidesteps that entirely.
  const res = await fetch(GEMINI_BASE_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-goog-api-key': apiKey,
    },
    body: JSON.stringify({
      model,
      messages: msgs,
      max_tokens: opts.maxTokens ?? 1024,
      stream: true,
    }),
  })

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => '')
    throw new Error(`Gemini ${res.status}: ${text.slice(0, 300) || res.statusText}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE frames are separated by double newlines. Process complete frames
    // and retain the partial tail for the next iteration.
    let boundary: number
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, boundary).trim()
      buffer = buffer.slice(boundary + 2)
      if (!frame.startsWith('data:')) continue
      const data = frame.slice(5).trim()
      if (data === '[DONE]') return
      try {
        const parsed = JSON.parse(data)
        const delta = parsed?.choices?.[0]?.delta?.content
        if (typeof delta === 'string' && delta.length > 0) {
          yield delta
        }
      } catch {
        // Silently drop malformed frames — the stream keeps going.
      }
    }
  }
}
