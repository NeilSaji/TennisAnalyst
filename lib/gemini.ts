// Streaming adapter over Gemini's NATIVE REST endpoint (not the OpenAI-compat
// shim). The compat endpoint requires `Authorization: Bearer`, and something
// in Vercel's runtime injects an extra credential, causing Gemini to reject
// every request with "Multiple authentication credentials received". The
// native endpoint authenticates via `x-goog-api-key` and doesn't have that
// problem.
//
// Docs:
//   https://ai.google.dev/api/generate-content#method:-models.streamgeneratecontent

const GEMINI_MODEL_DEFAULT = 'gemini-2.0-flash'

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

type GeminiPart = { text: string }
type GeminiContent = { role: 'user' | 'model'; parts: GeminiPart[] }

/**
 * Streams text deltas from Gemini.
 * Throws synchronously on config errors; mid-stream network errors propagate
 * through the async iterator and should be caught by the caller.
 */
export async function* streamGemini(opts: StreamOptions): AsyncGenerator<string> {
  const apiKey = opts.apiKey ?? process.env.GEMINI_API_KEY
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not set')
  }

  const model = opts.model ?? GEMINI_MODEL_DEFAULT
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`

  const contents: GeminiContent[] = opts.messages.map((m) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }))

  const body: Record<string, unknown> = {
    contents,
    generationConfig: { maxOutputTokens: opts.maxTokens ?? 1024 },
  }
  if (opts.systemPrompt) {
    body.systemInstruction = { parts: [{ text: opts.systemPrompt }] }
  }

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-goog-api-key': apiKey,
    },
    body: JSON.stringify(body),
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

    let boundary: number
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, boundary).trim()
      buffer = buffer.slice(boundary + 2)
      if (!frame.startsWith('data:')) continue
      const data = frame.slice(5).trim()
      if (!data || data === '[DONE]') continue
      try {
        const parsed = JSON.parse(data)
        // Each SSE frame carries a candidates[] array; walk the parts for text.
        const parts = parsed?.candidates?.[0]?.content?.parts
        if (Array.isArray(parts)) {
          for (const part of parts) {
            if (typeof part?.text === 'string' && part.text.length > 0) {
              yield part.text
            }
          }
        }
      } catch {
        // Silently drop malformed frames — the stream keeps going.
      }
    }
  }
}
