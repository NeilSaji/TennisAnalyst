// Shared sanitizer for user-supplied strings that flow verbatim into the LLM
// prompt or into telemetry columns that get surfaced back to the user. Strips
// newlines + control chars (so an attacker can't break out of their delimited
// section and inject new instructions) and caps length. Null return means
// "treat as missing" — callers can branch on that.
export function sanitizePromptInput(v: unknown, maxLen: number): string | null {
  if (typeof v !== 'string') return null
  const cleaned = v
    // Control chars (U+0000–U+001F, U+007F) including \n\r\t + unicode RTL overrides
    .replace(/[\u0000-\u001F\u007F\u200E\u200F\u202A-\u202E\u2066-\u2069]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, maxLen)
  return cleaned || null
}
