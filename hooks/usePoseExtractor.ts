'use client'

import { useCallback, useRef, useState } from 'react'
import { extractPoseFromVideo, type ExtractResult } from '@/lib/poseExtraction'

type Status = 'idle' | 'extracting' | 'done' | 'error'

export type UsePoseExtractorReturn = {
  extract: (file: File, fps?: number) => Promise<ExtractResult | null>
  progress: number
  status: Status
  isProcessing: boolean
  error: string | null
  abort: () => void
}

/**
 * Wrap extractPoseFromVideo with React state + the generation-counter
 * cancellation pattern UploadZone used: starting a new extract cancels any
 * in-flight one, and the stale run's resolved value is discarded.
 */
export function usePoseExtractor(): UsePoseExtractorReturn {
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)

  const generationRef = useRef(0)
  const controllerRef = useRef<AbortController | null>(null)

  const abort = useCallback(() => {
    controllerRef.current?.abort()
    controllerRef.current = null
  }, [])

  const extract = useCallback(async (file: File, fps?: number) => {
    // Supersede any in-flight run
    abort()
    const generation = ++generationRef.current
    const controller = new AbortController()
    controllerRef.current = controller

    setProgress(0)
    setStatus('extracting')
    setError(null)

    try {
      const result = await extractPoseFromVideo(file, {
        fps,
        onProgress: (pct) => {
          if (generationRef.current !== generation) return
          setProgress(pct)
        },
        abortSignal: controller.signal,
      })

      // Stale — a newer extract superseded us. Release the object URL since
      // nobody will claim it.
      if (generationRef.current !== generation) {
        if (result.objectUrl) URL.revokeObjectURL(result.objectUrl)
        return null
      }

      if (result.frames.length === 0) {
        // Could be aborted (objectUrl === null) or legitimately empty
        if (result.objectUrl === null) {
          setStatus('idle')
          return null
        }
        setStatus('error')
        setError('No pose detected in video. Try a clearer angle with your full body visible.')
        URL.revokeObjectURL(result.objectUrl)
        return null
      }

      setProgress(100)
      setStatus('done')
      return result
    } catch (err) {
      if (generationRef.current !== generation) return null
      setStatus('error')
      setError(err instanceof Error ? err.message : 'Pose extraction failed')
      return null
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
      }
    }
  }, [abort])

  return {
    extract,
    progress,
    status,
    isProcessing: status === 'extracting',
    error,
    abort,
  }
}
