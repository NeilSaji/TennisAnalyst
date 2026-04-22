'use client'

interface LiveSwingCounterProps {
  swingCount: number
  nextBatchAt?: number
}

export default function LiveSwingCounter({ swingCount, nextBatchAt }: LiveSwingCounterProps) {
  return (
    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 text-sm font-medium">
      <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
      <span>
        {swingCount} {swingCount === 1 ? 'swing' : 'swings'} detected
      </span>
      {nextBatchAt !== undefined && swingCount < nextBatchAt ? (
        <span className="text-emerald-300/60">· next tip at {nextBatchAt}</span>
      ) : null}
    </div>
  )
}
