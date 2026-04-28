import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import type { KeypointsJson } from '@/lib/supabase'
import type { StreamedSwing } from '@/lib/liveSwingDetector'

// We don't bring in fake-indexeddb (not a project dependency) — instead we
// provide a tiny in-memory shim with the exact subset of the IndexedDB
// surface that liveSessionRecovery uses (open/onupgradeneeded, transaction,
// objectStore.put/get/delete). Behaves synchronously under the hood and
// dispatches handlers via queueMicrotask so the module-under-test still
// awaits real Promises.

type StoreRow = { id: string; [k: string]: unknown }

// Helper that mimics the IndexedDB request lifecycle: the consumer sets
// onsuccess/onerror after our open()/put()/get()/delete() returns, then we
// fire those handlers on the next macrotask so the consumer's handler is
// definitely attached.
function fireOnNextTick(fn: () => void) {
  // setTimeout(0) instead of queueMicrotask so we settle *after* the
  // synchronous code that attaches handlers runs to completion.
  setTimeout(fn, 0)
}

class FakeRequest<T = unknown> {
  result: T | undefined
  error: unknown = null
  onsuccess: (() => void) | null = null
  onerror: (() => void) | null = null
  onupgradeneeded: (() => void) | null = null
}

class FakeObjectStore {
  constructor(private rows: Map<string, StoreRow>, private tx: FakeTransaction) {}
  put(record: StoreRow) {
    this.rows.set(record.id, record)
    const r = new FakeRequest<unknown>()
    this.tx.markPending()
    fireOnNextTick(() => {
      r.result = record.id
      r.onsuccess?.()
      this.tx.markDone()
    })
    return r
  }
  get(key: string) {
    const r = new FakeRequest<StoreRow | undefined>()
    this.tx.markPending()
    const value = this.rows.get(key)
    fireOnNextTick(() => {
      r.result = value
      r.onsuccess?.()
      this.tx.markDone()
    })
    return r
  }
  delete(key: string) {
    this.rows.delete(key)
    const r = new FakeRequest<undefined>()
    this.tx.markPending()
    fireOnNextTick(() => {
      r.onsuccess?.()
      this.tx.markDone()
    })
    return r
  }
}

class FakeTransaction {
  oncomplete: (() => void) | null = null
  onerror: (() => void) | null = null
  onabort: (() => void) | null = null
  private pending = 0
  private autoCompleteScheduled = false
  constructor(private rows: Map<string, StoreRow>) {
    // If no work was queued before the next tick, fire oncomplete then —
    // matches the real IDB behavior of completing an empty transaction.
    this.scheduleAutoComplete()
  }
  private scheduleAutoComplete() {
    if (this.autoCompleteScheduled) return
    this.autoCompleteScheduled = true
    fireOnNextTick(() => {
      if (this.pending === 0) this.oncomplete?.()
    })
  }
  markPending() {
    this.pending++
  }
  markDone() {
    this.pending--
    if (this.pending === 0) {
      // Wait one more tick so consecutive ops queued in the same callback
      // don't double-fire oncomplete.
      fireOnNextTick(() => {
        if (this.pending === 0) this.oncomplete?.()
      })
    }
  }
  objectStore() {
    return new FakeObjectStore(this.rows, this)
  }
}

class FakeDb {
  closed = false
  objectStoreNames = {
    has: (name: string) => this.stores.has(name),
    contains: (name: string) => this.stores.has(name),
  }
  constructor(private stores: Map<string, Map<string, StoreRow>>) {}
  transaction(_name: string) {
    const rows = this.stores.get(_name) ?? new Map()
    if (!this.stores.has(_name)) this.stores.set(_name, rows)
    return new FakeTransaction(rows)
  }
  close() {
    this.closed = true
  }
  createObjectStore(name: string) {
    const m = new Map<string, StoreRow>()
    this.stores.set(name, m)
    return new FakeObjectStore(m, new FakeTransaction(m))
  }
}

function installFakeIDB() {
  const stores = new Map<string, Map<string, StoreRow>>()
  // First open call needs to fire onupgradeneeded so the module can create
  // the object store.
  let upgraded = false
  const idb = {
    open(_name: string, _version: number) {
      const r = new FakeRequest<FakeDb>()
      const db = new FakeDb(stores)
      fireOnNextTick(() => {
        if (!upgraded) {
          upgraded = true
          r.result = db
          r.onupgradeneeded?.()
        }
        r.result = db
        r.onsuccess?.()
      })
      return r as unknown as IDBOpenDBRequest
    },
  }
  // @ts-expect-error patching global for tests
  globalThis.indexedDB = idb
  return {
    reset: () => {
      stores.clear()
      upgraded = false
    },
  }
}

let fake: ReturnType<typeof installFakeIDB>

describe('liveSessionRecovery', () => {
  beforeEach(() => {
    vi.resetModules()
    fake = installFakeIDB()
  })

  afterEach(() => {
    fake.reset()
    // @ts-expect-error
    delete globalThis.indexedDB
  })

  function makeBlob(text = 'fake-video-bytes'): Blob {
    return new Blob([text], { type: 'video/webm' })
  }

  function makeKeypoints(): KeypointsJson {
    return {
      fps_sampled: 15,
      frame_count: 0,
      frames: [],
      schema_version: 2,
    }
  }

  function makeSwings(n: number): StreamedSwing[] {
    return Array.from({ length: n }, (_, i) => ({
      swingIndex: i + 1,
      startFrameIndex: i * 10,
      endFrameIndex: i * 10 + 9,
      peakFrameIndex: i * 10 + 5,
      startMs: i * 1000,
      endMs: i * 1000 + 900,
      frames: [],
    }))
  }

  it('round-trips blob, keypoints, swings, and shotType', async () => {
    const { saveOrphanedSession, getOrphanedSession } = await import(
      '@/lib/liveSessionRecovery'
    )
    const blob = makeBlob('hello-blob')
    const kp = makeKeypoints()
    const swings = makeSwings(3)
    await saveOrphanedSession(blob, kp, swings, 'forehand')

    const found = await getOrphanedSession()
    expect(found).not.toBeNull()
    expect(found!.shotType).toBe('forehand')
    expect(found!.swings).toHaveLength(3)
    expect(found!.swings[0].swingIndex).toBe(1)
    expect(found!.keypoints.fps_sampled).toBe(15)
    expect(found!.blob).toBeInstanceOf(Blob)
    // Confirm the persisted Blob carries the same identity (in-memory IDB
    // stub doesn't structured-clone, so reference equality is what we get).
    expect(found!.blob).toBe(blob)
  })

  it('drops a session older than 24h on read (TTL eviction)', async () => {
    // We mock Date.now (NOT vi.useFakeTimers, which would also fake the
    // setTimeout we rely on inside the fake IDB). Save at T0, then set the
    // clock to T0 + 25h so the read-side TTL fires.
    const T0 = new Date('2026-04-20T00:00:00Z').getTime()
    const dateSpy = vi.spyOn(Date, 'now').mockReturnValue(T0)
    try {
      const { saveOrphanedSession, getOrphanedSession } = await import(
        '@/lib/liveSessionRecovery'
      )
      await saveOrphanedSession(makeBlob(), makeKeypoints(), [], 'serve')

      // Move the wall clock forward by 25 hours — past the 24h TTL.
      dateSpy.mockReturnValue(T0 + 25 * 60 * 60 * 1000)
      const found = await getOrphanedSession()
      expect(found).toBeNull()
    } finally {
      dateSpy.mockRestore()
    }
  })

  it('keeps a session younger than 24h on read', async () => {
    const T0 = new Date('2026-04-20T00:00:00Z').getTime()
    const dateSpy = vi.spyOn(Date, 'now').mockReturnValue(T0)
    try {
      const { saveOrphanedSession, getOrphanedSession } = await import(
        '@/lib/liveSessionRecovery'
      )
      await saveOrphanedSession(makeBlob(), makeKeypoints(), [], 'backhand')

      // Half a day later — still inside TTL.
      dateSpy.mockReturnValue(T0 + 23 * 60 * 60 * 1000)
      const found = await getOrphanedSession()
      expect(found).not.toBeNull()
      expect(found!.shotType).toBe('backhand')
    } finally {
      dateSpy.mockRestore()
    }
  })

  it('saving a new orphan replaces the previous one (single-row semantics)', async () => {
    const { saveOrphanedSession, getOrphanedSession } = await import(
      '@/lib/liveSessionRecovery'
    )
    const first = makeBlob('first')
    const second = makeBlob('second')
    await saveOrphanedSession(first, makeKeypoints(), [], 'forehand')
    await saveOrphanedSession(second, makeKeypoints(), makeSwings(2), 'serve')

    const found = await getOrphanedSession()
    expect(found).not.toBeNull()
    expect(found!.shotType).toBe('serve')
    expect(found!.swings).toHaveLength(2)
    // The second save should have replaced the first — the persisted blob
    // is the second one, not the first.
    expect(found!.blob).toBe(second)
    expect(found!.blob).not.toBe(first)
  })

  it('clearOrphanedSession() makes the next read return null', async () => {
    const { saveOrphanedSession, getOrphanedSession, clearOrphanedSession } =
      await import('@/lib/liveSessionRecovery')
    await saveOrphanedSession(makeBlob(), makeKeypoints(), [], 'volley')
    expect(await getOrphanedSession()).not.toBeNull()

    await clearOrphanedSession()
    expect(await getOrphanedSession()).toBeNull()
  })

  it('returns null and resolves cleanly when IndexedDB is unavailable', async () => {
    // Tear down the fake before importing — the module must handle no-IDB
    // environments (SSR / locked-down browsers) without throwing.
    // @ts-expect-error
    delete globalThis.indexedDB
    const { saveOrphanedSession, getOrphanedSession, clearOrphanedSession } =
      await import('@/lib/liveSessionRecovery')

    await expect(
      saveOrphanedSession(makeBlob(), makeKeypoints(), [], 'forehand'),
    ).resolves.toBeUndefined()
    await expect(getOrphanedSession()).resolves.toBeNull()
    await expect(clearOrphanedSession()).resolves.toBeUndefined()
  })
})
