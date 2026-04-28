/**
 * Orphaned-live-session recovery store.
 *
 * When a live session finishes recording and the upload-or-save step fails,
 * we don't want the player to lose the recording. We persist the blob,
 * its computed keypoints, and the detected swing list to IndexedDB so the
 * client can offer to resume on retry / cold start.
 *
 * Single-row semantics: only the most recent orphan is kept. A second save
 * replaces the first — runaway storage from repeated record-then-fail loops
 * would be much worse than losing the previous orphan.
 *
 * TTL: anything older than 24h is dropped on read. Stale recordings are
 * almost never useful and the blob alone can be 5–30MB.
 *
 * Implementation note: we use raw IndexedDB (no `idb` dependency) to avoid
 * adding a new package; the surface we need is small. In environments with
 * no `indexedDB` (SSR / older browsers / locked-down configs) every method
 * resolves to a no-op rather than throwing — recovery is a nice-to-have,
 * not a hard requirement.
 */

import type { KeypointsJson } from '@/lib/supabase'
import type { StreamedSwing } from '@/lib/liveSwingDetector'

const DB_NAME = 'tennis-live-recovery'
const DB_VERSION = 1
const STORE_NAME = 'orphans'
const SINGLETON_KEY = 'current'
const TTL_MS = 24 * 60 * 60 * 1000

export type OrphanedSession = {
  blob: Blob
  keypoints: KeypointsJson
  swings: StreamedSwing[]
  shotType: string
  savedAt: number
}

type StoredOrphan = OrphanedSession & {
  id: string
}

function getIdb(): IDBFactory | null {
  if (typeof indexedDB === 'undefined') return null
  return indexedDB
}

function openDb(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    const idb = getIdb()
    if (!idb) {
      resolve(null)
      return
    }
    let request: IDBOpenDBRequest
    try {
      request = idb.open(DB_NAME, DB_VERSION)
    } catch {
      resolve(null)
      return
    }
    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' })
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => resolve(null)
    request.onblocked = () => resolve(null)
  })
}

function txDone(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve) => {
    tx.oncomplete = () => resolve()
    tx.onerror = () => resolve()
    tx.onabort = () => resolve()
  })
}

function reqDone<T>(req: IDBRequest<T>): Promise<T | null> {
  return new Promise((resolve) => {
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => resolve(null)
  })
}

/**
 * Persist a single orphaned session, replacing any prior orphan.
 * Resolves once the write completes (or silently if IndexedDB is unavailable).
 */
export async function saveOrphanedSession(
  blob: Blob,
  keypoints: KeypointsJson,
  swings: StreamedSwing[],
  shotType: string,
): Promise<void> {
  const db = await openDb()
  if (!db) return
  try {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    const record: StoredOrphan = {
      id: SINGLETON_KEY,
      blob,
      keypoints,
      swings,
      shotType,
      savedAt: Date.now(),
    }
    store.put(record)
    await txDone(tx)
  } finally {
    db.close()
  }
}

/**
 * Read the currently-persisted orphan, if any. Returns null if no orphan
 * exists or if the orphan is older than the TTL (24h) — stale entries are
 * cleared as a side effect so subsequent reads are O(1).
 */
export async function getOrphanedSession(): Promise<OrphanedSession | null> {
  const db = await openDb()
  if (!db) return null
  try {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const record = (await reqDone(store.get(SINGLETON_KEY))) as StoredOrphan | null
    await txDone(tx)
    if (!record) return null
    const age = Date.now() - (record.savedAt ?? 0)
    if (!Number.isFinite(record.savedAt) || age > TTL_MS) {
      // Fire-and-forget eviction. Don't await — the caller doesn't care.
      void clearOrphanedSession()
      return null
    }
    return {
      blob: record.blob,
      keypoints: record.keypoints,
      swings: record.swings,
      shotType: record.shotType,
      savedAt: record.savedAt,
    }
  } finally {
    db.close()
  }
}

/**
 * Drop the currently-persisted orphan (if any). Idempotent.
 */
export async function clearOrphanedSession(): Promise<void> {
  const db = await openDb()
  if (!db) return
  try {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    store.delete(SINGLETON_KEY)
    await txDone(tx)
  } finally {
    db.close()
  }
}

// Exposed for tests so they can drive the TTL boundary deterministically
// without sleeping. Not part of the public API.
export const __INTERNAL = {
  TTL_MS,
  DB_NAME,
  STORE_NAME,
  SINGLETON_KEY,
}
