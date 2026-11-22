// Thin client for the Phase 3 API. Dev uses the Vite proxy at /api; production sets VITE_API_URL.
// VITE_DEMO=1 builds a server-free demo: responses come from static JSON exported by
// scripts/export_demo.py into public/demo (same shapes as the API), and uploads are disabled.
export const DEMO = import.meta.env.VITE_DEMO === '1'
const BASE = DEMO
  ? `${import.meta.env.BASE_URL}demo`.replace(/\/$/, '')
  : (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '')
const demoPath = (path) => `${path}.json`

export const DEMO_UPLOAD_MESSAGE =
  "This is the static demo, so it can't process uploads. Explore the AMI demo meetings, or run the full app with its backend to analyse your own video."

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status }
}

async function req(path, opts = {}) {
  let res
  if (DEMO && (opts.method || 'GET') !== 'GET') throw new ApiError(0, DEMO_UPLOAD_MESSAGE)
  try {
    res = await fetch(`${BASE}${DEMO ? demoPath(path) : path}`, opts)
  } catch {
    throw new ApiError(0, "The Group Pulse server is waking up (the free server sleeps when idle; this can take up to a minute). Retrying automatically…")
  }
  if (res.status === 204) return null
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new ApiError(res.status, body.detail || `Request failed (${res.status})`)
  return body
}

export const api = {
  sessions: () => req('/sessions'),
  session: (id) => req(`/sessions/${id}`),
  windows: (id) => req(`/sessions/${id}/windows`),
  person: (id, label) => req(`/sessions/${id}/people/${label}`),
  remove: (id) => req(`/sessions/${id}`, { method: 'DELETE' }),
  overlay: (id) => req(`/sessions/${id}/overlay`),
  videoUrl: (id) => `${BASE}/sessions/${id}/video`,
  upload: ({ file, title, people, consent }) => {
    const f = new FormData()
    f.append('file', file)
    f.append('title', title)
    f.append('people', String(people))
    f.append('consent', consent ? 'true' : 'false')
    return req('/sessions', { method: 'POST', body: f })
  },
}

export const fmtTime = (s) => {
  if (s == null) return '–'
  const m = Math.floor(s / 60)
  return `${m}:${String(Math.round(s % 60)).padStart(2, '0')}`
}
export const pct = (x) => (x == null ? '–' : `${Math.round(x * 100)}%`)
