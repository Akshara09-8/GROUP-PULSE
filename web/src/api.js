// Thin client for the Phase 3 API. Dev uses the Vite proxy at /api; production sets VITE_API_URL.
const BASE = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status }
}

async function req(path, opts = {}) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, opts)
  } catch {
    throw new ApiError(0, "Can't reach the Group Pulse server. If it was idle, it may be waking up; try again in a minute.")
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
