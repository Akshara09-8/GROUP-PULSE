import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, fmtTime, pct } from '../api.js'
import { GazeNetwork, TimelineChart } from '../components/charts.jsx'
import VideoOverlay from '../components/VideoOverlay.jsx'
import { Empty, ErrorBox, Loading, Person, Pill, StatusPill } from '../components/ui.jsx'
import { personColor } from '../theme.js'

function rapportWord(r) {
  return r >= 0.5 ? ['good', 'High'] : r >= 0.3 ? ['acc', 'Medium'] : ['warn', 'Low']
}

export default function SessionPage() {
  const { id } = useParams()
  const nav = useNavigate()
  const [s, setS] = useState(null)
  const [wins, setWins] = useState(null)
  const [error, setError] = useState(null)
  const [at, setAt] = useState(0)
  const [field, setField] = useState('attention')
  const [smooth, setSmooth] = useState(true)

  useEffect(() => {
    let live = true
    api.session(id).then((x) => {
      if (!live) return
      setS(x)
      if (x.status === 'done') api.windows(id).then((w) => live && setWins(w)).catch((e) => live && setError(e))
    }).catch((e) => live && setError(e))
    return () => { live = false }
  }, [id])

  const people = useMemo(() => Object.keys(s?.per_person || {}), [s])

  if (error) return <ErrorBox error={error} />
  if (!s) return <Loading what="Loading session" />
  if (s.status !== 'done') {
    return (
      <Empty title={s.title} action={<Link to="/" className="btn-ghost">Back to sessions</Link>}>
        This session is <StatusPill status={s.status} />{s.error ? ` — ${s.error}` : '. Results appear here when processing finishes.'}
      </Empty>
    )
  }

  const g = s.group
  const [rTone, rWord] = rapportWord(g.rapport_index)
  const w = wins?.[at]
  const spells = s.disengaged_spells || []

  async function remove() {
    if (!window.confirm || !confirm('Delete this session and its results?')) return
    await api.remove(id).catch(setError)
    nav('/')
  }

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="text-xs text-muted"><Link to="/" className="hover:underline">Sessions</Link> / {id}</div>
          <h1 className="text-2xl font-semibold">{s.title}</h1>
        </div>
        <div className="flex items-center gap-2">
          <Pill tone="good">Done · {fmtTime(s.duration_s)}</Pill>
          {s.source === 'upload' && <button onClick={remove} className="text-xs text-muted underline">Delete</button>}
        </div>
      </div>
      {s.warnings?.map((x) => <ErrorBox key={x} error={new Error(x)} />)}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div className="card">
          <div className="label">Group rapport</div>
          <div className="mt-1 flex items-baseline gap-2"><span className="num text-2xl">{g.rapport_index.toFixed(2)}</span><Pill tone={rTone}>{rWord}</Pill></div>
          <div className="mt-2 h-1.5 overflow-hidden rounded bg-line"><div className="h-full bg-good" style={{ width: pct(g.rapport_index) }} /></div>
          <div className="mt-2 text-xs text-muted">Mutual gaze {pct(g.mutual_gaze_rate)} · gaze following {pct(g.gaze_following_rate)}</div>
        </div>
        <div className="card">
          <div className="label">Most looked at</div>
          <div className="mt-1"><Person label={g.dominant} /> <span className="text-sm">· {pct(s.per_person[g.dominant].attention_received)} of others' gaze</span></div>
          <div className="mt-2 text-xs text-muted">Order: {g.dominance_rank.join(' › ')}</div>
        </div>
        <div className="card">
          <div className="label">Disengaged spells</div>
          {spells.length === 0 ? <div className="mt-1 text-sm">None of 30 s or longer</div> : (
            <ul className="mt-1 grid gap-0.5 text-sm">
              {spells.slice(0, 3).map((x, i) => <li key={i}><Person label={x.person} /> {fmtTime(x.start)}–{fmtTime(x.end)}</li>)}
              {spells.length > 3 && <li className="text-xs text-muted">+{spells.length - 3} more</li>}
            </ul>
          )}
        </div>
        <div className="card flex items-center gap-3">
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-accent font-mono text-xs font-bold text-accent-ink">BOT</span>
          <div>
            <div className="label">Robot should address{w ? ` at ${fmtTime(w.start)}` : ''}</div>
            {w?.addressee?.person ? <div className="text-sm"><Person label={w.addressee.person} /> · {w.addressee.reason}</div> : <div className="text-sm text-muted">–</div>}
          </div>
        </div>
      </div>

      {s.has_video && (
        <section className="card">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="label">Video with gaze</h2>
            <span className="text-xs text-muted">Deleted {s.expires_at ? new Date(s.expires_at).toLocaleString() : 'after 24 h'}</span>
          </div>
          <VideoOverlay sessionId={id} onTime={(t) => wins && setAt(Math.min(wins.length - 1, Math.floor(t / 10)))} />
        </section>
      )}

      <div className="grid gap-3 lg:grid-cols-[1.6fr_1fr]">
        <section className="card">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h2 className="label">Over time</h2>
            <div className="flex flex-wrap items-center gap-1 text-xs" role="tablist" aria-label="Measure">
              <label className="mr-2 flex items-center gap-1 text-muted">
                <input type="checkbox" checked={smooth} onChange={(e) => setSmooth(e.target.checked)} className="accent-[rgb(var(--accent))]" />
                1-min average
              </label>
              {[['attention', 'Attention received'], ['engagement', 'Engagement']].map(([k, t]) => (
                <button key={k} role="tab" aria-selected={field === k} onClick={() => setField(k)}
                  className={`rounded px-2 py-1 ${field === k ? 'bg-soft font-semibold text-accent' : 'text-muted'}`}>{t}</button>
              ))}
            </div>
          </div>
          {!wins ? <Loading what="Loading timeline" /> : (
            <>
              <TimelineChart windows={wins} field={field} people={people} highlight={at} smooth={smooth ? 6 : 1}
                threshold={field === 'engagement' ? 0.4 : undefined} ariaLabel={`${field} per person over time`} />
              <label className="mt-2 flex items-center gap-3 text-xs text-muted">
                <span className="shrink-0 tabular-nums">{fmtTime(w?.start)}</span>
                <input type="range" min="0" max={wins.length - 1} value={at} onChange={(e) => setAt(Number(e.target.value))}
                  aria-label="Choose a moment" className="w-full accent-[rgb(var(--accent))]" />
              </label>
              <div className="mt-2 flex flex-wrap gap-3 text-xs text-muted">
                {people.map((p) => (
                  <Link key={p} to={`/sessions/${id}/people/${p}`} className="inline-flex items-center gap-1 hover:text-ink">
                    <span className="h-2 w-2 rounded-sm" style={{ background: personColor(p) }} />{p}: {pct(w?.[field]?.[p])}
                  </Link>
                ))}
              </div>
            </>
          )}
        </section>
        <section className="card">
          <h2 className="label mb-1">Who looks at whom</h2>
          <GazeNetwork perPerson={s.per_person} />
          <p className="text-center text-[11px] text-muted">Line width = share of that person's gaze · circle size = attention received</p>
        </section>
      </div>

      <section className="card">
        <h2 className="label mb-2">People</h2>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {people.map((p) => {
            const d = s.per_person[p]
            return (
              <Link key={p} to={`/sessions/${id}/people/${p}`} className="rounded-md border border-line p-3 hover:border-accent">
                <Person label={p} />
                <div className="mt-1 text-xs text-muted">Attention received {pct(d.attention_received)} · engagement {d.engagement.toFixed(2)}</div>
                {d.disengaged_s > 0 && <div className="mt-1 text-xs font-semibold text-warn">Disengaged {fmtTime(d.disengaged_s)}</div>}
              </Link>
            )
          })}
        </div>
      </section>
    </div>
  )
}
