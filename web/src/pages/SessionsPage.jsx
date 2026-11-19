import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, fmtTime } from '../api.js'
import { Empty, ErrorBox, Loading, Person, StatusPill } from '../components/ui.jsx'

export default function SessionsPage() {
  const [sessions, setSessions] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let live = true
    const load = () => api.sessions().then((s) => live && setSessions(s)).catch((e) => live && setError(e))
    load()
    const t = setInterval(load, 5000)   // keeps processing uploads up to date
    return () => { live = false; clearInterval(t) }
  }, [])

  return (
    <div className="grid gap-5">
      <section className="card flex flex-wrap items-end justify-between gap-4 border-transparent bg-soft">
        <div>
          <div className="text-[11px] font-bold uppercase tracking-[0.08em] text-accent">Group dynamics from gaze and posture</div>
          <h1 className="mt-1 text-2xl font-semibold">Who leads, who drifts?</h1>
          <p className="mt-1 text-sm text-muted">Open an AMI demo meeting, or upload a meeting of 3–5 people.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link to="/upload" className="btn">Upload meeting</Link>
          {sessions?.some((s) => s.source === 'ami') &&
            <Link to={`/sessions/${sessions.find((s) => s.source === 'ami').id}`} className="btn-ghost">Try AMI demo</Link>}
        </div>
      </section>

      <div className="flex items-baseline justify-between">
        <h2 className="label">Recent sessions</h2>
        <span className="text-xs text-muted">Uploaded videos are deleted after 24 h; the scores stay</span>
      </div>

      {error && <ErrorBox error={error} />}
      {!sessions && !error && <Loading what="Loading sessions" />}
      {sessions?.length === 0 && (
        <Empty title="No sessions yet" action={<Link to="/upload" className="btn">Upload a meeting</Link>}>
          Upload a short video of a 3–5 person meeting, or load the AMI demo sessions on the server
          (<code className="font-mono">python -m grouppulse.api.import_ami</code>).
        </Empty>
      )}
      <ul className="grid gap-3">
        {sessions?.map((s) => (
          <li key={s.id}>
            <Link to={`/sessions/${s.id}`} className="card grid grid-cols-[1fr_auto] items-center gap-x-4 gap-y-1 hover:border-accent">
              <div className="min-w-0">
                <div className="truncate font-semibold">{s.title}</div>
                <div className="text-xs text-muted">
                  {s.source === 'ami' ? 'AMI demo' : 'Your upload'} · {s.people ?? '?'} people · {fmtTime(s.duration_s)}
                </div>
              </div>
              <StatusPill status={s.status} />
              <div className="text-xs text-muted">
                {s.status === 'done' && s.dominant ? <>Most looked at: <Person label={s.dominant} /></> : s.error || ' '}
              </div>
              <div className="text-right text-xs text-muted">
                {s.rapport_index != null && <>Rapport <b className="tabular-nums text-ink">{s.rapport_index.toFixed(2)}</b></>}
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
