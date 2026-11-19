import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, fmtTime, pct } from '../api.js'
import { EngagementChart, ShareBars } from '../components/charts.jsx'
import { ErrorBox, Loading, Pill, Stat } from '../components/ui.jsx'
import { personColor } from '../theme.js'

export default function PersonPage() {
  const { id, label } = useParams()
  const [d, setD] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let live = true
    api.person(id, label).then((x) => live && setD(x)).catch((e) => live && setError(e))
    return () => { live = false }
  }, [id, label])

  if (error) return <ErrorBox error={error} />
  if (!d) return <Loading what={`Loading ${label}`} />

  const below = d.engagement < d.group_mean_engagement
  const away = d.gaze_shares?.away ?? 0

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <span className="grid h-10 w-10 place-items-center rounded-full font-bold text-white" style={{ background: personColor(label) }}>{label}</span>
          <div>
            <div className="text-xs text-muted"><Link to="/" className="hover:underline">Sessions</Link> / <Link to={`/sessions/${id}`} className="hover:underline">{id}</Link> / participants</div>
            <h1 className="text-2xl font-semibold">Participant {label}</h1>
          </div>
        </div>
        {d.disengaged_s > 0 ? <Pill tone="warn">Disengaged {fmtTime(d.disengaged_s)}</Pill> : <Pill tone="good">No long disengaged spells</Pill>}
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Attention received" value={pct(d.attention_received)} note="share of others' gaze" />
        <Stat label="Engagement (avg)" value={d.engagement.toFixed(2)}
          note={`${below ? 'Below' : 'At or above'} group average ${d.group_mean_engagement.toFixed(2)}`} tone={below ? 'warn' : undefined} />
        <Stat label="Looking away" value={pct(away)} note="of frames with a known target" />
        <Stat label="Dominance" value={d.dominance.toFixed(2)} note="gaze-based" />
      </div>

      <div className="grid gap-3 lg:grid-cols-[1.6fr_1fr]">
        <section className="card">
          <h2 className="label mb-2">Engagement over time</h2>
          <EngagementChart series={d.engagement_over_time} spells={d.disengaged_spells} label={label} />
        </section>
        <section className="card">
          <h2 className="label mb-3">Where {label} looked</h2>
          <ShareBars shares={d.gaze_shares} />
        </section>
      </div>

      {d.disengaged_spells.map((x, i) => (
        <div key={i} className="rounded-r-md border-l-4 border-warn bg-warn/10 px-3 py-2 text-sm">
          <b>{fmtTime(x.start)}–{fmtTime(x.end)}</b>{' '}
          <span className="text-muted">Engagement stayed under 0.4 for {fmtTime(x.end - x.start)}. Estimate only; check the recording before drawing conclusions.</span>
        </div>
      ))}
    </div>
  )
}
