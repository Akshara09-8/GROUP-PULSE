import { personColor } from '../theme.js'

export const Dot = ({ label, size = 10 }) => (
  <span aria-hidden="true" className="inline-block rounded-full align-middle"
    style={{ width: size, height: size, background: personColor(label) }} />
)

export function Person({ label }) {
  return <span className="inline-flex items-center gap-1.5 font-semibold"><Dot label={label} />{label}</span>
}

export const STATUS = {
  queued: ['acc', 'Queued'], processing: ['acc', 'Processing'], done: ['good', 'Done'], failed: ['warn', 'Failed'],
}

export function Pill({ tone = 'acc', children }) {
  const cls = { good: 'bg-good/15 text-good', warn: 'bg-warn/15 text-warn', acc: 'bg-soft text-accent' }[tone]
  return <span className={`pill ${cls}`}>{children}</span>
}

export function StatusPill({ status }) {
  const [tone, text] = STATUS[status] || ['acc', status]
  return <Pill tone={tone}>{text}</Pill>
}

export function Stat({ label, value, note, tone }) {
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className="num mt-1 text-2xl">{value}</div>
      {note && <div className={`mt-1 text-xs ${tone === 'warn' ? 'text-warn font-semibold' : 'text-muted'}`}>{note}</div>}
    </div>
  )
}

export function Empty({ title, children, action }) {
  return (
    <div className="card flex flex-col items-start gap-2 py-8">
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className="max-w-prose text-sm text-muted">{children}</p>
      {action}
    </div>
  )
}

export function ErrorBox({ error }) {
  return (
    <div role="alert" className="rounded-md border-l-4 border-warn bg-warn/10 px-3 py-2 text-sm">
      {error?.message || String(error)}
    </div>
  )
}

export const Loading = ({ what = 'Loading' }) => <p className="text-sm text-muted" aria-live="polite">{what}…</p>

// Small bar row: one bar per person, height by value (0..1)
export function MiniBars({ values }) {
  const max = Math.max(0.0001, ...Object.values(values))
  return (
    <div className="flex h-7 items-end gap-[3px]" title="Dominance by person">
      {Object.entries(values).map(([p, v]) => (
        <span key={p} title={`${p} ${Math.round(v * 100)}%`} className="w-2 rounded-t-sm"
          style={{ height: `${4 + (v / max) * 24}px`, background: personColor(p) }} />
      ))}
    </div>
  )
}
