// Hand-drawn SVG charts. All colours come from the active scheme's CSS variables.
import { personColor } from '../theme.js'
import { fmtTime } from '../api.js'

const V = (name, a = 1) => `rgb(var(--${name}) / ${a})`

/** Rolling mean over `n` neighbouring windows (centred), per person. */
export function smoothWindows(windows, field, people, n) {
  if (n <= 1) return windows
  const h = Math.floor(n / 2)
  return windows.map((w, i) => {
    const part = windows.slice(Math.max(0, i - h), Math.min(windows.length, i + h + 1))
    const vals = Object.fromEntries(people.map((p) => [p, part.reduce((s, x) => s + (x[field]?.[p] ?? 0), 0) / part.length]))
    return { ...w, [field]: vals }
  })
}

/** Lines over time, one per person (values 0..1), with an optional highlighted window. */
export function TimelineChart({ windows: raw, field, people, height = 180, highlight, threshold, ariaLabel, smooth = 1 }) {
  const windows = smoothWindows(raw, field, people, smooth)
  const W = 640, H = height, x0 = 34, y0 = 10, w = W - x0 - 10, h = H - y0 - 24
  if (!windows?.length) return null
  const end = windows[windows.length - 1].end
  const X = (t) => x0 + (t / end) * w
  const Y = (v) => y0 + h - v * h
  const ticks = Array.from({ length: 5 }, (_, i) => (end / 4) * i)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img" aria-label={ariaLabel}>
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={x0} x2={x0 + w} y1={Y(v)} y2={Y(v)} stroke={V('line')} />
          <text x={x0 - 6} y={Y(v) + 3} textAnchor="end" fontSize="10" fill={V('muted')}>{v}</text>
        </g>
      ))}
      {highlight != null && windows[highlight] && (
        <rect x={X(windows[highlight].start)} y={y0} height={h}
          width={Math.max(2, X(windows[highlight].end) - X(windows[highlight].start))} fill={V('accent', 0.12)} />
      )}
      {threshold != null && (
        <g>
          <line x1={x0} x2={x0 + w} y1={Y(threshold)} y2={Y(threshold)} stroke={V('warn')} strokeDasharray="4 3" />
          <text x={x0 + w} y={Y(threshold) - 4} textAnchor="end" fontSize="10" fill={V('warn')}>disengaged below {threshold}</text>
        </g>
      )}
      {people.map((p) => {
        const pts = windows.map((win) => `${X((win.start + win.end) / 2).toFixed(1)},${Y(win[field]?.[p] ?? 0).toFixed(1)}`).join(' ')
        return <polyline key={p} points={pts} fill="none" stroke={personColor(p)} strokeWidth="1.8" strokeLinejoin="round" />
      })}
      {ticks.map((t, i) => (
        <text key={i} x={X(t)} y={H - 6} fontSize="10" fill={V('muted')} textAnchor={i === 0 ? 'start' : i === 4 ? 'end' : 'middle'}>{fmtTime(t)}</text>
      ))}
    </svg>
  )
}

/** One person's engagement over time with disengaged spells shaded. */
export function EngagementChart({ series, spells = [], label, threshold = 0.4 }) {
  const W = 640, H = 170, x0 = 34, y0 = 10, w = W - x0 - 10, h = H - y0 - 24
  if (!series?.length) return null
  const end = series[series.length - 1].end
  const X = (t) => x0 + (t / end) * w
  const Y = (v) => y0 + h - v * h
  const pts = series.map((s) => [X((s.start + s.end) / 2), Y(s.value ?? 0)])
  const line = pts.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img" aria-label={`Engagement over time for ${label}`}>
      {spells.map((s, i) => (
        <rect key={i} x={X(s.start)} y={y0} width={Math.max(2, X(s.end) - X(s.start))} height={h} fill={V('warn', 0.12)} />
      ))}
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={x0} x2={x0 + w} y1={Y(v)} y2={Y(v)} stroke={V('line')} />
          <text x={x0 - 6} y={Y(v) + 3} textAnchor="end" fontSize="10" fill={V('muted')}>{v}</text>
        </g>
      ))}
      <line x1={x0} x2={x0 + w} y1={Y(threshold)} y2={Y(threshold)} stroke={V('warn')} strokeDasharray="4 3" />
      <text x={x0 + w} y={Y(threshold) - 4} textAnchor="end" fontSize="10" fill={V('warn')}>disengaged below {threshold}</text>
      <polygon points={`${x0},${Y(0)} ${line} ${pts[pts.length - 1][0]},${Y(0)}`} fill={personColor(label)} opacity="0.14" />
      <polyline points={line} fill="none" stroke={personColor(label)} strokeWidth="2" />
      {[0, 0.25, 0.5, 0.75, 1].map((f, i) => (
        <text key={i} x={x0 + f * w} y={H - 6} fontSize="10" fill={V('muted')} textAnchor={i === 0 ? 'start' : i === 4 ? 'end' : 'middle'}>{fmtTime(end * f)}</text>
      ))}
    </svg>
  )
}

/** Who looks at whom: arrows between seats, width = share of that person's gaze. */
export function GazeNetwork({ perPerson }) {
  const people = Object.keys(perPerson)
  const n = people.length
  const R = 70, cx = 110, cy = 95
  const pos = Object.fromEntries(people.map((p, i) => {
    const a = (Math.PI * 2 * i) / n - Math.PI * 0.75
    return [p, [cx + R * Math.cos(a), cy + R * Math.sin(a)]]
  }))
  const att = Object.fromEntries(people.map((p) => [p, perPerson[p].attention_received ?? 0]))
  const maxAtt = Math.max(0.0001, ...Object.values(att))
  const edges = []
  for (const a of people) {
    for (const [b, share] of Object.entries(perPerson[a].gaze_shares || {})) {
      if (pos[b] && b !== a && share >= 0.03) edges.push([a, b, share])
    }
  }
  return (
    <svg viewBox="0 0 220 190" className="mx-auto block w-full max-w-[280px]" role="img" aria-label="Who looks at whom">
      {edges.map(([a, b, s]) => {
        const [x1, y1] = pos[a], [x2, y2] = pos[b]
        const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), o = 3
        const nx = (-dy / L) * o, ny = (dx / L) * o   // offset so a->b and b->a don't overlap
        return <line key={a + b} x1={x1 + nx} y1={y1 + ny} x2={x2 + nx} y2={y2 + ny}
          stroke={personColor(a)} strokeOpacity="0.75" strokeWidth={1 + s * 10} strokeLinecap="round" />
      })}
      {people.map((p) => (
        <g key={p}>
          <circle cx={pos[p][0]} cy={pos[p][1]} r={10 + (att[p] / maxAtt) * 8} fill={V('surface')} stroke={personColor(p)} strokeWidth="2.5" />
          <text x={pos[p][0]} y={pos[p][1] + 3.5} textAnchor="middle" fontSize="10" fontWeight="700" fill={V('ink')}>{p}</text>
        </g>
      ))}
    </svg>
  )
}

/** Horizontal share bars (where one person looked). */
export function ShareBars({ shares }) {
  const rows = Object.entries(shares || {}).sort((a, b) => b[1] - a[1])
  const max = Math.max(0.0001, ...rows.map(([, v]) => v))
  const colour = (k) => (/^P\d+$/.test(k) ? personColor(k) : k === 'away' ? V('warn') : V('muted'))
  return (
    <div className="grid gap-2">
      {rows.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[84px_1fr_40px] items-center gap-2 text-xs">
          <span className="capitalize">{k}</span>
          <div className="h-2 overflow-hidden rounded bg-line">
            <div className="h-full rounded" style={{ width: `${(v / max) * 100}%`, background: colour(k) }} />
          </div>
          <b className="text-right tabular-nums">{Math.round(v * 100)}%</b>
        </div>
      ))}
    </div>
  )
}
