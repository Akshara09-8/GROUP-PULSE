import { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { personColor } from '../theme.js'

// Find the overlay frame nearest to time t (frames are evenly spaced, sorted by t).
export function frameAt(overlay, t) {
  const fr = overlay?.frames
  if (!fr?.length) return null
  const step = 1 / (overlay.fps || 1)
  const i = Math.max(0, Math.min(fr.length - 1, Math.round(t / step)))
  return fr[i]
}

const centre = (b, w, h) => [((b[0] + b[2]) / 2) * w, ((b[1] + b[3]) / 2) * h]

function draw(ctx, frame, w, h) {
  ctx.clearRect(0, 0, w, h)
  if (!frame) return
  const css = getComputedStyle(document.documentElement)
  const people = frame.people || {}
  for (const [p, d] of Object.entries(people)) {
    const col = `rgb(${css.getPropertyValue(`--p${p.slice(1)}`).trim() || '255 255 255'})`
    const [x1, y1, x2, y2] = d.box
    ctx.lineWidth = 2
    ctx.strokeStyle = col
    ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h)
    ctx.fillStyle = col
    ctx.font = '600 12px Inter, sans-serif'
    ctx.fillText(p, x1 * w, Math.max(12, y1 * h - 4))
    const target = people[d.target]
    if (target) {
      const [ax, ay] = centre(d.box, w, h)
      const [bx, by] = centre(target.box, w, h)
      const ang = Math.atan2(by - ay, bx - ax)
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(bx, by)
      ctx.lineTo(bx - 9 * Math.cos(ang - 0.4), by - 9 * Math.sin(ang - 0.4))
      ctx.lineTo(bx - 9 * Math.cos(ang + 0.4), by - 9 * Math.sin(ang + 0.4))
      ctx.closePath(); ctx.fill()
    } else if (d.target === 'table' || d.target === 'away') {
      ctx.font = '11px Inter, sans-serif'
      ctx.fillText(d.target, x1 * w, y2 * h + 13)
    }
  }
}

/** Uploaded video with gaze overlay (T4.2). Calls onTime(seconds) as it plays. */
export default function VideoOverlay({ sessionId, onTime }) {
  const video = useRef(null)
  const canvas = useRef(null)
  const [overlay, setOverlay] = useState(null)
  const [error, setError] = useState(null)
  const [show, setShow] = useState(true)

  useEffect(() => {
    api.overlay(sessionId).then(setOverlay).catch(setError)
  }, [sessionId])

  useEffect(() => {
    let raf
    const tick = () => {
      const v = video.current, c = canvas.current
      if (v && c && overlay) {
        const w = v.clientWidth, h = v.clientHeight
        if (c.width !== w || c.height !== h) { c.width = w; c.height = h }
        const ctx = c.getContext('2d')   // null where canvas is unavailable (tests, locked-down browsers)
        if (ctx) draw(ctx, show ? frameAt(overlay, v.currentTime) : null, w, h)
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [overlay, show])

  return (
    <div className="mx-auto grid w-full max-w-2xl gap-2">
      {/* width-capped (not letterboxed) so the canvas lines up with the picture exactly */}
      <div className="relative overflow-hidden rounded-md bg-black">
        <video ref={video} src={api.videoUrl(sessionId)} controls playsInline preload="metadata"
          className="block w-full" onTimeUpdate={(e) => onTime?.(e.currentTarget.currentTime)}
          aria-label="Meeting video" />
        <canvas ref={canvas} className="pointer-events-none absolute left-0 top-0" aria-hidden="true" />
      </div>
      <label className="flex items-center gap-2 text-xs text-muted">
        <input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} className="accent-[rgb(var(--accent))]" />
        Show gaze overlay {overlay ? '' : error ? '(not available)' : '(loading)'}
      </label>
    </div>
  )
}
