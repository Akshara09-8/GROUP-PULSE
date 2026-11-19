import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api.js'
import { ErrorBox, StatusPill } from '../components/ui.jsx'

const STEPS = [['uploaded', 'Video uploaded'], ['queued', 'Waiting for the free server'],
  ['processing', 'Reading gaze and body orientation'], ['done', 'Results ready']]

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [people, setPeople] = useState(4)
  const [consent, setConsent] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [job, setJob] = useState(null)     // {id, status, error}
  const input = useRef(null)

  useEffect(() => {
    if (!job || ['done', 'failed'].includes(job.status)) return
    const t = setInterval(() => api.session(job.id).then((s) => setJob(s)).catch(setError), 2000)
    return () => clearInterval(t)
  }, [job])

  async function submit(e) {
    e.preventDefault()
    setError(null)
    if (!file) return setError(new Error('Choose a video file first.'))
    if (!consent) return setError(new Error('Please confirm that everyone in the video agreed to be analysed.'))
    setBusy(true)
    try {
      setJob(await api.upload({ file, title: title || file.name, people, consent }))
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  const reached = job ? (job.status === 'failed' ? 1 : STEPS.findIndex(([k]) => k === job.status)) : -1

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <form onSubmit={submit} className="card grid content-start gap-4" aria-label="New session">
        <h1 className="text-xl font-semibold">New session</h1>
        <button type="button" onClick={() => input.current?.click()}
          className="grid justify-items-center gap-1 rounded-lg border-2 border-dashed border-accent bg-soft px-4 py-6 text-center">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="rgb(var(--accent))" strokeWidth="1.8" aria-hidden="true">
            <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" />
          </svg>
          <b>{file ? file.name : 'Choose a video'}</b>
          <span className="text-xs text-muted">MP4, MOV, AVI or WebM · up to 10 minutes · 3–5 people · one fixed camera</span>
        </button>
        <input ref={input} id="video" type="file" accept="video/*" className="sr-only" aria-label="Video file"
          onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <label className="grid gap-1 text-sm">
          <span className="label">Session name</span>
          <input id="title" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Project kickoff"
            className="rounded-md border border-line bg-bg px-3 py-2" />
        </label>
        <label className="grid gap-1 text-sm">
          <span className="label">People in view</span>
          <select id="people" value={people} onChange={(e) => setPeople(Number(e.target.value))}
            className="rounded-md border border-line bg-bg px-3 py-2">
            {[2, 3, 4, 5, 6].map((n) => <option key={n} value={n}>{n}{n < 3 || n > 5 ? ' (outside the 3–5 the scores are designed for)' : ''}</option>)}
          </select>
        </label>
        <label className="flex items-start gap-2 text-sm">
          <input id="consent" type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} className="mt-1 accent-[rgb(var(--accent))]" />
          <span>Everyone in this video agreed to be analysed. No faces or names are stored, and the video is deleted after 24 hours.</span>
        </label>
        {error && <ErrorBox error={error} />}
        <button className="btn justify-self-start" disabled={busy || (job && !['done', 'failed'].includes(job.status))}>
          {busy ? 'Uploading…' : 'Start analysis'}
        </button>
      </form>

      <section className="card grid content-start gap-3" aria-live="polite">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Processing</h2>
          {job && <StatusPill status={job.status} />}
        </div>
        {!job && <p className="text-sm text-muted">Progress appears here after you start. You can leave this page; the session shows on Sessions when it is done.</p>}
        {job && (
          <ol className="grid gap-2">
            {STEPS.map(([k, text], i) => {
              const state = job.status === 'failed' && i === reached + 1 ? 'fail' : i <= reached ? 'done' : i === reached + 1 ? 'now' : 'todo'
              return (
                <li key={k} className="flex items-center gap-2 text-sm">
                  <span className={`grid h-5 w-5 place-items-center rounded-full text-[10px] font-bold ${
                    state === 'done' ? 'bg-good text-surface' : state === 'now' ? 'bg-accent text-accent-ink' : state === 'fail' ? 'bg-warn text-surface' : 'ring-1 ring-inset ring-line'}`}>
                    {state === 'done' ? '✓' : state === 'fail' ? '!' : i + 1}
                  </span>
                  <span className={state === 'todo' ? 'text-muted' : state === 'now' ? 'font-semibold' : ''}>{text}</span>
                </li>
              )
            })}
          </ol>
        )}
        {job?.status === 'failed' && <ErrorBox error={new Error(job.error || 'Processing failed.')} />}
        {job?.status === 'done' && <Link to={`/sessions/${job.id}`} className="btn justify-self-start">See results</Link>}
      </section>
    </div>
  )
}
