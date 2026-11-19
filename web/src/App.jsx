import { useState } from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import { SCHEMES, applyScheme, savedScheme } from './theme.js'
import SessionsPage from './pages/SessionsPage.jsx'
import UploadPage from './pages/UploadPage.jsx'
import SessionPage from './pages/SessionPage.jsx'
import PersonPage from './pages/PersonPage.jsx'

function Logo() {
  return (
    <svg width="20" height="20" viewBox="0 0 18 18" aria-hidden="true">
      <circle cx="5" cy="6" r="2.4" fill="rgb(var(--p1))" /><circle cx="13" cy="6" r="2.4" fill="rgb(var(--p2))" />
      <circle cx="5" cy="13" r="2.4" fill="rgb(var(--p3))" /><circle cx="13" cy="13" r="2.4" fill="rgb(var(--p4))" />
      <path d="M7 7.5l4 4" stroke="currentColor" strokeWidth="1.3" />
    </svg>
  )
}

export function SchemePicker() {
  const [scheme, setScheme] = useState(savedScheme())
  return (
    <label className="flex items-center gap-2 text-xs text-muted">
      <span className="hidden sm:inline">Colours</span>
      <select aria-label="Colour scheme" value={scheme}
        onChange={(e) => setScheme(applyScheme(e.target.value))}
        className="rounded-md border border-line bg-surface px-2 py-1 text-ink">
        {Object.entries(SCHEMES).map(([id, s]) => <option key={id} value={id}>{s.name}</option>)}
      </select>
    </label>
  )
}

const link = ({ isActive }) =>
  `text-sm py-1 ${isActive ? 'text-ink font-semibold shadow-[0_2px_0_rgb(var(--accent))]' : 'text-muted hover:text-ink'}`

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-surface">
        <nav className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3">
          <NavLink to="/" className="flex items-center gap-2 font-display text-base font-bold"><Logo />Group Pulse</NavLink>
          <NavLink to="/" end className={link}>Sessions</NavLink>
          <NavLink to="/upload" className={link}>Upload</NavLink>
          <span className="flex-1" />
          <SchemePicker />
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Routes>
          <Route path="/" element={<SessionsPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/sessions/:id" element={<SessionPage />} />
          <Route path="/sessions/:id/people/:label" element={<PersonPage />} />
          <Route path="*" element={<p className="text-muted">That page doesn't exist. <NavLink className="text-accent underline" to="/">Back to sessions</NavLink></p>} />
        </Routes>
      </main>
      <footer className="mx-auto max-w-6xl px-4 pb-8 text-xs text-muted">
        Scores are estimates from gaze and body orientation, not judgements of people. Demo data: AMI Meeting Corpus (CC BY 4.0).
      </footer>
    </div>
  )
}
