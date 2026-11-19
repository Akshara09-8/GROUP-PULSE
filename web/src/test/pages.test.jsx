import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import App from '../App.jsx'

const people = ['P1', 'P2', 'P3', 'P4']
const windows = [0, 10, 20].map((t, i) => ({
  start: t, end: t + 10, mutual_gaze: 0.1,
  attention: Object.fromEntries(people.map((p, k) => [p, (k + 1) / 10])),
  dominance: Object.fromEntries(people.map((p, k) => [p, (k + 1) / 10])),
  engagement: Object.fromEntries(people.map((p) => [p, p === 'P4' && i > 0 ? 0.2 : 0.8])),
  addressee: i === 2 ? { person: 'P4', reason: 're-engage least engaged' } : { person: 'P2', reason: 'address dominant' },
}))
const session = {
  id: 'es2002a', title: 'ES2002a · Kickoff', source: 'ami', status: 'done', people: 4, duration_s: 30,
  warnings: [], has_video: false,
  group: { dominant: 'P4', dominance_rank: ['P4', 'P3', 'P2', 'P1'], rapport_index: 0.31, mutual_gaze_rate: 0.05,
    gaze_following_rate: 0.38, facing_mean: 0 },
  per_person: Object.fromEntries(people.map((p, k) => [p, {
    attention_received: (k + 1) / 10, facing_received: 0, dominance: (k + 1) / 10, engagement: 0.7, disengaged_s: p === 'P4' ? 40 : 0,
    gaze_shares: { table: 0.5, [people[(k + 1) % 4]]: 0.3, away: 0.2 } }])),
  disengaged_spells: [{ person: 'P4', start: 10, end: 50 }],
}

function mockApi(routes) {
  vi.spyOn(global, 'fetch').mockImplementation(async (url, opts = {}) => {
    const path = String(url).replace(/^\/api/, '')
    const key = `${opts.method || 'GET'} ${path}`
    const hit = routes[key]
    if (!hit) return new Response(JSON.stringify({ detail: `no mock for ${key}` }), { status: 404 })
    const [status, body] = typeof hit === 'function' ? hit(opts) : hit
    return new Response(JSON.stringify(body), { status })
  })
}

const at = (path) => render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>)

describe('Sessions page', () => {
  it('T4.7 shows an empty state with a next step', async () => {
    mockApi({ 'GET /sessions': [200, []] })
    at('/')
    expect(await screen.findByText('No sessions yet')).toBeInTheDocument()
    expect(screen.getAllByRole('link', { name: /upload/i }).length).toBeGreaterThan(0)
  })
  it('lists sessions with status and most-looked-at person', async () => {
    mockApi({ 'GET /sessions': [200, [{ ...session, dominant: 'P4', rapport_index: 0.31 }, {
      id: 'up1', title: 'My upload', source: 'upload', status: 'failed', error: 'Uploads from a single camera are not supported yet.' }]] })
    at('/')
    expect(await screen.findByText('ES2002a · Kickoff')).toBeInTheDocument()
    expect(screen.getByText('Failed')).toBeInTheDocument()
    expect(screen.getByText(/not supported yet/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Try AMI demo' })).toHaveAttribute('href', '/sessions/es2002a')
  })
  it('explains when the server cannot be reached', async () => {
    vi.spyOn(global, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'))
    at('/')
    expect(await screen.findByRole('alert')).toHaveTextContent(/can't reach the group pulse server/i)
  })
})

describe('Session analysis page', () => {
  it('T4.3 shows the scores from the API and moves the robot suggestion with the slider', async () => {
    mockApi({ 'GET /sessions/es2002a': [200, session], 'GET /sessions/es2002a/windows': [200, windows] })
    at('/sessions/es2002a')
    expect(await screen.findByText('0.31')).toBeInTheDocument()
    expect(screen.getByText(/P4 › P3 › P2 › P1/)).toBeInTheDocument()
    await screen.findByLabelText('Choose a moment')
    expect(screen.getByText(/address dominant/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Choose a moment'), { target: { value: '2' } })
    expect(screen.getByText(/re-engage least engaged/)).toBeInTheDocument()
  })
  it('shows a failed upload with its reason', async () => {
    mockApi({ 'GET /sessions/x': [200, { id: 'x', title: 'Mine', status: 'failed', error: 'Not supported yet.' }] })
    at('/sessions/x')
    expect(await screen.findByText(/Not supported yet/)).toBeInTheDocument()
  })
})

describe('Participant page', () => {
  it('shows engagement against the group and the disengaged spell', async () => {
    mockApi({ 'GET /sessions/es2002a/people/P4': [200, { label: 'P4', ...session.per_person.P4, group_mean_engagement: 0.75,
      engagement_over_time: windows.map((w) => ({ start: w.start, end: w.end, value: w.engagement.P4 })),
      disengaged_spells: [{ person: 'P4', start: 10, end: 50 }] }] })
    at('/sessions/es2002a/people/P4')
    expect(await screen.findByText('Participant P4')).toBeInTheDocument()
    expect(screen.getByText(/Below group average 0.75/)).toBeInTheDocument()
    expect(screen.getByText('0:10–0:50')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /engagement over time for P4/i })).toBeInTheDocument()
  })
})

describe('Upload page', () => {
  it('requires consent before sending anything', async () => {
    const spy = vi.spyOn(global, 'fetch')
    at('/upload')
    fireEvent.change(screen.getByLabelText('Video file'), { target: { files: [new File(['x'], 'm.mp4', { type: 'video/mp4' })] } })
    fireEvent.click(screen.getByRole('button', { name: 'Start analysis' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/confirm that everyone/i)
    expect(spy).not.toHaveBeenCalled()
  })
  it('uploads, then follows the job until it fails with the server message', async () => {
    let polls = 0
    mockApi({
      'POST /sessions': [202, { id: 'abc', status: 'queued' }],
      'GET /sessions/abc': () => [200, ++polls < 2 ? { id: 'abc', status: 'processing' }
        : { id: 'abc', status: 'failed', error: 'Uploads from a single camera are not supported yet.' }],
    })
    vi.useFakeTimers({ shouldAdvanceTime: true })
    at('/upload')
    fireEvent.change(screen.getByLabelText('Video file'), { target: { files: [new File(['x'], 'm.mp4', { type: 'video/mp4' })] } })
    fireEvent.click(screen.getByLabelText(/agreed to be analysed/))
    fireEvent.click(screen.getByRole('button', { name: 'Start analysis' }))
    expect(await screen.findByText('Queued')).toBeInTheDocument()
    await vi.advanceTimersByTimeAsync(4500)
    await waitFor(() => expect(screen.getByText(/single camera are not supported yet/)).toBeInTheDocument())
    vi.useRealTimers()
  })
})
