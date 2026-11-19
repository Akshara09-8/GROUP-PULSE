# Web app (Phase 4)

React 18 + Vite 6 + Tailwind 3. Four pages: **Sessions**, **Upload**, **Session analysis**, **Participant detail**, in the four colour schemes from `../design/group-pulse-ui-schemes.html` (picker in the header; `?scheme=calm|night|coral|indigo` in a link picks one).

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api to the backend on :8000 (python -m grouppulse.api)
npm test         # vitest: pages against a mocked API, WCAG contrast of every scheme
npm run build    # dist/ (about 63 KB gzipped JS); set VITE_API_URL to the deployed backend
```

Needs Node 18 or newer. Without a system-wide install, a portable Node 22 LTS (the nodejs.org zip) works: add its folder to PATH for the session.

| Page | Shows |
| --- | --- |
| Sessions | AMI demo and uploaded sessions, status, most-looked-at person, rapport; empty state with next step |
| Upload | Video, session name, people in view, consent; step list that follows the job until done or failed |
| Session analysis | Rapport, most looked at, disengaged spells, robot addressee at the chosen moment, attention or engagement over time (1-minute average or raw 10 s windows), who-looks-at-whom graph, people |
| Participant | Attention received, engagement against the group, looking away, engagement over time with spells shaded, where they looked |

Phase 4 tests: T4.1 flow checked against the real API in the browser (Chrome); T4.3 per-person scores match the API; T4.4 every scheme passes WCAG AA text contrast (this caught Sunrise Coral's accent at 4.18:1, now `#B83147`, 4.82:1); T4.5 no horizontal scroll at 375 px on all four pages; T4.6 scheme remembered across reloads; T4.7 empty and failed states. **T4.2 (video with gaze overlay) is not built:** the AMI demo videos are not hosted, and uploads are not processed until the single-camera reader exists.
