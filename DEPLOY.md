# Launching Group Pulse

Everything is built and configured in this repository; the steps below are the ones that need **your** accounts. All of them are free tiers. Do them in order; each takes a few minutes.

| Piece | Where | What it gives |
| --- | --- | --- |
| Static demo | GitHub Pages | The 4 AMI demo meetings, no server, always fast. Uploads disabled. |
| Backend | Hugging Face Space (Docker, free CPU) | API, upload processing, video overlay. Sleeps when idle. |
| Full web app | Vercel (optional) | The app with uploads, talking to the backend. |
| Uptime + keep-warm | GitHub Actions | Pings `/health` every 5 minutes. |
| Error tracking | Sentry (optional) | Server errors by email. |

## 1. Static demo on GitHub Pages

1. GitHub → this repo → **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. **Settings → Secrets and variables → Actions → Variables → New variable**: `PAGES_ENABLED` = `true`.
3. **Actions → demo site → Run workflow** (it also runs on every push to `main`).
4. The site appears at `https://<your-user>.github.io/GROUP-PULSE/` (the workflow log prints the URL).

## 2. Backend on a Hugging Face Space

1. Sign in at huggingface.co → **New Space** → name e.g. `group-pulse-api`, SDK **Docker**, hardware **CPU basic (free)**, visibility your choice.
2. huggingface.co → **Settings → Access Tokens → New token** with **write** access. Copy it.
3. GitHub → this repo → **Settings → Secrets and variables → Actions**:
   - **Secrets → New secret**: `HF_TOKEN` = the token.
   - **Variables → New variable**: `HF_SPACE` = `<your-hf-user>/group-pulse-api`.
4. **Actions → deploy backend → Run workflow**. The first build takes about 10–15 minutes (it installs PyTorch and downloads the models). When the Space shows *Running*, open `https://<your-hf-user>-group-pulse-api.hf.space/docs`.
5. The four AMI demo sessions are loaded automatically on every start. The Space's disk is not permanent: uploads are lost on a restart (they are deleted after 24 hours anyway).

Optional, persistent database: create a free Supabase project, copy its **Connection string (URI)**, and add it to the Space as a **secret** named `GP_DATABASE_URL`, changing the prefix to `postgresql+psycopg://`. Add `psycopg[binary]` to the Dockerfile's pip line. Not tested yet; SQLite works out of the box.

## 3. Uptime check (T6.5)

GitHub → **Settings → Secrets and variables → Actions → Variables**: `API_URL` = `https://<your-hf-user>-group-pulse-api.hf.space`. The **uptime** workflow then pings `/health` every 5 minutes; a failed run means the API did not answer, and GitHub emails you. The pings also keep the free Space from sleeping.

## 4. Full app on Vercel (optional)

1. vercel.com → **Add New → Project** → import this GitHub repo.
2. **Root Directory**: `web`. Framework: Vite (detected).
3. **Environment Variables**: `VITE_API_URL` = your Space URL from step 2.
4. Deploy. `web/vercel.json` already handles page refreshes on deep links.

## 5. Error tracking (optional, T6.4)

Create a free Sentry project (Python / FastAPI), copy its **DSN**, and add it to the Hugging Face Space as a secret named `SENTRY_DSN`. Without it, error tracking is simply off.

## Check the launch (T6.x)

| Test | How to check |
| --- | --- |
| T6.1 page loads | Open the Pages or Vercel URL. If the Space is asleep, the Sessions page says the server is waking up and retries until it answers. |
| T6.2 AMI demo with no upload | Open any of the 4 demo meetings. |
| T6.3 videos deleted | Upload a short clip; `has_video` in `/sessions/<id>` turns false within an hour of the 24-hour expiry (sweep runs hourly). |
| T6.4 error tracking | With `SENTRY_DSN` set, errors appear in Sentry. |
| T6.5 uptime | **Actions → uptime** shows a green run every ~5 minutes. |

## Run it locally instead

```bash
pip install -e ".[api,perception,pose]"
python -m grouppulse.api.import_ami demo/sessions
python -m grouppulse.api --port 8000            # API + docs at http://127.0.0.1:8000/docs
cd web && npm install && npm run dev           # app at http://localhost:5173
```

Or the backend in Docker: `docker build -f deploy/huggingface/Dockerfile -t grouppulse-api . && docker run -p 7860:7860 grouppulse-api`.
