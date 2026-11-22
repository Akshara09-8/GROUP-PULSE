# Group Pulse

Detect the dominant speaker, low rapport or a disengaged member of a small group (3–5 people) from **gaze and body orientation alone**, and show it in a free web demo.

Planning doc: phases, test cases, tech stack and database options live in the *Group Pulse — Development Plan* doc. UI colour schemes: `design/group-pulse-ui-schemes.html` (also in Figma: *Group Pulse — UI Colour Schemes*).

## Status

| Phase | What | State |
| --- | --- | --- |
| 0 | Setup and data (repo, AMI loader, alignment, splits) | Done; checks pass on all 5 demo meetings |
| 1 | Perception pipeline (close-up and single-camera) | Done; T1.4 and T1.5 below target (see verdict). Single-camera uploads are experimental |
| 2 | Social metrics (dominance, engagement, rapport) | Done; T2.1–T2.8 pass; attention uses adjusted counting |
| 3 | Backend and storage (FastAPI, SQLite/Supabase) | Done; T3.1–T3.7 pass on SQLite; uploads processed end to end; Supabase untested |
| 4 | Frontend UI (React) | Done; four pages, four colour schemes, video with gaze overlay, 17 tests |
| 5 | Evaluation | Done: reproducible report in `reports/`. T5.3–T5.5 pass; T1.5, T5.1, T5.2 fail (see below) |
| 6 | Deploy and launch | Built and tested; going live needs your accounts: see [DEPLOY.md](DEPLOY.md) |

Tests: 141 Python (`pytest`) and 17 web (`npm test`).

## Layout

```
pipeline/grouppulse/   Python package: data, perception, social metrics, api, evaluation
configs/               Meeting lists, room layouts, calibrated gaze models, social-metric settings
demo/sessions/         Pre-computed AMI demo sessions shown without uploads
deploy/huggingface/    Backend container for a Hugging Face Docker Space
scripts/               setup.ps1, build_tracks.py (bulk processing), export_demo.py (static demo)
reports/               Phase 5 evaluation report and results
tests/                 pytest suite; tests/fixtures has a tiny AMI-style sample
api/                   Backend notes (code is in pipeline/grouppulse/api)
web/                   React + Vite frontend
design/                UI colour-scheme mockups
data/                  Downloaded data (git-ignored)
```

## Setup

Windows (PowerShell):

```powershell
./scripts/setup.ps1          # creates .venv, installs the package with dev extras, runs tests
```

Linux / macOS:

```bash
make setup && make test
```

Docker (no local Python needed):

```bash
docker compose -f docker/docker-compose.yml run --rm pipeline pytest
```

## Getting AMI data

```bash
# See what would be downloaded (no network writes)
python -m grouppulse.data.ami_download --dry-run

# Annotations (~23 MB) + close-up and corner views for the 5 demo meetings (~1.5 GB)
python -m grouppulse.data.ami_download --annotations --meetings demo --views Closeup1 Closeup2 Closeup3 Closeup4 room

# Check what you have
python -m grouppulse.data.check --data data/raw
```

Meeting lists and URL patterns are in `configs/ami.toml`. Splits are made **by meeting series** (for example all of ES2002a–d go to the same split) because a series reuses the same four people; splitting by single meeting would leak identities between train and test.

## Perception (Phase 1)

```bash
# Gaze and torso direction from the four close-up cameras (torso needs `pip install -e .[pose]`)
python -m grouppulse.perception.run --data data/raw --meeting ES2002a --fps 5 --max-seconds 120

# Gaze-target accuracy against AMI's own focus-of-attention labels (only the `focus` meetings)
python -m grouppulse.perception.evaluate --data data/raw --meeting ES2008a

# Frames + an empty CSV for hand labelling (T1.3/T1.4)
python -m grouppulse.perception.labels --data data/raw --meetings demo --n 200
```

### Phase 1 verdict (16 Nov 2026)

Scored on 200 frames sampled evenly from the 5 demo meetings (`perception.labels --sheets/--predict/--score`). The labels were made by Claude from contact sheets, one annotator, in 8 coarse direction bins; they are in `labels.csv` beside the frames and can be corrected and re-scored.

| Test | Result | Target | Verdict |
| --- | --- | --- | --- |
| T1.1 face found when the person is visible | 96.3% (0 false faces on empty seats) | ≥ 95% | Pass |
| T1.2 ID switches | Not applicable: one close-up per seat, identity comes from the camera | ≤ 1 / 5 min | N/A |
| T1.3 head direction | 63% exact bin, 95.5% within one bin; medians sit on bin centres (0°, ±28°, ±65°) | ≤ 10° error | Not measurable from bins; no bias seen |
| T1.4 torso direction (8-way) | 51% exact, 98% within one bin; best balanced accuracy 63% | ≥ 80% | **Fail** |
| T1.5 gaze target | 60.4% held-out (ES2008a, both halves); torso and an HMM add nothing | ≥ 70% | **Fail** |
| T1.6 partly hidden person | kept and flagged low-confidence | — | Pass |
| T1.7 / T1.8 low light, face turned away | no crash; gaze "unknown" when no face | — | Pass |
| T1.9 speed | 0.87–0.93× real time on CPU with torso | ≤ 2× | Pass |

Why T1.4 fails: 139 of 183 labelled torsos face the camera, and shoulder width (arccos) is very sensitive near frontal, so forward-facing frames read as ±25–35°. Settings that score higher overall only do so by calling everything "front" (balanced accuracy falls). Phase 2 measures "turned out of the group" relative to each person's own usual direction, so a constant offset does not hurt it.

Why T1.5 fails: AMI close-ups are 352×288 and MediaPipe cannot read faces turned past ~60°. Next options: an eye-gaze network on face crops (L2CS-Net), and cross-meeting validation on the 11 labelled Idiap meetings.

### Torso direction

YOLO-pose (Ultralytics) reads shoulder keypoints from **each person's close-up**, not the room camera: the Edinburgh room camera is a fisheye where YOLO found only 0–2 of 4 people, while close-ups showed both shoulders in 90–100% of sampled frames. Torso is sampled at 1 fps and held in between (`--torso-fps`), which keeps the whole pipeline at 0.93× real time on CPU with the median within 3.8° of a full 5 fps run. `--torso room` uses the room camera instead (works on Idiap's `C` view). Weights are fetched into `models/` with Python's `urllib`, because Ultralytics' own downloader fails certificate checks on machines whose antivirus re-signs HTTPS.

There is no torso ground truth in AMI; on 2 minutes of ES2002a, head and torso turned the same way in 98–100% of frames where both were turned more than 20°. T1.4 still needs hand labels.

### Gaze-target calibration

```bash
python -m grouppulse.perception.run --data D --meeting ES2008a --no-torso --gaze-model geometry
python -m grouppulse.perception.calibrate --data D --meetings ES2008a   # writes configs/vfoa_edinburgh.toml
```

`run` uses the learned model automatically once `configs/vfoa_<site>.toml` exists, and uses the **hybrid** face reader when `mediapipe` is installed (`pip install -e .[perception]`): YuNet finds the face, MediaPipe Face Landmarker reads a 478-point mesh and eye direction from a crop around it. MediaPipe cannot read near-profile faces (turned past about 60°); those rows keep YuNet's head pose and the model's yaw/pitch fallback.

Results on ES2008a, fitting on one half and testing on the other, averaged over both directions (12 Nov 2026):

| Face reader | Features | Held-out accuracy |
| --- | --- | --- |
| — | Always guess the most common target | 27.9% |
| YuNet | Geometric seat layout | 14.3% |
| YuNet | Learned VFOA, yaw + pitch | 48.0% |
| Hybrid | Learned VFOA, yaw + pitch | 48.2% |
| **Hybrid** | **Learned VFOA, yaw + pitch + eye direction (saved)** | **60.4%** (68.1% / 52.6% by direction) |

T1.5 needs 70%, so it is not met yet. Eye direction is what helps: it catches glances down at the table that head pose misses. Only one Edinburgh meeting has focus labels, so these numbers come from a single meeting; cross-meeting validation needs the Idiap (IS) labelled meetings.

People are named by seat: P*N* is the person filmed by Close-up *N*. AMI speaker letters are not in camera order (in ES2002a, speaker B is on Close-up 4), so `seat_of_speaker` maps them using `meetings.xml`. Room geometry for turning head pose into a gaze target is in `configs/seat_layouts.toml`; those numbers are first guesses and must be calibrated on real frames.

## Social metrics (Phase 2)

```bash
# Session summary JSON (per person, group, 10 s windows, disengaged spells, robot addressee)
python -m grouppulse.social.report --tracks data/tracks/ES2002a.csv --out data/sessions/ES2002a.json

# T2.2: dominance ranking vs who spoke most (AMI speaking time; used for scoring only, never as input)
python -m grouppulse.social.report --data D --tracks-dir D/tracks --evaluate ES2002a ES2002b ES2003a
```

Settings are in `configs/social.toml`. Dominance is currently **gaze only** (share of the others' gaze a person receives): on ES2008a, "facing" computed from the uncalibrated seat layout ranked speakers in reverse and is switched off until the layout is calibrated.

**T2.2 results (16 Nov 2026):** dominance rank vs speaking-time rank (Spearman, 4 people)

| Meeting | Gaze source | Top-1 | Rank corr |
| --- | --- | --- | --- |
| ES2008a | AMI true gaze labels | hit | +1.00 |
| ES2008a | predicted (model fitted on this meeting: optimistic) | hit | +1.00 |
| ES2002a, ES2002b, ES2003a | predicted (unseen meetings) | 0 / 3 | −0.20 mean |
| same | predicted, each person's pose re-centred | 1 / 3 | +0.07 mean |

The metric itself works (true gaze → perfect ranking), but the gaze-target model fitted on one meeting's four people does not carry over to other people: on unseen meetings it predicts "P2" far too often. Fix: train the gaze model on many groups (the 11 labelled Idiap meetings). Caveat: in all four Edinburgh meetings the most talkative person sits at seat P4 (the project manager's seat), so seat alone would score 3/3; future evaluation needs meetings where the leader's seat varies.

## Evaluation (Phase 5)

```bash
python -m grouppulse.evaluation --data path/to/ami-data --idiap-tracks path/to/ami-data/tracks_idiap \
    --edinburgh-tracks path/to/ami-data/tracks_full --out reports --repro
```

One command runs every experiment and writes [`reports/phase5_report.md`](reports/phase5_report.md) and `reports/phase5_results.json`. Intervals are bootstrap 95% over meetings (2000 resamples, seed 13); two runs give byte-identical output. Results of 20 Nov 2026, 11 labelled Idiap meetings (IS1008c's video could not be downloaded) plus the Edinburgh set:

| Test | Result | Verdict |
| --- | --- | --- |
| T1.5 gaze target ≥ 70% | 59.5% [54.0, 65.4] held out by meeting series; baseline 28.7% | Fail |
| T5.1 dominance vs seat baseline | Idiap ρ +0.22 (adjusted counting; raw +0.18) vs +0.44; Edinburgh −0.20 vs +0.35 | Fail |
| T5.2 disengagement F1 ≥ 0.6 | 11.1% [3.9, 18.1] against a gaze-label proxy | Fail |
| T5.3 ablation gaze / orientation / both | Table produced (Edinburgh) | Pass |
| T5.4 reproducible | Byte-identical JSON across two runs | Pass |
| T5.5 confidence intervals | Reported for every mean | Pass |

What the numbers say:

- **The hypothesis holds on true gaze.** With AMI's labelled gaze, attention received ranks speakers at ρ +0.60 [+0.31, +0.85] on Idiap and +1.00 on ES2008a.
- **Engagement over time is tracked.** Window engagement from predicted gaze correlates +0.62 [+0.53, +0.69] with the true-gaze version, even though rare 30 s spells are mostly missed.
- **The gaze reader is the bottleneck.** About 60% gaze accuracy is not enough to rank people: the model over-predicts "P2" as the target (most looked-at in 7 of 11 Idiap meetings and all 3 unseen Edinburgh meetings).
- **Where "P2" comes from, and the fix (22 Nov 2026).** Measured confusion matrices show two camera-specific errors in the Idiap room: glances at the *whiteboard* are read as P2 (64% from Close-up 1, 53% from Close-up 3), and glances at P4 are read as P2 (39% from Close-up 3). P2's attention received was over-estimated in 9 of 11 held-out meetings. Session-level attention now uses **adjusted counting** (`perception/quantify.py`): each camera's confusion is measured on held-out training series and undone when counting. Top-1 rose from 2/11 to 5/11, rank correlation only from +0.18 to +0.22, still below the seat baseline. (A prototype showed +0.44, but that came from a non-standard solver and did not hold up.) Flat class priors (+0.22) and per-meeting EM prior adaptation (accuracy fell ~10 points) were also tried and not adopted.
- **Seating predicts talking.** The project manager tends to take the same seat, so a seat baseline is strong (Idiap top-1 73%). Any future claim has to beat it.

## Uploads (single camera, experimental)

An upload is one camera filming the whole group, unlike AMI's one close-up per person. `perception/single.py` finds every face (YuNet), links detections into fixed seats (tracks locate the seats; every detection goes to the nearest seat), reads head pose (MediaPipe mesh on a crop, else YuNet) and picks gaze targets in the image plane: looking down is "table", otherwise the other head nearest to the gaze line within 35°, else "away". Torso comes from YOLO-pose at 1 fps. It runs at about 0.93× real time on CPU at 2.5 fps sampling.

It is **not validated**: there are no labels for single-camera video, and the image has no depth, so a person further along the gaze line can be chosen over a nearer one. Every upload's results carry a warning saying so. The worker saves per-frame face boxes and targets beside the video; the Session page plays the video with boxes and gaze arrows in sync (T4.2), and both are deleted after 24 hours.

## Deploy (Phase 6)

All deploy pieces are in the repo and tested: a Docker image for a Hugging Face Space (`deploy/huggingface/`), a static demo build that needs no server (`VITE_DEMO=1`, published to GitHub Pages by `.github/workflows/pages.yml`), a Hugging Face deploy workflow, a 5-minute uptime ping, an hourly video clean-up, a "server is waking up" state in the app, and optional Sentry error tracking. Going live needs your own free accounts; [DEPLOY.md](DEPLOY.md) lists every step. The Docker image has not been built on this machine (Docker is not installed); its first build happens on Hugging Face.

## Data credit and licence

This project uses the **AMI Meeting Corpus**, released under the **Creative Commons Attribution 4.0 International licence (CC BY 4.0)**.
Carletta, J. et al. (2005). *The AMI Meeting Corpus: A Pre-announcement.* Machine Learning for Multimodal Interaction (MLMI). Corpus home: https://groups.inf.ed.ac.uk/ami/corpus/

Uploaded videos in the demo are deleted after 24 hours. No face recognition is performed; participants are labelled P1–P5 by seat.
