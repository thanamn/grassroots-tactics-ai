# Grassroots Tactics AI

Explainable AI tactical software for low-resource football teams.
A GenAI + HCI course project — live-pipeline prototype using ordinary video,
player tracking, geometric tactical metrics, and LLM-generated coach explanations.

## Quick start

```powershell
# 1. Create virtual environment (Python 3.11)
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API key
Copy-Item .env.example .env
# edit .env and paste your DeepSeek API key

# 4. Launch the web app (serves frontend + API on the same port)
.venv\Scripts\uvicorn.exe backend.main:app --reload --port 8000
# open http://localhost:8000 in your browser
```

Upload any football clip (up to 500 MB) through the browser UI. The pipeline
runs automatically: tracking → team assignment → metrics → visualizer → AI explanation.

### Offline pipeline (CLI)

```powershell
# Step-by-step if you prefer the terminal
.venv\Scripts\python.exe -m src.tracking --input data/clips/clip.mp4
.venv\Scripts\python.exe scripts/assign_teams.py --tracking data/tracking/clip.json --video data/clips/clip.mp4
.venv\Scripts\python.exe -m src.metrics  --input data/tracking/clip.json
.venv\Scripts\python.exe -m src.visualizer --video data/clips/clip.mp4 --tracking data/tracking/clip.json
.venv\Scripts\python.exe -m src.explainer --input data/cache/clip_metrics.json

# Fallback Streamlit UI (loads pre-computed cached results)
.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

### Utility scripts

```powershell
# Re-run assign_teams -> metrics -> visualizer on all uploaded jobs
# (after a code fix, without re-running slow YOLO tracking)
.venv\Scripts\python.exe scripts/rerun_pipeline.py

# Single job only:
.venv\Scripts\python.exe scripts/rerun_pipeline.py --job <job_id>

# Retroactively fix GK cls labels in existing tracking JSONs
# (applies new ≥5 frame / >50% threshold without re-running YOLO)
.venv\Scripts\python.exe scripts/fix_tracking_cls.py
```

## Project structure

```
grassroots-tactics-ai/
├── src/
│   ├── config.py           # paths, constants, FPS, pitch dimensions
│   ├── tracking.py         # YOLOv8 football model + IoU centroid tracker → tracking JSON
│   ├── metrics.py          # convex hull area, centroid, spread → metrics JSON
│   ├── ball_metrics.py     # possession % and pass stats from ball + player data
│   ├── visualizer.py       # pairwise player-connection lines + GK marker → overlay MP4
│   ├── explainer.py        # DeepSeek API → headline / implication / coaching_cue
│   └── coach_chat.py       # free-form Q&A grounded in spacing metrics
├── backend/
│   ├── main.py             # FastAPI server (upload, jobs, chat, re-explain)
│   ├── pipeline_runner.py  # subprocess: runs full pipeline for a job
│   └── jobs.py             # job state (JSON files in data/jobs/)
├── web/
│   ├── index.html          # React SPA entry point (React loaded from CDN)
│   └── app.jsx             # upload form, live job-status polling, results view
├── scripts/
│   ├── assign_teams.py         # two-pass jersey-colour clustering (DBSCAN → k-means)
│   ├── fix_tracking_cls.py     # retroactive GK cls label fix (no re-tracking needed)
│   ├── rerun_pipeline.py       # re-run stages 1-3 for all/one job after code fixes
│   ├── find_active_play.py     # motion scorer → best 20-second active-play window
│   └── download_active_clips.py  # download UCL clips + auto-trim to active play
├── app/
│   └── streamlit_app.py    # fallback UI for cached clips (user-study control)
├── prompts/
│   └── tactical_explainer.py   # bilingual (en/th) prompt templates
├── notebooks/
│   └── prompt_exploration.py   # try prompts with mock metrics (no clip needed)
├── models/
│   └── football_players.pt     # YOLOv8 weights (classes: ball/GK/player/referee)
├── study/
│   ├── sus_questionnaire.md    # 10 standard SUS items + 5 custom trust/explainability items
│   └── interview_guide.md      # 12-question semi-structured interview guide
├── data/
│   ├── clips/              # raw video uploads (gitignored)
│   ├── tracking/           # YOLO + tracker output JSON (gitignored)
│   ├── cache/              # overlay MP4s, metrics JSON, explanation JSON (gitignored)
│   └── jobs/               # job status records (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Architecture — live pipeline

The browser uploads a clip to the FastAPI backend, which spawns a background
process that runs the full pipeline and writes results to `data/cache/`. The
frontend polls `/api/jobs/{id}` until status is `done`, then loads the overlay
video and AI explanation.

```
[clip.mp4]  →  src/tracking.py          →  data/tracking/<id>.json
               scripts/assign_teams.py     (jersey colour clustering, DBSCAN/k-means)
               src/metrics.py           →  data/cache/<id>_metrics.json
               src/ball_metrics.py         (possession %, pass stats)
               src/visualizer.py        →  data/cache/<id>_overlay.mp4
               src/explainer.py         →  data/cache/<id>_explanation_<lang>.json
                                               ↓
                       web/ (React, polls /api/jobs/<id>)
                       app/streamlit_app.py  (fallback, cached clips only)
```

### Visualizer — how the overlay is drawn

- **Player-connection lines** — for each team, every pair of outfield players
  within 25 % of the frame's longest dimension is connected with a line.
  Shows the actual formation mesh (not just the outer convex-hull boundary).
- **GK detection** — the football model emits `cls=1` for goalkeepers.
  `tracking.py` accumulates per-track class votes; a track is labelled GK
  when **≥5 frames were seen AND >50 % of detections were class 1**.
  GKs are drawn with a white-cross marker and excluded from formation lines
  so they don't distort the outfield shape.
- **Convex hull fill** — a 13 % opacity fill shows each team's occupied zone.
- **Temporal smoothing** — hull is built from a sliding window of recent
  tracked frames (`SMOOTH_W × vid_stride`) so a briefly-occluded player
  doesn't cause the hull to collapse and flicker.

### Team colour clustering — how assign_teams.py works

1. Sample every 5th frame; crop the shirt zone (15–50 % of bbox height);
   mask out grass-green pixels in HSV; compute median L\*a\*b\* colour.
2. Fit **DBSCAN** on the a\*b\* (hue) dimensions — no need to guess k.
   Falls back to k-means k=2 if DBSCAN finds fewer than 2 clusters.
3. If k-means a\*b\* separation < 12 units (dark-navy vs dark kits look
   similar in hue), retry in full 3-D L\*a\*b\* space so luminance helps.
4. In pass 2, two filters before assigning a team label:
   - **Y-boundary guard** — any detection whose bottom edge is below 90 % of
     frame height is silently skipped (catches coaching staff on the touchline).
   - **Colour outlier guard** — any non-GK detection whose 3-D L\*a\*b\*
     distance from both team centres exceeds 40 units gets `team=None`
     (catches referees, linesmen, sideline staff in distinctive colours).
5. All remaining detections are assigned to the nearest team centre.
   Players with `team=None` are excluded from hull, lines, and metrics.

## Data plan

- **5 UCL tactical-view clips** (20 s each, active play only) — downloaded to
  `C:\Users\TSURUGI\Desktop\football_clips\` and uploaded through the web UI.
  Used for technical validation and demo.
- 1 **grassroots clip** (YouTube amateur/Sunday-league) for the user study.
  Lower quality, matches the target deployment context.

## Course deliverable map

| Week | Deliverable | Status |
|---|---|---|
| 4 | End-to-end pipeline + live upload | done |
| 5 | Metrics tuning, overlay visual fixes (outlier filter, GK threshold, y-boundary) | **done** |
| 6 | User study with 5–8 participants | upcoming |
| 7 | Full paper draft (≥24 pages, ACM one-column) | upcoming |
| 8 | Final paper + presentation + live demo | upcoming |

## Tactical scope

This prototype analyses **team spacing / compactness only**.
Metrics: convex hull area, team centroid, inter-player distance variance,
possession %, pass count.
Other tactical concepts (pressing, transitions, formation recognition) are
discussed in the paper as future work but not implemented.
