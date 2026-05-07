# Grassroots Tactics AI

Explainable AI tactical software for low-resource football teams.
A GenAI + HCI course project — live-pipeline prototype using ordinary video,
player tracking, geometric tactical metrics, and LLM-generated coach explanations.

## Quick start

```bash
# 1. Create virtual environment (Python 3.11)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API key
cp .env.example .env
# edit .env and paste your DeepSeek API key

# 4. Launch the web app (serves frontend + API on the same port)
.venv\Scripts\uvicorn backend.main:app --reload --port 8000
# open http://localhost:8000 in your browser
```

Upload any football clip (up to 500 MB) through the browser UI. The pipeline
runs automatically: tracking → team assignment → metrics → visualizer → AI explanation.

### Offline pipeline (CLI)

```bash
# Step-by-step if you prefer the terminal
python -m src.tracking --input data/clips/clip.mp4
python scripts/assign_teams.py --tracking data/tracking/clip.json --video data/clips/clip.mp4
python -m src.metrics  --input data/tracking/clip.json
python -m src.visualizer --video data/clips/clip.mp4 --tracking data/tracking/clip.json
python -m src.explainer --input data/cache/clip_metrics.json

# Fallback Streamlit UI (loads pre-computed cached results)
streamlit run app/streamlit_app.py
```

## Project structure

```
grassroots-tactics-ai/
├── src/
│   ├── config.py           # paths, constants, FPS, pitch dimensions
│   ├── tracking.py         # YOLOv8 football model + IoU centroid tracker → tracking JSON
│   │                         saves cls=1 (GK) / cls=2 (player) per detection
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
│   └── assign_teams.py     # two-pass jersey-colour clustering (DBSCAN → k-means)
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
│   ├── tracking/           # YOLO + tracker output JSON
│   ├── cache/              # overlay MP4s, metrics JSON, explanation JSON
│   └── jobs/               # job status records
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
  This shows the actual formation mesh (not just the outer convex-hull boundary).
- **GK detection** — the football model emits `cls=1` for goalkeepers.
  `tracking.py` accumulates per-track class votes; a track is labelled GK
  when >40 % of its detections were class 1. GKs are drawn with a white-cross
  marker and excluded from the formation lines so they don't distort the
  outfield shape.
- **Convex hull fill** — a 13 % opacity fill shows each team's occupied zone.
- **Fallback** — clips tracked before the `cls` field was added use the old
  "drop the player furthest from centroid" heuristic automatically.

## Data plan

- **Pro tactical clips (EPL/UCL, ~2 clips)** — technical validation.
  Clean colour separation, easy to track, gives ground truth for the metrics.
- **Grassroots clip (~1 clip)** — user study. Lower quality, matches the
  target deployment context. Search YouTube: "amateur football match",
  "Sunday league football", "youth football tactical view".

## Course deliverable map

| Week | Deliverable | Status |
|---|---|---|
| 4 | End-to-end pipeline + live upload | done |
| 5 | Metrics tuning (possession/pass), overlay fixes | in progress |
| 6 | User study with 5–8 participants | upcoming |
| 7 | Full paper draft (≥24 pages, ACM one-column) | upcoming |
| 8 | Final paper + presentation + live demo | upcoming |

## Tactical scope

This prototype analyses **team spacing / compactness only**.
Metrics: convex hull area, team centroid, inter-player distance variance,
possession %, pass count.
Other tactical concepts (pressing, transitions, formation recognition) are
discussed in the paper as future work but not implemented.
