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

### Demo walkthrough

The repo includes a curated precomputed demo bundle so you can show the full
product flow without waiting for live CPU processing.

1. Start the app:
   `.venv\Scripts\uvicorn.exe backend.main:app --reload --port 8000`
2. Open [http://localhost:8000](http://localhost:8000)
3. Click `Try demo mode`
4. Pick one of the three PSG demo clips
5. Walk through the normal tagging, processing, done, and analysis screens

The committed demo assets live in:

- `data/eval/runs/psg_annotated_base_convex_v1/trimmed_clips/`
- `data/eval/runs/psg_annotated_base_convex_v1/overlays/`
- `data/eval/runs/psg_annotated_base_convex_v1/refined_tracking/`

The browser demo is served by `backend/demo_store.py` and reuses the same
analysis UI as a normal finished job.

For maintenance and rebuild instructions, see `study/demo_mode_setup.md`.

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

### Evaluation benchmark

The current local evaluation set lives in `data/eval/`. It contains three
20-second PSG vs Milan clips with 40 manually annotated frames and 840 player
points. It also has ball-center labels for 39 visible-ball frames. Use it for
player tracking, team assignment, detector comparison, and first-pass ball
localization. It does not yet validate true possession or passing because those
need explicit possession-owner and pass-event labels.

Open the annotation UI at `http://localhost:8000/annotator` while the FastAPI
server is running. Use `Ball` mode, or shortcut `4`, to click/drag the football
on each frame. The side panel also records possession owner or `Ball absent`.

```powershell
$cfg = Get-Content study\football_finetuned_model_presets.json | ConvertFrom-Json
$presets = $cfg.PSObject.Properties.Name

foreach ($preset in $presets) {
  .venv\Scripts\python.exe scripts/run_tracking_benchmark.py `
    --manifest-id psg_annotation_v1 `
    --run-id psg_tracking_football_finetuned_v1 `
    --preset-file study\football_finetuned_model_presets.json `
    --preset $preset
}

$playerArgs = @()
$cfg.PSObject.Properties |
  Where-Object { $_.Value.player_classes.Count -gt 0 } |
  ForEach-Object { $playerArgs += @("--model", $_.Name) }

.venv\Scripts\python.exe scripts/score_tracking_benchmark.py `
  --manifest-id psg_annotation_v1 `
  --run-id psg_tracking_football_finetuned_v1 `
  --baseline uisikdag_yolov8_football_players `
  --bootstrap-iterations 5000 `
  @playerArgs

$ballArgs = @()
$cfg.PSObject.Properties |
  Where-Object { $_.Value.ball_classes.Count -gt 0 } |
  ForEach-Object { $ballArgs += @("--model", $_.Name) }

.venv\Scripts\python.exe scripts/score_ball_benchmark.py `
  --manifest-id psg_annotation_v1 `
  --run-id psg_tracking_football_finetuned_v1 `
  @ballArgs
```

The fair football-finetuned comparison uses
`study\football_finetuned_model_presets.json`. It compares 10 player-capable
football detectors and 12 ball-capable football/soccer-ball detectors. Generic
COCO models are kept only as control baselines, because their class maps are
not trained specifically for football balls or football player roles.

Method details and the latest comparison tables are in
`study/tracking_evaluation_methodology.md`.

### LLM evaluation benchmark

The LLM is used in two places: `src/explainer.py` turns metrics into a
schema-bound tactical explanation, and `src/coach_chat.py` answers short
coach questions grounded in the same metrics. Both currently use the OpenAI
SDK pointed at DeepSeek's OpenAI-compatible API, with `deepseek-v4-pro` as
the default production model.

Use the LLM benchmark to compare models, prompt variants, schema reliability,
grounding, scope control, jargon avoidance, Thai output quality, latency, and
token use. The main model comparison should keep the prompt fixed at
`current_v2`; prompt variants are evaluated separately as ablations.

```powershell
# See all model presets, including automated API routes and manual fallbacks
.venv\Scripts\python.exe scripts/run_llm_benchmark.py --list-presets

# Smoke-test prompt/case rendering without API calls
.venv\Scripts\python.exe scripts/run_llm_benchmark.py --dry-run --run-id llm_dry_run_v1

# Same smoke test through the one-command runner
powershell -ExecutionPolicy Bypass -File scripts\run_llm_eval_plan.ps1 -Mode smoke -RunId llm_smoke_v1

# Production-like default comparison: DeepSeek V4 Pro vs V4 Flash
.venv\Scripts\python.exe scripts/run_llm_benchmark.py --run-id llm_default_v1 --repeat 2
.venv\Scripts\python.exe scripts/score_llm_benchmark.py --run-id llm_default_v1

# Same default run, skipping providers whose keys are missing
powershell -ExecutionPolicy Bypass -File scripts\run_llm_eval_plan.ps1 -Mode default -RunId llm_default_v1 -Repeat 2

# 10+ model comparison, assuming the relevant API keys are in .env
.venv\Scripts\python.exe scripts/run_llm_benchmark.py `
  --run-id llm_10plus_v1 `
  --prompt current_v2 `
  --repeat 2 `
  --preset deepseek_v4_pro `
  --preset deepseek_v4_flash `
  --preset openai_gpt_5_4_mini `
  --preset openai_gpt_5_1 `
  --preset google_gemini_2_5_flash `
  --preset google_gemini_2_5_flash_lite `
  --preset mistral_small_latest `
  --preset groq_llama_3_1_8b `
  --preset groq_llama_3_3_70b `
  --preset groq_qwen3_32b `
  --preset together_llama_3_3_70b_turbo
.venv\Scripts\python.exe scripts/score_llm_benchmark.py --run-id llm_10plus_v1

# One-command 10+ model plan. It runs only providers with keys in .env.
powershell -ExecutionPolicy Bypass -File scripts\run_llm_eval_plan.ps1 -Mode 10plus -RunId llm_10plus_v1 -Repeat 2
```

Manual packets for models without a simple API path can be generated with:

```powershell
.venv\Scripts\python.exe scripts/run_llm_benchmark.py `
  --run-id llm_manual_packets_v1 `
  --manual-packets-only `
  --all-presets `
  --prompt current_v2
```

Method details are in `study/llm_evaluation_methodology.md`. Model presets,
prompt variants, and cases live in `study/llm_model_presets.json`,
`study/llm_prompt_variants.json`, and `study/llm_eval_cases.json`.
Manual web/playground links and paste rules are in
`study/manual_llm_access_list.md`.

If you collect an answer manually from a provider website, put the answer in a
text file and append it to the run:

```powershell
.venv\Scripts\python.exe scripts/import_manual_llm_result.py `
  --run-id llm_manual_packets_v1 `
  --model-preset anthropic_claude_sonnet_4_6 `
  --prompt current_v2 `
  --case explainer_defensive_stretch_en `
  --response-file path\to\copied_answer.txt
```

## Project structure

```
grassroots-tactics-ai/
├── src/
│   ├── config.py           # paths, constants, FPS, pitch dimensions
│   ├── tracking.py         # Ultralytics detector + IoU centroid tracker → tracking JSON
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
│   ├── run_tracking_benchmark.py # detector comparison runner
│   ├── score_tracking_benchmark.py # benchmark scorer + bootstrap comparisons
│   ├── score_ball_benchmark.py # ball localisation and possession-proxy scorer
│   ├── run_llm_benchmark.py    # LLM model/prompt benchmark runner
│   ├── score_llm_benchmark.py  # LLM schema/grounding/scope scorer
│   ├── import_manual_llm_result.py # append website/playground LLM outputs
│   ├── run_llm_eval_plan.ps1   # one-command LLM evaluation plans
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
│   ├── interview_guide.md      # 12-question semi-structured interview guide
│   ├── football_finetuned_model_presets.json # fair football-specific detector presets
│   ├── model_presets_10plus.json # expanded detector presets for 15-model evaluation
│   ├── llm_model_presets.json  # automated/manual LLM comparison candidates
│   ├── llm_prompt_variants.json # prompt ablation variants
│   ├── llm_eval_cases.json     # synthetic explainer/chat cases for repeatable LLM tests
│   ├── llm_evaluation_methodology.md
│   ├── manual_llm_access_list.md # web/playground links and paste rules
│   └── tracking_evaluation_methodology.md
├── data/
│   ├── clips/              # raw video uploads (gitignored)
│   ├── tracking/           # YOLO + tracker output JSON (gitignored)
│   ├── cache/              # overlay MP4s, metrics JSON, explanation JSON (gitignored)
│   ├── jobs/               # job status records (gitignored)
│   └── eval/               # local evaluation clips, labels, runs, and reports
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
