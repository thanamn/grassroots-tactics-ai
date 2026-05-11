# Demo Mode Setup

This repo includes a committed, precomputed demo bundle so the site can show a
full upload-style walkthrough without waiting for the live CPU pipeline.

## What is committed

The curated demo bundle is:

- `data/eval/runs/psg_annotated_base_convex_v1/`

It contains:

- `trimmed_clips/` — original clip segments used by the demo
- `overlays/` — polished convex/spread overlay videos
- `refined_tracking/` — annotation-guided refined tracking JSON
- `metrics/` — per-clip tactical metrics JSON
- `stills/` — preview images used by the dashboard
- `summary.json` — the index consumed by `backend/demo_store.py`

The demo mode is surfaced through:

- `backend/demo_store.py`
- `backend/main.py`
- `web/app.jsx`

## How to run the demo

1. Create and activate the virtual environment.
2. Install dependencies from `requirements.txt`.
3. Start the server:

```powershell
.venv\Scripts\uvicorn.exe backend.main:app --reload --port 8000
```

4. Open `http://localhost:8000`
5. Click `Try demo mode`
6. Choose a demo clip and walk through the normal flow:
   upload-style entry -> tagging -> processing simulation -> done -> analysis

## Why the demo bundle is committed

The live pipeline is intentionally CPU-first and can be slow during a meeting or
supervision demo. The committed bundle guarantees that the UI can always show:

- the same three polished clips
- the same overlays
- the same key moments
- the same analysis page

without depending on model downloads, API availability, or long local runtime.

## What is intentionally *not* committed

Large scratch evaluation artefacts are ignored on purpose. The repo keeps the
demo-ready bundle plus a small amount of evaluation metadata, but not every
benchmark run, frame cache, or intermediate export.

Important examples that stay local-only:

- `data/eval/frames/`
- most of `data/eval/runs/`
- large benchmark report dumps in `data/eval/reports/`

## How to refresh the committed demo bundle

If you already have the local PSG evaluation assets and the base tracking run on
disk, rebuild the committed demo bundle with:

```powershell
.venv\Scripts\python.exe scripts\render_annotated_base_convex_overlays.py `
  --manifest psg_annotation_v1 `
  --tracking-run psg_tracking_latest_v1 `
  --preset football_players `
  --run-id psg_annotated_base_convex_v1
```

This script uses:

- `data/eval/annotations/psg_annotation_v1.json` as the hard annotation base
- `data/eval/runs/psg_tracking_latest_v1/football_players/tracking_with_teams/*.json` as dense motion input
- the existing repo visualizer and metrics pipeline for the final overlay style

If those source tracking assets are not present locally, the committed demo
bundle can still be used as-is.

## Git strategy

Recommended practice:

- commit the curated demo bundle
- keep heavyweight benchmark byproducts ignored
- update `summary.json` and the still previews whenever the demo bundle changes
- smoke-test the site by opening demo mode before pushing
