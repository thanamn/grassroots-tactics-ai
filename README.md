# Grassroots Tactics AI

Explainable AI tactical software for low-resource football teams.
A GenAI + HCI course project — Wizard-of-Oz prototype using ordinary video,
player tracking, geometric tactical metrics, and LLM-generated coach explanations.

## Quick start

```bash
# 1. Create virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API key
cp .env.example .env
# edit .env and paste your Gemini API key
# (free key from https://aistudio.google.com/apikey)

# 4. Try the prompt exploration script (no clips needed yet)
python notebooks/prompt_exploration.py

# 5. Once you have a tracking JSON, run the full pipeline
python -m src.tracking --input data/clips/sample.mp4
python -m src.metrics  --input data/tracking/sample.json
python -m src.explainer --input data/cache/sample_metrics.json

# 6. Launch the demo UI
streamlit run app/streamlit_app.py
```

## Project structure

```
grassroots-tactics-ai/
├── src/
│   ├── config.py          # paths, constants, FPS, pitch dimensions
│   ├── tracking.py        # YOLOv8 + ByteTrack → tracking JSON
│   ├── metrics.py         # convex hull, centroid, spacing → metrics JSON
│   ├── visualizer.py      # render heatmap/overlay onto video
│   └── explainer.py       # call Gemini API → tactical explanation
├── app/
│   └── streamlit_app.py   # the demo UI (video + heatmap + AI text)
├── prompts/
│   └── tactical_explainer.py  # prompt templates
├── notebooks/
│   └── prompt_exploration.py  # try prompts with mock metrics
├── study/
│   ├── sus_questionnaire.md   # standard SUS for week 6
│   └── interview_guide.md     # qualitative interview script
├── data/
│   ├── clips/             # raw video (gitignored)
│   ├── tracking/          # YOLO output
│   └── cache/             # pre-computed Wizard-of-Oz results
├── requirements.txt
├── .env.example
└── .gitignore
```

## Wizard-of-Oz architecture

The user-facing app does NOT run YOLO or call Claude in real time.
We pre-compute everything offline and cache it as JSON + rendered video.
The Streamlit app just loads cached results. This is deliberate — it
removes runtime risk during user studies and lets us iterate on
explanation quality without re-running the CV pipeline every time.

```
[clip.mp4] → tracking.py → [tracking.json]
                              ↓
                          metrics.py → [metrics.json]
                              ↓                 ↓
                       visualizer.py        explainer.py
                              ↓                 ↓
                      [overlay.mp4]    [explanation.json]
                              ↓                 ↓
                              └──── streamlit_app.py ────→ user
```

## Data plan

- **Pro tactical clips (EPL/World Cup, ~2 clips)** — for technical validation.
  Easy to track, clean colour separation between teams.
- **Grassroots clip (~1 clip)** — for user study. Lower quality, real
  target context. Search YouTube: "amateur football match", "Sunday
  league football", "youth football tactical view".

## Course deliverable map

| Week | Deliverable | Files / output |
|---|---|---|
| 4 (this week) | End-to-end pipeline runs on 1 clip | tracking.py, metrics.py, explainer.py all green |
| 5 | Streamlit UI + pilot test among ourselves | app/streamlit_app.py polished |
| 6 | User study with 5–8 participants | study/ filled with raw + analysed data |
| 7 | Full paper draft (≥24 pages) submitted | paper/ folder |
| 8 | Final paper + presentation + demo | submission package |

## Tactical scope

This prototype analyses **team spacing / compactness only**.
Specifically: convex hull area, team centroid, inter-player distance variance.
Other tactical concepts (pressing, transitions, defensive shape) are mentioned
in the paper as future work but NOT implemented.
