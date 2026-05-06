# Project context for Claude Code

This file is read automatically by Claude Code on every session. It is the
condensed memory of an earlier planning conversation — read it first before
suggesting any architectural change.

## What this project is

A GenAI + HCI course project at university (year-2 student team). The deliverable
is a Wizard-of-Oz prototype that takes ordinary football video, computes simple
tactical-spacing metrics from player tracking, and uses an LLM to explain those
metrics in language a grassroots coach can use.

The full concept paper (in the team's submission) frames this as "explainable AI
tactical software for low-resource football teams" and lists six tactical
concepts in scope (shape, spacing, defensive organization, pressing, transitions,
off-ball support). **The prototype implements only one of these: team
spacing/compactness.** The paper presents the others as future work.

## Hard scope rules — do not break

1. **Implement spacing/compactness only.** Do not add pressing, transitions,
   formation detection, or per-player skill metrics, even if it seems easy.
   Scope creep is the biggest risk for this team.

2. **Live pipeline is the product (as of 2026-05-05).** Earlier the project
   was Wizard-of-Oz; that rule was overridden by the user mid-week-4. The
   FastAPI backend at `backend/` accepts video uploads and runs the full
   pipeline (tracking → assign_teams → metrics → visualizer → explainer)
   on the server. Frontend at `web/` polls for status. The Streamlit app
   in `app/streamlit_app.py` is kept as a fallback for cached clips.
   Upload is capped at ~10 min of footage — long videos are CPU-bound and
   will not finish during a user-study session. Do not silently extend
   the cap; if longer videos are needed, route via Colab GPU explicitly.

3. **GenAI + HCI is the centre, not CV.** The course grades on AI explanation
   quality and user-study design, not on player-detection accuracy. If a CV
   shortcut (using SoccerNet annotations, manual team labelling, pro broadcast
   clips) saves time, take it. The user has only ~5 weeks and is working
   essentially solo (teammates are on exam burnout).

4. **Use Gemini, not Claude/OpenAI.** The student is on Gemini free tier with
   `google-genai` SDK (the modern one, not deprecated `google-generativeai`).
   Default model: `gemini-2.5-flash`. JSON output is enforced via
   `response_schema`, not via prompt-side instructions.

5. **The explanation is bilingual (en/th).** Thai users are the realistic target
   for grassroots fieldwork. Both prompt templates exist in
   `prompts/tactical_explainer.py`. Don't drop Thai support to "simplify".

## Architecture — current state

```
[clip.mp4] → src/tracking.py    → data/tracking/<id>.json
                                   (manual team-label step here)
              src/metrics.py    → data/cache/<id>_metrics.json
              src/visualizer.py → data/cache/<id>_overlay.mp4
              src/explainer.py  → data/cache/<id>_explanation_<lang>.json
                                   ↓
                    app/streamlit_app.py loads all four
```

Pipeline modules:

- `src/tracking.py` — YOLOv8 + ByteTrack via `ultralytics`. Uses bbox centroid
  as player position. Team field is left null; manually filled in the JSON
  afterwards (3 clips, ~2 min each, faster than coding colour clustering).
- `src/metrics.py` — convex hull area, team centroid, distance-from-centroid
  std, inter-team centroid distance. Excludes the player furthest from team
  centroid as a goalkeeper proxy. Detects "events" where hull area changes
  ≥25% within 1.5 s and dedupes within 1-second windows.
- `src/visualizer.py` — OpenCV polygon fill + outline + centroid dot per team.
- `src/explainer.py` — `google.genai.Client` with `response_mime_type=
  'application/json'` and a strict `EXPLANATION_SCHEMA`. Returns three
  fields: `headline`, `implication`, `coaching_cue`.
- `prompts/tactical_explainer.py` — versioned prompt templates with explicit
  banned-jargon list (xG, PPDA, half-spaces, Voronoi, etc.) and a `make_mock_metrics()`
  function for prompt iteration without real video.
- `notebooks/prompt_exploration.py` — three named scenarios
  (`defending_phase`, `stable_block`, `stretched_attack`). Supports
  `--render-only` for token-free debugging.
- `app/streamlit_app.py` — sidebar clip picker, language toggle, "show AI
  explanation" toggle (for the user-study control condition), line chart of
  hull area over time with event annotations.
- `study/` — SUS questionnaire (10 standard items + 5 custom items on trust
  and explainability) and a 12-question semi-structured interview guide.

## Course timeline

8-week course, currently in week 4.

- Week 4 (NOW): end-to-end pipeline runs on 1 clip
- Week 5: Streamlit polished, internal pilot
- Week 6: user study with 5–8 grassroots-football participants
- Week 7: full paper draft (≥24 pages, ACM one-column format) submitted
- Week 8: final paper + presentation + live demo

## Data plan

- 1–2 **pro tactical-view clips** (EPL/World Cup, ~15–20 s) for technical
  validation — clean colour separation, easy to track, gives clean ground truth
  for the metrics. Justified in paper as "technical validation against
  high-quality footage".
- 1 **grassroots clip** (YouTube amateur/Sunday-league/youth-football) for the
  user study. Lower quality but matches the target deployment context. This
  is the primary "we built it for grassroots" evidence in the paper.

## Python environment

Project uses a local **venv** at the repo root (Python 3.13). Always use it,
never bare `python` / `python3` — those resolve to the Microsoft Store stub
on Windows machines and break import paths.

```
.venv\Scripts\python.exe        # invoke modules: -m src.metrics, etc.
.venv\Scripts\streamlit.exe     # run the demo app
```

PowerShell example:

```powershell
.\.venv\Scripts\streamlit run app\streamlit_app.py
```

## Conventions and preferences

- The student writes Thai casually with English mixed in. Match that register
  in chat replies. Be direct, not overly hedged. Treat them as a peer building
  with you, not a customer.
- Short, copyable code/commands without surrounding boilerplate. They like
  pasting things straight into a terminal.
- Code style: prose-y docstrings explaining *why* a choice was made, not just
  *what* the function does. The team uses these docstrings as paper-writing
  fodder later.
- Don't propose libraries that need GPU unless asked. They may not have one
  and the fallback plan is Google Colab T4 free tier.

## Things explicitly out of scope (don't suggest)

- Player re-identification across clips
- Event detection beyond the spacing-spike heuristic (no pass detection,
  shot detection, possession chains, etc.)
- Tactical formation recognition (4-3-3 vs 4-4-2 etc.)
- Personal training/fitness/rehabilitation features (the paper's introduction
  explicitly excludes these)
- Real-time inference, video streaming, mobile app, multi-user collaboration
- Comparing against TacticAI or other elite-data systems with the same
  evaluation methodology — we deliberately have a different evaluation frame
  (coach usability + explanation quality, not predictive accuracy)

## When in doubt

Ask the student first. Do not refactor large surfaces autonomously. If a
file change touches more than three files, propose the diff in chat before
applying it.
