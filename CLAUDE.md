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
   pipeline (tracking → assign_teams → metrics + ball_metrics → visualizer → explainer)
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
[clip.mp4] → src/tracking.py         → data/tracking/<id>.json
              scripts/assign_teams.py   (auto team colour clustering)
              src/metrics.py          → data/cache/<id>_metrics.json
              src/ball_metrics.py       (called inside metrics pipeline)
              src/visualizer.py       → data/cache/<id>_overlay.mp4
              src/explainer.py        → data/cache/<id>_explanation_<lang>.json
                                          ↓
                    web/ (React, polling /api/jobs/<id>)
                    app/streamlit_app.py (fallback for cached clips)
```

Pipeline modules:

- `src/tracking.py` — football-specific YOLOv8 (`football_players.pt`) +
  custom IoU centroid tracker (no ByteTrack DLL). Uses **bottom-centre** of
  bbox as player position (foot contact point). Ball detected as class 0,
  interpolated across short gaps (≤30 frames). Team field left null here.
- `scripts/assign_teams.py` — two-pass jersey colour clustering. Pass 1:
  crop shirt zone (15–50% of bbox height), mask grass pixels in HSV, compute
  median L\*a\*b\*. Fit **DBSCAN** on a\*b\* (no k assumption; falls back to
  k-means k=2). Pass 2: assign every player to nearest team centre — no player
  is left as `team=None` (GK exclusion happens downstream in metrics/visualizer).
- `src/metrics.py` — convex hull area, team centroid, spread std,
  inter-team centroid distance. Excludes the player furthest from team centroid
  as a goalkeeper proxy. Detects spacing-spike events (hull ≥25% change within
  1.5 s, deduped in 1-second windows).
- `src/ball_metrics.py` — possession % and pass stats from ball + player data.
  Key constants: `POSSESSION_THRESHOLD_PX=150`, `MIN_SPELL_DURATION_S=0.2`
  (filters tracker ID-switch noise), `MAX_PASS_GAP_S=3.0`.
- `src/visualizer.py` — convex hull fill + outline per team, centroid dot,
  and **per-player foot-position dots** (4 px circles). GK-proxy player is
  excluded from hull but dot is still drawn.
- `src/explainer.py` — DeepSeek API (`deepseek-reasoner`) via OpenAI-compatible
  endpoint. Returns `headline`, `implication`, `coaching_cue` as JSON.
- `prompts/tactical_explainer.py` — versioned prompt templates with explicit
  banned-jargon list (xG, PPDA, half-spaces, Voronoi, etc.).
- `app/streamlit_app.py` — sidebar clip picker, language toggle, "show AI
  explanation" toggle (for the user-study control condition).
- `study/` — SUS questionnaire (10 standard items + 5 custom items on trust
  and explainability) and a 12-question semi-structured interview guide.

## Course timeline

8-week course, currently in week 5.

- Week 4: end-to-end pipeline shipped (live upload, tracking, team colour, metrics, explainer)
- Week 5 (NOW): metrics accuracy tuned (possession/pass), overlay visual fixes
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

Project uses the **`dsde` conda env** managed by miniforge3/mamba (Python 3.11).
Never use bare `python` / `python3` — on Windows those resolve to the Microsoft
Store stub. Always invoke via the full path:

```
C:\Users\Ittyy\.local\share\mamba\envs\dsde\python.exe
C:\Users\Ittyy\.local\share\mamba\envs\dsde\Scripts\uvicorn.exe
```

PowerShell example (run backend from project root):

```powershell
C:\Users\Ittyy\.local\share\mamba\envs\dsde\Scripts\uvicorn.exe backend.main:app --reload --port 8000
```

Frontend (Vite/React) runs separately on port 5173:

```powershell
cd web && npm run dev
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
- Expanding pass/possession metrics beyond what `src/ball_metrics.py` already
  does (shot detection, possession chains, heat maps, etc.)
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
