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

4. **LLM backend is DeepSeek (`deepseek-reasoner`) via OpenAI-compatible endpoint.**
   Uses the `openai` SDK pointed at DeepSeek's API. JSON output via
   `response_format={"type": "json_object"}`. API key in `.env` as
   `DEEPSEEK_API_KEY`. Do not switch to Gemini or Claude without being asked.

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
  bbox as player position (foot contact point). Saves `cls` field per player
  (1=GK, 2=outfield) by accumulating per-track class votes across frames;
  **`dominant_cls=1` when ≥5 frames AND >50% of votes were class 1** (raised
  from the original >40% / no minimum — prevents short-track false GK labels).
  Ball detected as class 0, interpolated across short gaps (≤30 frames).
  Team field left null here.
- `scripts/assign_teams.py` — two-pass jersey colour clustering. Pass 1:
  crop shirt zone (15–50% of bbox height), mask grass pixels in HSV, compute
  median L\*a\*b\*. Fit **DBSCAN** on a\*b\* (no k assumption; falls back to
  k-means k=2). When the k-means a\*b\* inter-cluster distance is below
  `MIN_INTER_DIST=12`, retries in full 3-D L\*a\*b\* space (handles dark-navy
  vs white kits where hue alone fails). Pass 2: assign every player to nearest
  team centre in 3-D L\*a\*b\*. Two filters before assignment:
  (a) **y-boundary filter** (`MAX_PLAYER_Y_FRAC=0.90`) — skips any detection
  whose bbox bottom edge is below 90% of frame height; catches technical-area
  coaching staff on broadcast tactical shots without needing pitch homography.
  GKs (cls=1) are exempt.
  (b) **outlier filter** (`OUTLIER_THRESH=40.0`) — skips non-GK detections
  whose 3-D L\*a\*b\* distance from both team centres exceeds the threshold;
  catches referees, linesmen, and staff in distinctive colours.
  Players left as `team=None` are excluded from hull/lines downstream.
- `src/metrics.py` — convex hull area, team centroid, spread std,
  inter-team centroid distance. Excludes the player furthest from team centroid
  as a goalkeeper proxy. Detects spacing-spike events (hull ≥25% change within
  1.5 s, deduped in 1-second windows).
- `src/ball_metrics.py` — possession % and pass stats from ball + player data.
  Key constants: `POSSESSION_THRESHOLD_PX=150`, `MIN_SPELL_DURATION_S=0.2`
  (filters tracker ID-switch noise), `MAX_PASS_GAP_S=3.0`.
- `src/visualizer.py` — per-team overlay with: light convex hull fill (13%
  opacity), **pairwise formation lines** between outfield players within 25% of
  max frame dimension, centroid dot. GKs identified via `cls=1` votes are drawn
  with a distinct circle + white cross marker and excluded from hull/lines.
  Smoothing window threshold: **≥3 frames in window AND >50% cls=1** (raised
  from >40% to match tracking.py). Legacy fallback (no cls field): drops
  furthest-from-centroid player as GK proxy. Temporal smoothing window
  (SMOOTH_W × stride frames) prevents hull flicker when a player is briefly
  occluded.
- `src/explainer.py` — DeepSeek API (`deepseek-reasoner`) via OpenAI-compatible
  endpoint. Returns `headline`, `implication`, `coaching_cue` as JSON.
- `prompts/tactical_explainer.py` — versioned prompt templates with explicit
  banned-jargon list (xG, PPDA, half-spaces, Voronoi, etc.).
- `scripts/find_active_play.py` — standalone motion scorer. Frame-difference
  (`cv2.absdiff`) at stride=4 on 320×180 frames; sliding 20-second window
  scored as `mean − 0.4·std + 0.3·weakest_chunk`. Returns best start time.
- `scripts/download_active_clips.py` — downloads 90-second sections from UCL
  matches via yt-dlp, runs motion analysis, trims to best 20-second active-play
  window with ffmpeg. Output: `C:\Users\TSURUGI\Desktop\football_clips\`.
  5 clips already downloaded (liverpool_vs_psg, benfica_vs_realmadrid,
  psg_vs_newcastle, psg_vs_arsenal, psg_vs_intermilan).
- `scripts/fix_tracking_cls.py` — one-time utility; retroactively re-applies
  the new GK threshold (≥5 frames, >50%) to existing tracking JSONs without
  re-running YOLO inference. Already applied to all 5 UCL clip JSONs.
- `scripts/rerun_pipeline.py` — re-runs assign_teams → metrics → visualizer
  for all (or one specific) job IDs. Updates data/jobs/<id>.json summary so
  the web UI reflects new numbers. Useful after code fixes without re-tracking.
- `app/streamlit_app.py` — sidebar clip picker, language toggle, "show AI
  explanation" toggle (for the user-study control condition).
- `study/` — SUS questionnaire (10 standard items + 5 custom items on trust
  and explainability) and a 12-question semi-structured interview guide.

## Week 5 fixes applied (2026-05-08)

Four bugs fixed during frame-by-frame overlay audit across all 5 UCL clips:

1. **Outlier filter was a no-op** — `assign_teams.py` had `pass` instead of
   `continue` in the outlier check, so referees/linesmen in distinctive
   colours were being assigned to teams anyway. Fixed to `continue`.

2. **Team assignment was 2-D instead of 3-D** — pass 2 used a\*b\* distance
   to assign teams, but the cluster centres were computed in 3-D L\*a\*b\*.
   Mismatched dimensions silently degraded assignment quality. Fixed to use
   full 3-D distance consistently.

3. **GK threshold too loose** — tracking.py used >40% with no frame minimum;
   players briefly misclassified as GK got false markers. Raised to ≥5 frames
   AND >50%. Visualizer's smoothing window threshold raised to match.

4. **Y-boundary filter missing** — coaching staff in team-coloured clothing
   passed the outlier filter and pulled the convex hull toward the touchline.
   Added `MAX_PLAYER_Y_FRAC=0.90` in assign_teams.py to silently exclude
   detections in the bottom 10% of frame height (technical area strip).

All fixes applied, all 5 clips re-processed. Tracking JSONs retroactively
patched by `scripts/fix_tracking_cls.py`.

## Course timeline

8-week course, currently in week 5 (end).

- Week 4: end-to-end pipeline shipped (live upload, tracking, team colour, metrics, explainer)
- Week 5: overlay visual fixes — outlier filter, 3-D L*a*b* assignment, GK threshold, y-boundary filter. All 5 UCL clips re-processed. **DONE.**
- Week 6: user study with 5–8 grassroots-football participants
- Week 7: full paper draft (≥24 pages, ACM one-column format) submitted
- Week 8: final paper + presentation + live demo

## Data plan

- **5 UCL tactical-view clips** (20 s each, active play only) — downloaded to
  `C:\Users\TSURUGI\Desktop\football_clips\` via `scripts/download_active_clips.py`.
  Auto-trimmed by motion analysis to avoid dead balls / set pieces. Use for
  technical validation and demo. All 5 processed and cached in `data/`.
  Clip-to-job-ID mapping:
  - `benfica_vs_realmadrid_active.mp4` → `6a532b15a9ba`
  - `psg_vs_intermilan_active.mp4`     → `28986de4ebe1`
  - `psg_vs_arsenal_active.mp4`        → `895cb788c736`
  - `liverpool_vs_psg_active.mp4`      → `cecfb434ae74`
  - `psg_vs_newcastle_active.mp4`      → `dae7666f479d`
- 1 **grassroots clip** (YouTube amateur/Sunday-league/youth-football) for the
  user study. Lower quality but matches the target deployment context. This
  is the primary "we built it for grassroots" evidence in the paper.
  **Not yet downloaded — needed before week 6 user study.**

## Python environment

Project uses a `.venv` virtualenv (Python 3.11) in the project root.
Never use bare `python` / `python3` — on Windows those resolve to the Microsoft
Store stub. Always invoke via the venv path:

```
.venv\Scripts\python.exe
.venv\Scripts\uvicorn.exe
```

PowerShell example (run backend from project root):

```powershell
.venv\Scripts\uvicorn.exe backend.main:app --reload --port 8000
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
