"""Streamlit demo UI for the Wizard-of-Oz user study.

Loads pre-computed video, metrics, and explanation from data/cache/,
shows them side-by-side. No live YOLO or LLM calls during the study.

Run:
    streamlit run app/streamlit_app.py

Layout:
    [video player + overlay]      [AI explanation: headline / implication / cue]
    [summary metric cards]
    [team spread chart with event markers]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CACHE_DIR

st.set_page_config(page_title="Grassroots Tactics AI", layout="wide")

# ── Sidebar ─────────────────────────────────────────────────────────────────

def list_clips() -> list[str]:
    """Find clips that have all three artefacts ready."""
    metrics     = {p.stem.replace("_metrics", "")      for p in CACHE_DIR.glob("*_metrics.json")}
    overlays    = {p.stem.replace("_overlay", "")      for p in CACHE_DIR.glob("*_overlay.mp4")}
    explanations = {
        p.stem.replace("_explanation_en", "").replace("_explanation_th", "")
        for p in CACHE_DIR.glob("*_explanation_*.json")
    }
    return sorted(metrics & overlays & explanations)


st.sidebar.title("Grassroots Tactics AI")
st.sidebar.caption("AI-assisted tactical analysis for grassroots football")

clip_ids = list_clips()
if not clip_ids:
    st.sidebar.warning("No clips found in data/cache/")
    st.info(
        "Prepare a clip first:\n\n"
        "```bash\n"
        "python -m src.tracking   --input data/clips/yourclip.mp4\n"
        "python scripts/assign_teams.py --tracking data/tracking/yourclip.json "
        "--video data/clips/yourclip.mp4\n"
        "python -m src.metrics    --input data/tracking/yourclip.json\n"
        "python -m src.visualizer --video data/clips/yourclip.mp4 "
        "--tracking data/tracking/yourclip.json\n"
        "python -m src.explainer  --input data/cache/yourclip_metrics.json\n"
        "```"
    )
    st.stop()

clip_id = st.sidebar.selectbox("Clip", clip_ids)
lang = st.sidebar.radio("Language", ["en", "th"], horizontal=True,
                        format_func=lambda x: "English" if x == "en" else "ภาษาไทย")

st.sidebar.divider()

# Facilitator-only section — collapse before handing device to participant
with st.sidebar.expander("⚙️ Study settings", expanded=False):
    show_explanation = st.toggle("Show AI explanation", value=True,
                                 help="Turn off for the control condition")

# ── Load artefacts ───────────────────────────────────────────────────────────

metrics_path     = CACHE_DIR / f"{clip_id}_metrics.json"
overlay_path     = CACHE_DIR / f"{clip_id}_overlay.mp4"
explanation_path = CACHE_DIR / f"{clip_id}_explanation_{lang}.json"

metrics     = json.loads(metrics_path.read_text(encoding="utf-8"))
explanation = (json.loads(explanation_path.read_text(encoding="utf-8"))
               if explanation_path.exists() else None)

summary = metrics.get("summary", {})
team_a  = summary.get("team_A", {})
team_b  = summary.get("team_B", {})
cdist   = summary.get("centroid_distance", {})
events  = metrics.get("events", [])

# ── Header ───────────────────────────────────────────────────────────────────

st.title("Grassroots Tactics AI")
st.caption(f"Clip: **{clip_id}** — {metrics['duration_s']:.1f} s @ {metrics['fps']:.0f} fps")

# ── Top row: video + explanation ─────────────────────────────────────────────

video_col, panel_col = st.columns([3, 2])

with video_col:
    st.video(str(overlay_path))
    st.caption(
        "Orange shape = Team A playing area · Blue shape = Team B playing area · "
        "Dot = team centre of gravity"
    )

with panel_col:
    if show_explanation and explanation:
        st.subheader("What the AI sees")
        st.markdown(f"**{explanation['headline']}**")
        st.markdown(f"*{explanation['implication']}*")
        st.divider()
        st.success(f"**Coaching tip:** {explanation['coaching_cue']}")
        st.caption(
            f"AI model: `{explanation.get('model', 'unknown')}` · "
            f"prompt v{explanation.get('prompt_version', '?')}"
        )
    elif show_explanation:
        st.warning("No explanation file found for this clip + language.")
    else:
        st.info(
            "AI analysis is hidden.\n\n"
            "Watch the clip and share your own observations first."
        )

# ── Summary metric cards ─────────────────────────────────────────────────────

st.divider()
st.subheader("Match snapshot")

m1, m2, m3, m4 = st.columns(4)

a_mean = team_a.get("hull_area", {}).get("mean", 0)
b_mean = team_b.get("hull_area", {}).get("mean", 0)
cd_mean = cdist.get("mean", 0)

m1.metric(
    "Team A avg spread",
    f"{a_mean / 1_000:.0f} k px²",
    help="Average area of Team A's playing shape across the clip. Larger = more stretched out.",
)
m2.metric(
    "Team B avg spread",
    f"{b_mean / 1_000:.0f} k px²",
    help="Average area of Team B's playing shape across the clip.",
)
m3.metric(
    "Avg gap between teams",
    f"{cd_mean:.0f} px",
    help="Average distance between the two teams' centres of gravity.",
)
m4.metric(
    "Shape-change events",
    len(events),
    help="Number of times a team's playing shape changed suddenly (≥25% in 1.5 s).",
)

# ── Chart ────────────────────────────────────────────────────────────────────

st.subheader("Team spread over time")
st.caption("Higher = team is more spread out. Sudden spikes = shape changed quickly.")

df_rows = []
for f in metrics["per_frame"]:
    row = {"Time (s)": f["t"]}
    if "team_A" in f:
        row["Team A spread"] = f["team_A"]["hull_area"]
    if "team_B" in f:
        row["Team B spread"] = f["team_B"]["hull_area"]
    df_rows.append(row)

df = pd.DataFrame(df_rows).set_index("Time (s)")
st.line_chart(df, height=260, color=["#f5a623", "#2196f3"])

# ── Events ───────────────────────────────────────────────────────────────────

if events:
    st.subheader("Key moments")
    for ev in events:
        team_label = "Team A" if ev["team"] == "team_A" else "Team B"
        delta = ev["delta_pct"]
        if ev["type"] == "stretch":
            icon, verb, note = "📐", "spread out", "creating space — opponents can exploit the gaps"
        else:
            icon, verb, note = "🔒", "compressed", "defending tight — harder for opponents to play through"
        st.write(
            f"{icon} **t = {ev['t']:.1f} s** — {team_label} suddenly **{verb}** "
            f"({delta:+.0f}%) · {note}"
        )
else:
    st.info("No sudden shape changes detected in this clip.")
