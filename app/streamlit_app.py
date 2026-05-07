"""Streamlit demo UI for the Wizard-of-Oz user study.

Loads pre-computed video, metrics, and explanation from data/cache/,
shows them side-by-side. No live YOLO or LLM calls during the study.

Run:
    streamlit run app/streamlit_app.py

Layout:
    [hero block: title + clip metadata pills]
    [video player + overlay]      [AI explanation card: headline / implication / cue]
    [summary metric cards]
    [team spread chart with event markers]
    [event list as colour-coded rows]

Visual language follows the "Grassroots Tactics AI" Claude design (dark
navy background, neon-green accent, Rajdhani+Inter typography). The
restyle is cosmetic only — every metric, threshold, and bilingual string
stays identical to the prior version. No new tactical concepts.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CACHE_DIR

st.set_page_config(page_title="Grassroots Tactics AI", layout="wide",
                   initial_sidebar_state="expanded")

# ── Design tokens ───────────────────────────────────────────────────────────
# Centralised so the matplotlib chart and the CSS-rendered cards agree on
# the same palette. Pulled from the Claude-design React prototype
# (app.jsx C constant). Don't change values here without changing the
# matching .streamlit/config.toml.
GT_BG          = "#0D1B2A"
GT_CARD        = "#1A2B3C"
GT_BORDER      = "#243B52"
GT_GREEN       = "#39FF14"
GT_GRAY        = "#7A9BB5"
GT_GRAY_LIGHT  = "#B0C7D9"
GT_TEAM_A      = "#f5a623"   # orange — must match overlay video
GT_TEAM_B      = "#2196f3"   # blue   — must match overlay video

# ── Inject CSS (Google Fonts + design language) ─────────────────────────────
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"], .stApp, .main, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', sans-serif;
  }}
  h1, h2, h3, h4, h5,
  [data-testid="stMarkdownContainer"] h1,
  [data-testid="stMarkdownContainer"] h2,
  [data-testid="stMarkdownContainer"] h3 {{
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 0.02em;
    font-weight: 700;
  }}
  /* Slim default block padding so the hero sits closer to the top */
  .block-container {{ padding-top: 2rem; }}

  /* Sidebar tightening */
  [data-testid="stSidebar"] {{
    background: {GT_CARD};
    border-right: 1px solid {GT_BORDER};
  }}
  [data-testid="stSidebar"] .stRadio > label,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
    color: {GT_GRAY} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}

  /* Metric cards (st.metric) — restyled to match the design's StatCard */
  [data-testid="stMetric"] {{
    background: {GT_CARD};
    border: 1px solid {GT_BORDER};
    border-radius: 12px;
    padding: 16px 20px;
  }}
  [data-testid="stMetricLabel"] {{
    color: {GT_GRAY} !important;
  }}
  [data-testid="stMetricLabel"] p {{
    font-size: 11px !important;
    color: {GT_GRAY} !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600 !important;
  }}
  [data-testid="stMetricValue"] {{
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 32px !important;
    color: {GT_GREEN} !important;
    line-height: 1.1 !important;
  }}

  /* Bordered video legend caption */
  [data-testid="stCaptionContainer"] p,
  small {{ color: {GT_GRAY} !important; }}

  /* Dividers */
  hr {{ border-color: {GT_BORDER} !important; opacity: 0.5; }}

  /* Custom scrollbar */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-thumb {{ background: {GT_BORDER}; border-radius: 3px; }}

  /* ── Hero ─────────────────────────────────────────────────────────── */
  .gt-hero {{
    margin: 0 0 24px;
    padding: 20px 0 4px;
  }}
  .gt-hero-eyebrow {{
    display: flex; align-items: center; gap: 8px;
    font-size: 12px;
    color: {GT_GREEN};
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }}
  .gt-pulse-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: {GT_GREEN};
    box-shadow: 0 0 8px {GT_GREEN};
    animation: gt-pulse 2s infinite;
  }}
  @keyframes gt-pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%      {{ opacity: 0.4; transform: scale(0.85); }}
  }}
  .gt-hero h1 {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 44px;
    font-weight: 700;
    margin: 0 0 12px;
    line-height: 1.05;
  }}
  .gt-hero h1 .accent {{ color: {GT_GREEN}; }}
  .gt-pill-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .gt-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    border: 1px solid {GT_BORDER};
    background: rgba(26,43,60,0.6);
    color: {GT_GRAY_LIGHT};
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }}
  .gt-pill .gt-pill-key {{
    color: {GT_GRAY};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 10px;
  }}

  /* ── AI explanation card ─────────────────────────────────────────── */
  .gt-ai-card {{
    background: {GT_CARD};
    border: 1px solid {GT_BORDER};
    border-radius: 12px;
    padding: 22px;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }}
  .gt-ai-eyebrow {{
    display: flex; align-items: center; gap: 8px;
    font-size: 11px;
    color: {GT_GREEN};
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }}
  .gt-ai-eyebrow .gt-pulse-dot {{ width: 6px; height: 6px; box-shadow: none; }}
  .gt-ai-headline {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 19px;
    font-weight: 600;
    color: #FFFFFF;
    line-height: 1.35;
  }}
  .gt-ai-implication {{
    font-size: 14px;
    color: {GT_GRAY_LIGHT};
    line-height: 1.6;
    font-style: italic;
  }}
  .gt-ai-cue {{
    background: rgba(57,255,20,0.08);
    border-left: 3px solid {GT_GREEN};
    padding: 12px 14px;
    border-radius: 8px;
    font-size: 13px;
    color: {GT_GRAY_LIGHT};
    line-height: 1.55;
  }}
  .gt-ai-cue .gt-cue-label {{
    display: block;
    color: {GT_GREEN};
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }}
  .gt-ai-credit {{
    margin-top: auto;
    padding-top: 8px;
    border-top: 1px solid {GT_BORDER};
    font-size: 11px;
    color: {GT_GRAY};
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }}
  .gt-ai-empty {{
    background: {GT_CARD};
    border: 1px dashed {GT_BORDER};
    border-radius: 12px;
    padding: 22px;
    color: {GT_GRAY_LIGHT};
    font-size: 14px;
    line-height: 1.6;
  }}
  .gt-ai-empty .gt-ai-empty-title {{
    color: {GT_GREEN};
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  /* ── Section heading ─────────────────────────────────────────────── */
  .gt-section-h {{
    display: flex; align-items: center; gap: 10px;
    margin: 24px 0 14px;
  }}
  .gt-section-h h3 {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 22px;
    font-weight: 700;
    margin: 0;
  }}
  .gt-section-h .gt-section-sub {{
    color: {GT_GRAY};
    font-size: 13px;
    margin-left: auto;
  }}

  /* ── Event row list ──────────────────────────────────────────────── */
  .gt-event-list {{ display: flex; flex-direction: column; gap: 8px; }}
  .gt-event-row {{
    display: flex; align-items: center; gap: 14px;
    background: {GT_CARD};
    border: 1px solid {GT_BORDER};
    border-left: 3px solid var(--gt-row-accent);
    border-radius: 10px;
    padding: 12px 16px;
    transition: border-color 0.15s;
  }}
  .gt-event-row:hover {{ border-color: var(--gt-row-accent); }}
  .gt-event-time {{
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-weight: 600;
    font-size: 13px;
    color: var(--gt-row-accent);
    min-width: 64px;
  }}
  .gt-event-icon {{ font-size: 16px; flex-shrink: 0; }}
  .gt-event-body {{ flex: 1; min-width: 0; font-size: 13px; color: {GT_GRAY_LIGHT}; line-height: 1.5; }}
  .gt-event-body strong {{ color: #FFFFFF; }}
  .gt-event-delta {{
    flex-shrink: 0;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 20px;
    background: var(--gt-row-accent-bg);
    color: var(--gt-row-accent);
    letter-spacing: 0.04em;
  }}
  .gt-no-events {{
    background: {GT_CARD};
    border: 1px dashed {GT_BORDER};
    border-radius: 10px;
    padding: 16px;
    color: {GT_GRAY};
    font-size: 13px;
    text-align: center;
  }}

  .gt-unit-note {{
    color: {GT_GRAY};
    font-size: 12px;
    margin: 8px 0 4px;
    line-height: 1.5;
  }}
</style>
""", unsafe_allow_html=True)

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


st.sidebar.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
      <span class="gt-pulse-dot" style="width:8px;height:8px;border-radius:50%;background:{GT_GREEN};display:inline-block"></span>
      <span style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:18px;letter-spacing:0.05em">
        GRASSROOTS <span style="color:{GT_GREEN}">TACTICS</span> AI
      </span>
    </div>
    <div style="color:{GT_GRAY};font-size:12px;margin-bottom:14px">
      AI-assisted tactical analysis for grassroots football
    </div>
    """,
    unsafe_allow_html=True,
)

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

# Facilitator-only section — collapse before handing device to participant.
# Sidebar stays in English on purpose: it's for the researcher running the
# study, not the participant being studied. Keeps the operator's interface
# stable across sessions even when the participant view switches language.
with st.sidebar.expander("⚙️ Study settings", expanded=False):
    show_explanation = st.toggle("Show AI explanation", value=True,
                                 help="Turn off for the control condition")

# ── Bilingual labels ─────────────────────────────────────────────────────────
# Every participant-facing string lives in this dict so we can swap languages
# without touching layout code. Keep this dict the single source of truth —
# don't sprinkle hard-coded English elsewhere.

T = {
    "en": {
        "hero_eyebrow": "AI Tactical Spacing Analysis",
        "video_legend": (
            "Orange shape = Team A playing area · "
            "Blue shape = Team B playing area · Dot = team centre of gravity"
        ),
        "ai_subheader": "What the AI sees",
        "ai_no_explanation": "No explanation file found for this clip + language.",
        "ai_hidden_title": "Control condition",
        "ai_hidden": (
            "AI analysis is hidden. Watch the clip and share your own "
            "observations first."
        ),
        "coaching_tip": "Coaching cue",
        "snapshot": "Match snapshot",
        "team_a_spread": "Team A avg spread",
        "team_a_help": (
            "Average area of Team A's playing shape across the clip. "
            "Larger = more stretched out."
        ),
        "team_b_spread": "Team B avg spread",
        "team_b_help": "Average area of Team B's playing shape across the clip.",
        "gap": "Avg gap between teams",
        "gap_help": "Average distance between the two teams' centres of gravity.",
        "events_count": "Shape-change events",
        "events_help": (
            "Number of times a team's playing shape changed suddenly "
            "(≥25% in 1.5 s)."
        ),
        "unit_note": (
            "Areas are in k px² (thousand pixels squared) — pixel-space, not "
            "metres. Compare clips relatively, not absolutely."
        ),
        "chart_title": "Team spread over time",
        "chart_caption": (
            "Higher = team is more spread out. Vertical dashed lines mark "
            "the moments listed below."
        ),
        "chart_x": "Time (s)",
        "chart_y": "Hull area (px²)",
        "chart_a": "Team A spread",
        "chart_b": "Team B spread",
        "events_title": "Key moments",
        "events_sub": "Sudden ≥25% shape changes within 1.5 s",
        "no_events": "No sudden shape changes detected in this clip.",
        "team_a_label": "Team A",
        "team_b_label": "Team B",
        "stretch_verb": "spread out",
        "compress_verb": "compressed",
        "stretch_note": "creating space — opponents can exploit the gaps",
        "compress_note": "defending tight — harder for opponents to play through",
        "poss_title": "Possession & Passing",
        "poss_a": "Team A possession",
        "poss_b": "Team B possession",
        "pass_ct_a": "Team A passes",
        "pass_ct_b": "Team B passes",
        "pass_acc_a": "Team A pass acc.",
        "pass_acc_b": "Team B pass acc.",
    },
    "th": {
        "hero_eyebrow": "AI วิเคราะห์การยืนตำแหน่งของทีม",
        "video_legend": (
            "รูปสีส้ม = พื้นที่เล่นทีม A · "
            "รูปสีฟ้า = พื้นที่เล่นทีม B · จุด = ศูนย์กลางของทีม"
        ),
        "ai_subheader": "AI วิเคราะห์ว่า",
        "ai_no_explanation": "ไม่พบไฟล์คำอธิบายสำหรับคลิปและภาษานี้",
        "ai_hidden_title": "เงื่อนไขควบคุม",
        "ai_hidden": (
            "ปิดการแสดงผล AI อยู่ — ลองดูคลิปแล้วบอกสิ่งที่คุณเห็นเองก่อน"
        ),
        "coaching_tip": "คำแนะนำสำหรับโค้ช",
        "snapshot": "ภาพรวมการแข่ง",
        "team_a_spread": "พื้นที่เฉลี่ยทีม A",
        "team_a_help": (
            "ขนาดพื้นที่เฉลี่ยที่ทีม A ครอบครองในคลิป "
            "ค่ามาก = ทีมยืนกระจายตัวมาก"
        ),
        "team_b_spread": "พื้นที่เฉลี่ยทีม B",
        "team_b_help": "ขนาดพื้นที่เฉลี่ยที่ทีม B ครอบครองในคลิป",
        "gap": "ระยะเฉลี่ยระหว่างสองทีม",
        "gap_help": "ระยะห่างเฉลี่ยระหว่างศูนย์กลางของทั้งสองทีม",
        "events_count": "จังหวะที่รูปแบบเปลี่ยน",
        "events_help": (
            "จำนวนครั้งที่รูปแบบของทีมเปลี่ยนแปลงอย่างรวดเร็ว "
            "(≥25% ภายใน 1.5 วินาที)"
        ),
        "unit_note": (
            "พื้นที่แสดงเป็นหน่วย k px² (พันพิกเซล²) — เป็นค่าในระนาบภาพ ไม่ใช่เมตร "
            "ใช้เปรียบเทียบระหว่างคลิปกัน ไม่ใช่ดูค่าสัมบูรณ์"
        ),
        "chart_title": "การเปลี่ยนแปลงพื้นที่ตามเวลา",
        "chart_caption": (
            "ค่ามาก = ทีมยืนกระจายตัว · "
            "เส้นประแนวตั้ง = จังหวะสำคัญที่อธิบายไว้ด้านล่าง"
        ),
        "chart_x": "เวลา (วินาที)",
        "chart_y": "พื้นที่ (px²)",
        "chart_a": "พื้นที่ทีม A",
        "chart_b": "พื้นที่ทีม B",
        "events_title": "ช่วงสำคัญ",
        "events_sub": "การเปลี่ยนแปลงรูปแบบ ≥25% ภายใน 1.5 วินาที",
        "no_events": "ไม่พบการเปลี่ยนแปลงรูปแบบอย่างชัดเจนในคลิปนี้",
        "team_a_label": "ทีม A",
        "team_b_label": "ทีม B",
        "stretch_verb": "ยืนกระจายตัว",
        "compress_verb": "บีบเข้ามา",
        "stretch_note": "เปิดช่องว่าง — คู่แข่งใช้ประโยชน์จากช่องว่างได้",
        "compress_note": "ป้องกันแน่น — คู่แข่งเล่นผ่านได้ยาก",
        "poss_title": "การครองบอล & การส่งบอล",
        "poss_a": "ครองบอลทีม A",
        "poss_b": "ครองบอลทีม B",
        "pass_ct_a": "ส่งบอลทีม A",
        "pass_ct_b": "ส่งบอลทีม B",
        "pass_acc_a": "ความแม่นทีม A",
        "pass_acc_b": "ความแม่นทีม B",
    },
}
t = T[lang]

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

# Defensive sort — metrics.py emits events in chronological order today, but
# the chart's vertical event markers and the "Key moments" list below both
# read narrative-y when ordered by time, so guarantee it here.
events  = sorted(metrics.get("events", []), key=lambda e: e["t"])

# ── Hero ─────────────────────────────────────────────────────────────────────
# The hero replaces st.title — pulse-dot eyebrow + branded headline + a
# pill row showing the actual clip metadata. Pills are read-only display
# (not interactive); the clip *picker* stays in the sidebar.

st.markdown(
    f"""
    <div class="gt-hero">
      <div class="gt-hero-eyebrow">
        <span class="gt-pulse-dot"></span>
        <span>{html.escape(t['hero_eyebrow'])}</span>
      </div>
      <h1>Grassroots <span class="accent">Tactics</span> AI</h1>
      <div class="gt-pill-row">
        <span class="gt-pill"><span class="gt-pill-key">CLIP</span> {html.escape(clip_id)}</span>
        <span class="gt-pill"><span class="gt-pill-key">DURATION</span> {metrics['duration_s']:.1f} s</span>
        <span class="gt-pill"><span class="gt-pill-key">FPS</span> {metrics['fps']:.0f}</span>
        <span class="gt-pill"><span class="gt-pill-key">EVENTS</span> {len(events)}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Top row: video + explanation card ────────────────────────────────────────

video_col, panel_col = st.columns([3, 2])

with video_col:
    st.video(str(overlay_path))
    st.caption(t["video_legend"])

with panel_col:
    if show_explanation and explanation:
        # Single-blob HTML so Streamlit doesn't wrap each piece in its own
        # stMarkdown div and break the card boundary. html.escape on every
        # AI-generated string in case Gemini ever outputs an angle bracket.
        st.markdown(
            f"""
            <div class="gt-ai-card">
              <div class="gt-ai-eyebrow">
                <span class="gt-pulse-dot"></span>
                <span>{html.escape(t['ai_subheader'])}</span>
              </div>
              <div class="gt-ai-headline">{html.escape(explanation['headline'])}</div>
              <div class="gt-ai-implication">{html.escape(explanation['implication'])}</div>
              <div class="gt-ai-cue">
                <span class="gt-cue-label">{html.escape(t['coaching_tip'])}</span>
                {html.escape(explanation['coaching_cue'])}
              </div>
              <div class="gt-ai-credit">
                model: {html.escape(str(explanation.get('model', 'unknown')))}
                · prompt v{html.escape(str(explanation.get('prompt_version', '?')))}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif show_explanation:
        st.markdown(
            f"""
            <div class="gt-ai-empty">
              <div class="gt-ai-empty-title">⚠ {html.escape(t['ai_subheader'])}</div>
              {html.escape(t['ai_no_explanation'])}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="gt-ai-empty">
              <div class="gt-ai-empty-title">{html.escape(t['ai_hidden_title'])}</div>
              {html.escape(t['ai_hidden'])}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Summary metric cards ─────────────────────────────────────────────────────

st.markdown(
    f"""<div class="gt-section-h"><h3>{html.escape(t['snapshot'])}</h3></div>""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

a_mean = team_a.get("hull_area", {}).get("mean", 0)
b_mean = team_b.get("hull_area", {}).get("mean", 0)
cd_mean = cdist.get("mean", 0)

m1.metric(t["team_a_spread"], f"{a_mean / 1_000:.0f} k px²", help=t["team_a_help"])
m2.metric(t["team_b_spread"], f"{b_mean / 1_000:.0f} k px²", help=t["team_b_help"])
m3.metric(t["gap"], f"{cd_mean:.0f} px", help=t["gap_help"])
m4.metric(t["events_count"], len(events), help=t["events_help"])

# Possession & pass metrics — only shown when ball tracking produced data
ball_m  = metrics.get("ball_metrics") or {}
poss    = ball_m.get("possession_pct") or {}
passes  = ball_m.get("pass_count") or {}
acc     = ball_m.get("pass_accuracy") or {}

if poss:
    st.markdown(
        f"""<div class="gt-section-h"><h3>{html.escape(t['poss_title'])}</h3></div>""",
        unsafe_allow_html=True,
    )
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric(t["poss_a"],    f"{poss.get('A',  0):.0f}%")
    p2.metric(t["poss_b"],    f"{poss.get('B',  0):.0f}%")
    p3.metric(t["pass_ct_a"], str(passes.get("A", "–")))
    p4.metric(t["pass_ct_b"], str(passes.get("B", "–")))
    p5.metric(t["pass_acc_a"], f"{acc['A']*100:.0f}%" if acc.get("A") is not None else "–")
    p6.metric(t["pass_acc_b"], f"{acc['B']*100:.0f}%" if acc.get("B") is not None else "–")

# Coaches will ask "what's a px²?". Spelling it out once below the cards is
# cheaper than discovering this in user-study session 3.
st.markdown(
    f"""<div class="gt-unit-note">{html.escape(t['unit_note'])}</div>""",
    unsafe_allow_html=True,
)

# ── Chart ────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="gt-section-h">
      <h3>{html.escape(t['chart_title'])}</h3>
      <span class="gt-section-sub">{html.escape(t['chart_caption'])}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

df_rows = []
for f in metrics["per_frame"]:
    row = {"Time (s)": f["t"]}
    if "team_A" in f:
        row[t["chart_a"]] = f["team_A"]["hull_area"]
    if "team_B" in f:
        row[t["chart_b"]] = f["team_B"]["hull_area"]
    df_rows.append(row)

df = pd.DataFrame(df_rows).set_index("Time (s)")

# Rendered with matplotlib instead of st.line_chart because st.line_chart
# routes through PyArrow, whose native DLL is blocked by Windows Application
# Control on this machine.
import matplotlib.pyplot as plt

# Default DejaVu Sans has no Thai glyphs, so Thai labels render as boxes.
# Tahoma ships with Windows and covers both Latin and Thai cleanly; keep
# DejaVu Sans as the fallback for non-Windows machines (e.g. Colab/Linux).
plt.rcParams["font.family"] = ["Tahoma", "DejaVu Sans"]

fig, ax = plt.subplots(figsize=(10, 2.8))
# Match the design's card surface — figure and axes both on GT_CARD so the
# chart sits flush inside the page background without a white slab around it.
fig.patch.set_facecolor(GT_CARD)
ax.set_facecolor(GT_CARD)

if t["chart_a"] in df.columns:
    ax.plot(df.index, df[t["chart_a"]], color=GT_TEAM_A, label=t["chart_a"], linewidth=1.6)
if t["chart_b"] in df.columns:
    ax.plot(df.index, df[t["chart_b"]], color=GT_TEAM_B, label=t["chart_b"], linewidth=1.6)

# Vertical event markers — colour-matched to the team that triggered them.
# Lets a participant visually trace each "Key moment" entry below back to the
# spike on the chart without doing time-arithmetic in their head. Drawn first
# (well, after the lines) with low alpha so they read as background hints
# rather than competing with the data.
for ev in events:
    ev_color = GT_TEAM_A if ev["team"] == "team_A" else GT_TEAM_B
    ax.axvline(x=ev["t"], color=ev_color, linestyle="--", alpha=0.45, linewidth=1)

ax.set_xlabel(t["chart_x"], color=GT_GRAY_LIGHT, fontsize=10)
ax.set_ylabel(t["chart_y"], color=GT_GRAY_LIGHT, fontsize=10)
ax.tick_params(colors=GT_GRAY, labelsize=9)
for spine in ax.spines.values():
    spine.set_color(GT_BORDER)
ax.grid(True, alpha=0.18, color=GT_BORDER, linewidth=0.6)
legend = ax.legend(loc="upper right", facecolor=GT_BG, edgecolor=GT_BORDER,
                   labelcolor=GT_GRAY_LIGHT, fontsize=9)
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

# ── Events ───────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="gt-section-h">
      <h3>{html.escape(t['events_title'])}</h3>
      <span class="gt-section-sub">{html.escape(t['events_sub'])}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if events:
    rows_html = ['<div class="gt-event-list">']
    for ev in events:
        team_label = t["team_a_label"] if ev["team"] == "team_A" else t["team_b_label"]
        accent     = GT_TEAM_A         if ev["team"] == "team_A" else GT_TEAM_B
        accent_bg  = "rgba(245,166,35,0.12)" if ev["team"] == "team_A" else "rgba(33,150,243,0.12)"
        if ev["type"] == "stretch":
            icon, verb, note = "📐", t["stretch_verb"], t["stretch_note"]
        else:
            icon, verb, note = "🔒", t["compress_verb"], t["compress_note"]
        rows_html.append(
            f"""
            <div class="gt-event-row" style="--gt-row-accent:{accent};--gt-row-accent-bg:{accent_bg}">
              <span class="gt-event-time">{ev['t']:.1f}s</span>
              <span class="gt-event-icon">{icon}</span>
              <div class="gt-event-body">
                <strong>{html.escape(team_label)}</strong> {html.escape(verb)} ·
                {html.escape(note)}
              </div>
              <span class="gt-event-delta">{ev['delta_pct']:+.0f}%</span>
            </div>
            """
        )
    rows_html.append("</div>")
    st.markdown("".join(rows_html), unsafe_allow_html=True)
else:
    st.markdown(
        f"""<div class="gt-no-events">{html.escape(t['no_events'])}</div>""",
        unsafe_allow_html=True,
    )
