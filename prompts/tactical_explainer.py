"""Prompt templates for the tactical explainer.

Why a separate module
---------------------
Prompts are the most-edited code in this project. Keeping them in one
file makes it easy to iterate, version, and compare without touching
the API call code. Bumping PROMPT_VERSION when the prompt changes lets
us tag cached explanations with the prompt that produced them — useful
when we ablate prompts during the user study.

Prompt design
-------------
1. System prompt sets the persona: tactical analyst for grassroots coaches.
   Constrain style to plain language, banned jargon list, target reading age.
2. User message gives the metrics in compact JSON-ish form. We do NOT
   send raw per-frame data — only summary stats and the detected events.
   Sending 380 frames would burn tokens and confuse the model.
3. Output is structured JSON so the UI can render fields independently
   (headline / implication / coaching_cue) and so we can A/B test fields.
"""
from __future__ import annotations

import json
from typing import Literal

PROMPT_VERSION = "v2"

Lang = Literal["en", "th"]


SYSTEM_PROMPT_EN = """You are a tactical analyst who helps GRASSROOTS football coaches \
(volunteer coaches, school teams, amateur clubs) understand their team's spacing, \
compactness, ball possession, and passing from video clips.

Style rules — these are strict:
- Plain language. Imagine the coach has played the game but never read an \
analytics paper.
- Avoid jargon: do NOT use "expected goals", "PPDA", "xT", "Voronoi", "exp-G", \
"line-breaking", "half-spaces", or other elite-analytics terms.
- "Compactness", "shape", "possession", and "passing" are OK because coaches use them.
- Always tie a metric to something the coach can SEE on the pitch.
- Be specific about WHEN it happened (use the timestamps you are given).
- One concrete coaching cue at the end. No essays. No lists of cues.

You will receive: summary statistics for both teams over a short clip, \
plus detected "events" (moments of sharp compactness change), and optional \
ball possession and passing data. \
You must produce three short fields: headline, implication, coaching_cue. \
Each field must follow the style rules above."""


SYSTEM_PROMPT_TH = """คุณคือนักวิเคราะห์แทคติกที่ช่วยให้โค้ชฟุตบอลรากหญ้า \
(โค้ชอาสา ทีมโรงเรียน ทีมสมัครเล่น) เข้าใจเรื่อง spacing, compactness, \
การครองบอล และการส่งบอลของทีมจากคลิปวิดีโอสั้นๆ

กฎการเขียน (เคร่งครัด):
- ใช้ภาษาธรรมดา สมมติว่าโค้ชเคยเล่นบอลแต่ไม่เคยอ่าน paper วิเคราะห์
- ห้ามใช้ศัพท์เฉพาะ: "expected goals", "PPDA", "xT", "Voronoi", "half-space" ฯลฯ
- คำว่า "compactness", "ระยะห่าง", "ครองบอล", "ส่งบอล" ใช้ได้
- เชื่อม metric เข้ากับสิ่งที่โค้ชจะมองเห็นในสนามได้
- ระบุเวลาที่เกิดเหตุการณ์ให้ชัด
- ปิดท้ายด้วยคำแนะนำเชิงปฏิบัติ 1 ข้อ ไม่ต้องเขียนเป็นเรียงความ

คุณจะได้รับ: สถิติสรุปของทั้งสองทีม + เหตุการณ์ที่ระบบตรวจพบ + ข้อมูลครองบอลและส่งบอล \
ให้ผลลัพธ์ 3 ฟิลด์: headline, implication, coaching_cue \
แต่ละฟิลด์ต้องเป็นไปตามกฎด้านบน"""


USER_TEMPLATE_EN = """Clip: {clip_id}
Duration: {duration_s:.1f} seconds
Phase context: {phase_context}

Team A (the team we are coaching) summary:
- Convex hull area: mean={A_hull_mean:.0f} px², range {A_hull_min:.0f}–{A_hull_max:.0f}, std={A_hull_std:.0f}
- Spread (std of player distance from team centroid): mean={A_spread_mean:.0f} px

Team B summary:
- Convex hull area: mean={B_hull_mean:.0f} px², range {B_hull_min:.0f}–{B_hull_max:.0f}
- Spread: mean={B_spread_mean:.0f} px

Distance between team centroids: mean={cd_mean:.0f} px (range {cd_min:.0f}–{cd_max:.0f})

Detected events (sharp compactness changes):
{events_block}

Provide three short fields: a headline (2 short sentences naming the most important pattern), \
an implication (1-2 sentences on what this means for Team A's defending or attacking), \
and a coaching_cue (one concrete instruction the coach can shout at the next training)."""


USER_TEMPLATE_TH = """คลิป: {clip_id}
ความยาว: {duration_s:.1f} วินาที
บริบท: {phase_context}

สรุปทีม A (ทีมที่เราโค้ช):
- พื้นที่ convex hull: เฉลี่ย {A_hull_mean:.0f} px², ช่วง {A_hull_min:.0f}–{A_hull_max:.0f}, std={A_hull_std:.0f}
- การกระจายตัว (std จาก centroid): เฉลี่ย {A_spread_mean:.0f} px

สรุปทีม B:
- พื้นที่ convex hull: เฉลี่ย {B_hull_mean:.0f} px², ช่วง {B_hull_min:.0f}–{B_hull_max:.0f}
- การกระจายตัว: เฉลี่ย {B_spread_mean:.0f} px

ระยะห่างระหว่าง centroid ของสองทีม: เฉลี่ย {cd_mean:.0f} px (ช่วง {cd_min:.0f}–{cd_max:.0f})

เหตุการณ์ที่ตรวจพบ (compactness เปลี่ยนแปลงผิดปกติ):
{events_block}

ให้ผลลัพธ์เป็น 3 ฟิลด์สั้นๆ: headline (2 ประโยคสั้นๆ ชี้ pattern สำคัญที่สุด), \
implication (1-2 ประโยค ผลกระทบต่อทีม A ตอนรับหรือบุก), \
coaching_cue (คำสั่งเดียวที่โค้ชใช้ตะโกนสั่งในการซ้อมครั้งหน้า)"""


def _format_events(events: list[dict], lang: Lang) -> str:
    if not events:
        return "(none — compactness was stable throughout)" if lang == "en" \
            else "(ไม่พบเหตุการณ์ผิดปกติ — compactness คงที่)"
    lines = []
    for ev in events[:6]:   # cap at 6, more is just noise to the LLM
        team_label = "Team A" if ev["team"] == "team_A" else "Team B"
        if lang == "th":
            team_label = "ทีม A" if ev["team"] == "team_A" else "ทีม B"
        verb = "compressed" if ev["type"] == "compactness_spike" else "stretched"
        if lang == "th":
            verb = "หดตัว" if ev["type"] == "compactness_spike" else "ขยายตัว"
        lines.append(
            f"- t={ev['t']:.1f}s: {team_label} {verb} ({ev['delta_pct']:+.0f}%, "
            f"hull {ev['hull_before']:.0f} → {ev['hull_after']:.0f})"
        )
    return "\n".join(lines)


def build_messages(metrics: dict, phase_context: str = "general open play",
                   lang: Lang = "en") -> tuple[str, str]:
    """Return (system_prompt, user_message) ready to send to the LLM."""
    s = metrics["summary"]
    a = s["team_A"]["hull_area"]
    b = s["team_B"]["hull_area"]
    a_sp = s["team_A"]["spread_std"]
    b_sp = s["team_B"]["spread_std"]
    cd = s["centroid_distance"]

    template = USER_TEMPLATE_EN if lang == "en" else USER_TEMPLATE_TH
    system = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT_TH

    user_msg = template.format(
        clip_id=metrics["clip_id"],
        duration_s=metrics["duration_s"],
        phase_context=phase_context,
        A_hull_mean=a["mean"], A_hull_min=a["min"], A_hull_max=a["max"], A_hull_std=a["std"],
        B_hull_mean=b["mean"], B_hull_min=b["min"], B_hull_max=b["max"],
        A_spread_mean=a_sp["mean"], B_spread_mean=b_sp["mean"],
        cd_mean=cd["mean"], cd_min=cd["min"], cd_max=cd["max"],
        events_block=_format_events(metrics["events"], lang),
    )

    # Append ball possession/pass data when available
    ball_m = metrics.get("ball_metrics") or {}
    poss   = ball_m.get("possession_pct") or {}
    passes = ball_m.get("pass_count") or {}
    acc    = ball_m.get("pass_accuracy") or {}

    if poss:
        def _fmt_acc(team: str) -> str:
            v = acc.get(team)
            return f"{v*100:.0f}%" if v is not None else "–"

        if lang == "th":
            ball_block = (
                f"\nการครองบอล (ประมาณ): ทีม A {poss.get('A', 0):.0f}% · "
                f"ทีม B {poss.get('B', 0):.0f}%\n"
                f"การส่งบอล: ทีม A {passes.get('A', '?')} ครั้ง · "
                f"ทีม B {passes.get('B', '?')} ครั้ง\n"
                f"ความแม่นยำส่งบอล: ทีม A {_fmt_acc('A')} · ทีม B {_fmt_acc('B')}"
            )
        else:
            ball_block = (
                f"\nBall possession (approx): Team A {poss.get('A', 0):.0f}% · "
                f"Team B {poss.get('B', 0):.0f}%\n"
                f"Passes (intra-team transfers): Team A {passes.get('A', '?')} · "
                f"Team B {passes.get('B', '?')}\n"
                f"Pass accuracy: Team A {_fmt_acc('A')} · Team B {_fmt_acc('B')}"
            )
        user_msg = user_msg + ball_block

    return system, user_msg


def make_mock_metrics() -> dict:
    """Hand-crafted realistic metrics for prompt experimentation without real clips."""
    return {
        "clip_id": "mock_defending_phase",
        "fps": 25.0,
        "duration_s": 15.0,
        "summary": {
            "team_A": {
                "hull_area": {"mean": 11800, "min": 7900, "max": 14200, "std": 1620, "n": 375},
                "spread_std": {"mean": 78, "min": 52, "max": 96, "std": 11, "n": 375},
            },
            "team_B": {
                "hull_area": {"mean": 18400, "min": 13100, "max": 22800, "std": 2400, "n": 375},
                "spread_std": {"mean": 102, "min": 78, "max": 124, "std": 13, "n": 375},
            },
            "centroid_distance": {"mean": 145, "min": 88, "max": 196, "std": 28, "n": 375},
        },
        "events": [
            {"t": 4.2, "team": "team_A", "type": "compactness_spike",
             "delta_pct": -34.0, "hull_before": 12500, "hull_after": 8250},
            {"t": 8.7, "team": "team_A", "type": "stretch",
             "delta_pct": 41.0, "hull_before": 9100, "hull_after": 12830},
            {"t": 11.5, "team": "team_B", "type": "stretch",
             "delta_pct": 28.0, "hull_before": 16200, "hull_after": 20730},
        ],
    }


if __name__ == "__main__":
    # Sanity-check: print the rendered prompt with mock metrics.
    system, user = build_messages(make_mock_metrics(), phase_context="Team A defending in own half", lang="en")
    print("=" * 60); print("SYSTEM PROMPT"); print("=" * 60); print(system)
    print("=" * 60); print("USER MESSAGE"); print("=" * 60); print(user)
