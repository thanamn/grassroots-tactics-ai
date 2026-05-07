"""Chat-style Q&A grounded in a finished match's tactical metrics.

Why a separate module from explainer.py
---------------------------------------
The explainer's job is a one-shot, schema-bound JSON output
(headline / implication / coaching_cue). Chat needs free-form prose,
multi-turn history, and a tighter system prompt to keep the model on-topic.

Scope guard
-----------
The system instruction limits answers to spacing, compactness, possession,
and passing — the four metrics the software actually computes. If the coach
asks about formations, pressing, or set pieces, redirect to what the data shows.
"""
from __future__ import annotations

import time
from typing import Literal

import openai

from src.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from src.explainer import RETRY_DELAYS, _is_transient

Lang = Literal["en", "th"]


SYSTEM_EN = """You are an AI coaching assistant helping a grassroots football coach
understand their team's SPACING, COMPACTNESS, POSSESSION, and PASSING accuracy
— and ONLY those topics.

Topics you must NOT discuss: formations (4-3-3, 4-4-2…), pressing,
transitions, set pieces, individual skill, fitness.
If asked about any of those, gently redirect to what the metrics show.

Banned jargon: xG, PPDA, half-spaces, Voronoi, expected threat,
progressive passes. Use plain language a volunteer parent-coach
would understand.

Ground every answer in the metrics block below. Reference specific
timestamps (e.g. "around 12.4 s") or numbers when possible. Keep
answers to 2-3 sentences. Be direct and concrete — never hedge with
"it depends" or "more analysis is needed".
"""

SYSTEM_TH = """คุณเป็นผู้ช่วยโค้ช AI ที่ช่วยโค้ชฟุตบอลรากหญ้าเข้าใจ
"การยืนตำแหน่ง" "ความแน่น/หลวมของทีม" "การครองบอล" และ "ความแม่นยำการส่งบอล"
เท่านั้น — ห้ามพูดนอกหัวข้อนี้

หัวข้อที่ห้ามพูด: แผนการเล่น (4-3-3, 4-4-2…), การเพรส,
การเปลี่ยนเกม, ลูกตั้งเตะ, ทักษะส่วนตัว, ความฟิต
ถ้าโค้ชถามเรื่องเหล่านี้ ให้พาเขากลับมาที่ข้อมูลที่มีอยู่

ห้ามใช้ศัพท์เทคนิค: xG, PPDA, half-spaces, Voronoi, expected threat,
progressive passes ใช้ภาษาง่าย ๆ ที่โค้ชอาสาสมัครเข้าใจ

อ้างอิงคำตอบจากตัวเลขในบล็อกข้อมูลด้านล่างเสมอ ระบุช่วงเวลา
(เช่น "ราววินาทีที่ 12.4") หรือตัวเลขเฉพาะเมื่อทำได้
ตอบสั้น 2-3 ประโยค ตรงประเด็น
"""


def _summary_block(metrics: dict, lang: Lang) -> str:
    """Build a compact metrics snapshot to give the chat model context."""
    summary    = metrics.get("summary", {})
    a_hull     = summary.get("team_A", {}).get("hull_area", {})
    b_hull     = summary.get("team_B", {}).get("hull_area", {})
    cd         = summary.get("centroid_distance", {})
    events     = metrics.get("events", [])
    stretches  = sum(1 for e in events if e["type"] == "stretch")
    compresses = sum(1 for e in events if e["type"] == "compactness_spike")
    duration   = metrics.get("duration_s", 0.0)

    ball_m  = metrics.get("ball_metrics") or {}
    poss    = ball_m.get("possession_pct") or {}
    passes  = ball_m.get("pass_count") or {}
    acc     = ball_m.get("pass_accuracy") or {}

    # Top-3 events by absolute delta
    big = sorted(events, key=lambda e: abs(e["delta_pct"]), reverse=True)[:3]
    big_lines = "\n".join(
        f"  - t={e['t']:.1f}s · {e['team']} · {e['type']} ({e['delta_pct']:+.0f}%)"
        for e in big
    ) or "  (none)"

    def _fmt_acc(team: str) -> str:
        v = acc.get(team)
        return f"{v*100:.0f}%" if v is not None else "–"

    if lang == "th":
        ball_block = ""
        if poss:
            ball_block = (
                f"  ครองบอล: ทีม A {poss.get('A', 0):.0f}% · ทีม B {poss.get('B', 0):.0f}%\n"
                f"  ส่งบอล: ทีม A {passes.get('A', '–')} ครั้ง · ทีม B {passes.get('B', '–')} ครั้ง\n"
                f"  ความแม่น: ทีม A {_fmt_acc('A')} · ทีม B {_fmt_acc('B')}\n"
            )
        return (
            f"ข้อมูลคลิป (อ้างอิงเมื่อตอบ):\n"
            f"  ความยาวคลิป: {duration:.1f} วินาที\n"
            f"  พื้นที่เฉลี่ยทีม A: {a_hull.get('mean', 0)/1000:.0f} k px²\n"
            f"  พื้นที่เฉลี่ยทีม B: {b_hull.get('mean', 0)/1000:.0f} k px²\n"
            f"  ระยะเฉลี่ยระหว่างศูนย์กลางสองทีม: {cd.get('mean', 0):.0f} px\n"
            f"{ball_block}"
            f"  จังหวะรูปแบบเปลี่ยน: {len(events)} ครั้ง "
            f"(กระจายตัว {stretches}, บีบเข้า {compresses})\n"
            f"  จังหวะที่เปลี่ยนรุนแรงที่สุด:\n{big_lines}\n"
        )

    ball_block = ""
    if poss:
        ball_block = (
            f"  Possession: Team A {poss.get('A', 0):.0f}% · Team B {poss.get('B', 0):.0f}%\n"
            f"  Passes: Team A {passes.get('A', '–')} · Team B {passes.get('B', '–')}\n"
            f"  Pass accuracy: Team A {_fmt_acc('A')} · Team B {_fmt_acc('B')}\n"
        )
    return (
        f"Clip metrics (refer to these in your answer):\n"
        f"  Duration: {duration:.1f} s\n"
        f"  Team A avg hull area: {a_hull.get('mean', 0)/1000:.0f} k px²\n"
        f"  Team B avg hull area: {b_hull.get('mean', 0)/1000:.0f} k px²\n"
        f"  Avg centroid distance: {cd.get('mean', 0):.0f} px\n"
        f"{ball_block}"
        f"  Shape-change events: {len(events)} total "
        f"({stretches} stretches, {compresses} compressions)\n"
        f"  Biggest events:\n{big_lines}\n"
    )


def chat(metrics: dict, question: str,
         history: list[dict] | None = None,
         lang: Lang = "en", max_tokens: int = 320) -> str:
    """Run one chat turn against DeepSeek, return the model's reply text."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Add it to .env and restart the server."
        )

    base_system = SYSTEM_TH if lang == "th" else SYSTEM_EN
    system = base_system + "\n\n" + _summary_block(metrics, lang)

    messages: list[dict] = [{"role": "system", "content": system}]
    for msg in (history or []):
        role = "user" if msg.get("role") == "user" else "assistant"
        text = (msg.get("text") or "").strip()
        if text:
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": question})

    client = openai.OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    attempts = len(RETRY_DELAYS) + 1
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.6,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if i == attempts - 1 or not _is_transient(e):
                raise
            time.sleep(RETRY_DELAYS[i])
    raise last_exc  # type: ignore[misc]
