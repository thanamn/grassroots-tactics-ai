"""Chat-style Q&A grounded in a finished match's spacing metrics.

Why a separate module from explainer.py
---------------------------------------
The explainer's job is a one-shot, schema-bound JSON output
(headline / implication / coaching_cue). Chat needs free-form prose,
multi-turn history, and a *much* tighter system prompt to keep the
model from drifting into formation/pressing/transitions territory
(scope rule #1 in CLAUDE.md). Mixing the two would mean two prompt
templates fighting over the same explainer surface.

Scope guard
-----------
The system instruction explicitly forbids the banned-jargon list from
prompts/tactical_explainer.py and limits answers to spacing /
compactness only. If the coach asks "what about pressing?" the model
is told to redirect back to spacing. We also keep responses to 2–3
sentences to mirror the explainer's voice.
"""
from __future__ import annotations

import time
from typing import Literal

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.explainer import RETRY_DELAYS, _is_transient

Lang = Literal["en", "th"]


SYSTEM_EN = """You are an AI coaching assistant helping a grassroots football coach
understand their team's SPACING and COMPACTNESS — and ONLY those.

Topics you must NOT discuss: formations (4-3-3, 4-4-2…), pressing,
transitions, set pieces, individual skill, fitness, possession %.
If asked about any of those, gently redirect to what the spacing
metrics show.

Banned jargon: xG, PPDA, half-spaces, Voronoi, expected threat,
progressive passes. Use plain language a volunteer parent-coach
would understand.

Ground every answer in the metrics block below. Reference specific
timestamps (e.g. "around 12.4 s") or numbers when possible. Keep
answers to 2-3 sentences. Be direct and concrete — never hedge with
"it depends" or "more analysis is needed".
"""

SYSTEM_TH = """คุณเป็นผู้ช่วยโค้ช AI ที่ช่วยโค้ชฟุตบอลรากหญ้าเข้าใจ
"การยืนตำแหน่ง" และ "ความแน่น/หลวมของทีม" เท่านั้น

หัวข้อที่ห้ามพูด: แผนการเล่น (4-3-3, 4-4-2…), การเพรส,
การเปลี่ยนเกม, ลูกตั้งเตะ, ทักษะส่วนตัว, ความฟิต, % ครองบอล
ถ้าโค้ชถามเรื่องเหล่านี้ ให้พาเขากลับมาที่เรื่องการยืนตำแหน่ง

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

    # Top-3 events by absolute delta — saves the model from scanning
    # the full events list when the coach asks "biggest moments".
    big = sorted(events, key=lambda e: abs(e["delta_pct"]), reverse=True)[:3]
    big_lines = "\n".join(
        f"  - t={e['t']:.1f}s · {e['team']} · {e['type']} ({e['delta_pct']:+.0f}%)"
        for e in big
    ) or "  (none)"

    if lang == "th":
        return (
            f"ข้อมูลคลิป (อ้างอิงเมื่อตอบ):\n"
            f"  ความยาวคลิป: {duration:.1f} วินาที\n"
            f"  พื้นที่เฉลี่ยทีม A: {a_hull.get('mean', 0)/1000:.0f} k px²\n"
            f"  พื้นที่เฉลี่ยทีม B: {b_hull.get('mean', 0)/1000:.0f} k px²\n"
            f"  ระยะเฉลี่ยระหว่างศูนย์กลางสองทีม: {cd.get('mean', 0):.0f} px\n"
            f"  จังหวะรูปแบบเปลี่ยน: {len(events)} ครั้ง "
            f"(กระจายตัว {stretches}, บีบเข้า {compresses})\n"
            f"  จังหวะที่เปลี่ยนรุนแรงที่สุด:\n{big_lines}\n"
        )
    return (
        f"Clip metrics (refer to these in your answer):\n"
        f"  Duration: {duration:.1f} s\n"
        f"  Team A avg hull area: {a_hull.get('mean', 0)/1000:.0f} k px²\n"
        f"  Team B avg hull area: {b_hull.get('mean', 0)/1000:.0f} k px²\n"
        f"  Avg centroid distance: {cd.get('mean', 0):.0f} px\n"
        f"  Shape-change events: {len(events)} total "
        f"({stretches} stretches, {compresses} compressions)\n"
        f"  Biggest events:\n{big_lines}\n"
    )


def chat(metrics: dict, question: str,
         history: list[dict] | None = None,
         lang: Lang = "en", max_tokens: int = 320) -> str:
    """Run one chat turn against Gemini, return the model's reply text.

    `history` is the full prior conversation as
    `[{"role": "user"|"assistant", "text": "..."}]`. The first turn
    can pass an empty list. We don't persist history server-side; the
    frontend sends it back each time so the user can clear locally.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to .env and restart the server."
        )
    from google import genai
    from google.genai import types

    base_system = SYSTEM_TH if lang == "th" else SYSTEM_EN
    system = base_system + "\n\n" + _summary_block(metrics, lang)

    contents: list = []
    for msg in (history or []):
        role = "user" if msg.get("role") == "user" else "model"
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        contents.append(types.Content(
            role=role, parts=[types.Part.from_text(text=text)],
        ))
    contents.append(types.Content(
        role="user", parts=[types.Part.from_text(text=question)],
    ))

    client = genai.Client(api_key=GEMINI_API_KEY)
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        max_output_tokens=max_tokens,
        temperature=0.6,
    )

    # Same retry pattern as the explainer — Gemini free tier 503's
    # routinely under load, especially on back-to-back calls.
    attempts = len(RETRY_DELAYS) + 1
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=contents, config=cfg,
            )
            return (response.text or "").strip()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if i == attempts - 1 or not _is_transient(e):
                raise
            time.sleep(RETRY_DELAYS[i])
    raise last_exc  # type: ignore[misc]
