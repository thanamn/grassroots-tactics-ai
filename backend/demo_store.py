"""Precomputed demo clips exposed as job-like records for the web UI.

These demos let the frontend open polished analysis screens instantly without
making the user wait through upload + pipeline processing during a live demo.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import EVAL_DIR, EVAL_RUNS_DIR

DEMO_RUN_ID = "psg_annotated_base_convex_v1"
DEMO_JOB_PREFIX = "demo:"

DEMO_META = {
    "psg_eval_01": {
        "title": "PSG Clip 01",
        "subtitle_en": "PSG spacing review clip",
        "subtitle_th": "คลิปรีวิวการยืนตำแหน่งของ PSG",
    },
    "psg_eval_02": {
        "title": "PSG Clip 02",
        "subtitle_en": "PSG spacing review clip",
        "subtitle_th": "คลิปรีวิวการยืนตำแหน่งของ PSG",
    },
    "psg_eval_03": {
        "title": "PSG Clip 03",
        "subtitle_en": "PSG spacing review clip",
        "subtitle_th": "คลิปรีวิวการยืนตำแหน่งของ PSG",
    },
}


def _demo_run_dir() -> Path:
    return EVAL_RUNS_DIR / DEMO_RUN_ID


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _web_path(path: Path) -> str:
    rel = path.relative_to(EVAL_DIR).as_posix()
    return f"/eval-media/{rel}"


@lru_cache(maxsize=1)
def _load_summary() -> dict[str, Any]:
    path = _demo_run_dir() / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Demo summary not found: {path}")
    return _load_json(path)


def _clip_entry(clip_id: str) -> dict[str, Any]:
    summary = _load_summary()
    for clip in summary.get("clips", []):
        if clip.get("clip_id") == clip_id:
            return clip
    raise FileNotFoundError(f"Demo clip not found: {clip_id}")


def _metrics_for_clip(clip_id: str) -> dict[str, Any]:
    clip = _clip_entry(clip_id)
    return _load_json(Path(clip["metrics_path"]))


def _still_path(clip_id: str) -> Path | None:
    stills_dir = _demo_run_dir() / "stills"
    candidates = sorted(stills_dir.glob(f"{clip_id}_overlay_f*.png"))
    return candidates[0] if candidates else None


def _source_video_path(clip_id: str) -> Path:
    clip = _clip_entry(clip_id)
    return Path(clip["trimmed_video_path"])


def _explanation_from_metrics(clip_id: str, metrics: dict[str, Any], lang: str) -> dict[str, Any]:
    summary = metrics.get("summary", {})
    team_a = float(summary.get("team_A", {}).get("hull_area", {}).get("mean", 0.0))
    team_b = float(summary.get("team_B", {}).get("hull_area", {}).get("mean", 0.0))
    gap = float(summary.get("centroid_distance", {}).get("mean", 0.0))
    events = metrics.get("events", [])
    strongest = max(events, key=lambda e: abs(float(e.get("delta_pct", 0.0))), default=None)
    team = "Team A"
    action_en = "shape change"
    action_th = "การเปลี่ยนรูปแบบ"
    delta = 0.0
    moment = 0.0
    if strongest:
        team = "Team A" if strongest.get("team") == "team_A" else "Team B"
        moment = float(strongest.get("t", 0.0))
        delta = float(strongest.get("delta_pct", 0.0))
        if strongest.get("type") == "stretch":
            action_en = "spread sharply"
            action_th = "กระจายตัวชัดเจน"
        else:
            action_en = "compressed sharply"
            action_th = "บีบรูปทรงเข้าหากัน"

    wider_team = "Team A" if team_a >= team_b else "Team B"
    tighter_team = "Team B" if wider_team == "Team A" else "Team A"

    if lang == "th":
        headline = (
            f"{team} มี{action_th}เด่นที่สุดในคลิปนี้"
            if strongest
            else "คลิปนี้แสดงความต่างของการยืนระหว่างสองทีมได้ชัดเจน"
        )
        implication = (
            f"ค่าเฉลี่ยพื้นที่ทีมอยู่ที่ประมาณ {team_a/1000:.0f}k px² สำหรับ Team A "
            f"และ {team_b/1000:.0f}k px² สำหรับ Team B โดยมีระยะกึ่งกลางเฉลี่ย {gap:.0f} px. "
            f"{wider_team} กระจายกว่าส่วน {tighter_team} เก็บรูปทรงแน่นกว่า."
        )
        if strongest:
            coaching = (
                f"เริ่มดูที่ {moment:.1f} วินาที ซึ่ง {team} เปลี่ยนพื้นที่ประมาณ {delta:+.0f}%. "
                f"ใช้จังหวะนี้อธิบายว่าหน่วยหลังและกลางขยับตามกันทันหรือไม่."
            )
        else:
            coaching = (
                "ใช้คลิปนี้เปรียบเทียบความกว้างและความลึกของทั้งสองทีม "
                "แล้วชี้ให้ผู้เล่นเห็นว่าช่วงไหนควรบีบหรือกระจายพร้อมกัน."
            )
    else:
        headline = (
            f"{team} shows the clearest {action_en} pattern in this clip."
            if strongest
            else "This clip gives a clean view of how the two teams occupy space."
        )
        implication = (
            f"Average occupied area is about {team_a/1000:.0f}k px² for Team A and "
            f"{team_b/1000:.0f}k px² for Team B, with centroids {gap:.0f}px apart. "
            f"{wider_team} stays wider on average while {tighter_team} holds the more compact shape."
        )
        if strongest:
            coaching = (
                f"Start at {moment:.1f}s where {team} shifts by about {delta:+.0f}%. "
                f"Use that moment to discuss whether the back and midfield units moved together."
            )
        else:
            coaching = (
                "Use the clip to compare width and depth across both teams, then coach when the unit should "
                "squeeze together and when it should stretch to create a passing lane."
            )

    return {
        "clip_id": clip_id,
        "phase_context": "precomputed demo clip",
        "language": lang,
        "model": "demo_curated_v1",
        "prompt_version": "demo_v1",
        "headline": headline,
        "implication": implication,
        "coaching_cue": coaching,
        "raw_response": None,
    }


def _demo_job_from_metrics(clip_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
    meta = DEMO_META.get(clip_id, {})
    clip = _clip_entry(clip_id)
    source_video = _source_video_path(clip_id)
    event_count = len(metrics.get("events", []))
    return {
        "job_id": f"{DEMO_JOB_PREFIX}{clip_id}",
        "demo_id": clip_id,
        "is_demo": True,
        "filename": f"{clip_id}.mp4",
        "session_type": "match",
        "opponent": meta.get("title", clip_id),
        "notes": meta.get("subtitle_en"),
        "language": "en",
        "video_size": source_video.stat().st_size if source_video.exists() else None,
        "duration_s": float(metrics.get("duration_s", 0.0)),
        "fps": float(metrics.get("fps", 25.0)),
        "status": "done",
        "stage_message": "Precomputed demo clip ready.",
        "stage_index": 5,
        "stage_total": 5,
        "error": None,
        "explainer_error": None,
        "events_count": event_count,
        "summary": metrics.get("summary"),
        "created_at": "2026-05-12T00:00:00+00:00",
        "updated_at": "2026-05-12T00:00:00+00:00",
        "overlay_url": f"/api/demo/clips/{clip_id}/overlay",
        "source_video_url": _web_path(source_video),
        "thumbnail_url": _web_path(_still_path(clip_id)) if _still_path(clip_id) else None,
        "frame_count": clip.get("frame_count"),
    }


def list_demo_clips() -> list[dict[str, Any]]:
    summary = _load_summary()
    out: list[dict[str, Any]] = []
    for clip in summary.get("clips", []):
        clip_id = clip["clip_id"]
        metrics = _metrics_for_clip(clip_id)
        job = _demo_job_from_metrics(clip_id, metrics)
        meta = DEMO_META.get(clip_id, {})
        out.append({
            **job,
            "title": meta.get("title", clip_id),
            "subtitle_en": meta.get("subtitle_en"),
            "subtitle_th": meta.get("subtitle_th"),
        })
    return out


def load_demo_clip(clip_id: str) -> dict[str, Any] | None:
    for clip in list_demo_clips():
        if clip.get("demo_id") == clip_id:
            return clip
    return None


def load_demo_result(clip_id: str, lang: str = "en") -> dict[str, Any]:
    metrics = _metrics_for_clip(clip_id)
    job = _demo_job_from_metrics(clip_id, metrics)
    explanation = _explanation_from_metrics(clip_id, metrics, lang)
    return {"job": job, "metrics": metrics, "explanation": explanation}


def demo_overlay_path(clip_id: str) -> Path:
    clip = _clip_entry(clip_id)
    return Path(clip["overlay_path"])
