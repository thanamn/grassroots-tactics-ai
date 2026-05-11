"""Persistence helpers for evaluation manifests and manual annotations.

Evaluation annotations are researcher-authored study data, so they live beside
but separate from product job records. The UI can incrementally save player
points, ball position, and possession labels without touching uploaded jobs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import EVAL_ANNOTATIONS_DIR, EVAL_MANIFESTS_DIR, EVAL_RUNS_DIR


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_path(manifest_id: str) -> Path:
    return EVAL_MANIFESTS_DIR / f"{manifest_id}.json"


def annotation_path(manifest_id: str) -> Path:
    return EVAL_ANNOTATIONS_DIR / f"{manifest_id}.json"


def load_manifest(manifest_id: str) -> dict[str, Any] | None:
    path = manifest_path(manifest_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_manifests() -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(EVAL_MANIFESTS_DIR.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        manifests.append({
            "manifest_id": obj.get("manifest_id", path.stem),
            "frame_count": len(obj.get("frames", [])),
            "created_at": obj.get("created_at"),
            "description": obj.get("description"),
        })
    return manifests


def load_annotations(manifest_id: str) -> dict[str, Any]:
    path = annotation_path(manifest_id)
    if not path.exists():
        return {
            "manifest_id": manifest_id,
            "saved_at": None,
            "frames": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_annotations(manifest_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["manifest_id"] = manifest_id
    payload["saved_at"] = _now_iso()
    annotation_path(manifest_id).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def list_benchmark_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(EVAL_RUNS_DIR.glob("*/summary.json")):
        try:
            obj = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        runs.append({
            "run_id": obj.get("run_id", summary_path.parent.name),
            "manifest_id": obj.get("manifest_id"),
            "created_at": obj.get("created_at"),
            "models": [
                {
                    "preset": model.get("preset"),
                    "model": model.get("model"),
                    "clip_count": len(model.get("clips", [])),
                }
                for model in obj.get("models", [])
            ],
        })
    return runs


def _tracking_frames_by_index(path: Path) -> dict[int, list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(frame["frame"]): frame.get("players", [])
        for frame in obj.get("frames", [])
    }


def seed_annotations_from_run(
    manifest_id: str,
    run_id: str,
    preset: str,
    *,
    overwrite: bool = False,
    include_other: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_id)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found: {manifest_id}")

    run_dir = EVAL_RUNS_DIR / run_id / preset / "tracking_with_teams"
    if not run_dir.exists():
        raise FileNotFoundError(f"Tracking output not found: {run_dir}")

    annotations = load_annotations(manifest_id)
    annotations.setdefault("frames", {})

    clip_cache: dict[str, dict[int, list[dict[str, Any]]]] = {}
    seeded_frames = 0

    for frame in manifest.get("frames", []):
        frame_id = frame["frame_id"]
        existing = annotations["frames"].get(frame_id, {})
        if not overwrite and existing.get("points"):
            continue

        clip_id = frame["clip_id"]
        if clip_id not in clip_cache:
            tracking_path = run_dir / f"{clip_id}.json"
            if not tracking_path.exists():
                raise FileNotFoundError(tracking_path)
            clip_cache[clip_id] = _tracking_frames_by_index(tracking_path)

        players = clip_cache[clip_id].get(int(frame["frame_index"]), [])
        points = []
        for player in players:
            team = player.get("team")
            if team not in ("A", "B"):
                if not include_other:
                    continue
                team = "Other"
            points.append({
                "id": f"T{int(player['track_id'])}",
                "x": float(player["x"]),
                "y": float(player["y"]),
                "team": team,
            })
        points.sort(key=lambda pt: (pt["team"], pt["x"], pt["y"]))

        next_frame = dict(existing)
        next_frame["points"] = points
        annotations["frames"][frame_id] = next_frame
        seeded_frames += 1

    annotations["seed_info"] = {
        "run_id": run_id,
        "preset": preset,
        "overwrite": overwrite,
        "include_other": include_other,
        "seeded_frames": seeded_frames,
        "seeded_at": _now_iso(),
    }
    return save_annotations(manifest_id, annotations)
