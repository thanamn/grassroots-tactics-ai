"""Run detector/tracker presets on the evaluation clips.

This script is intentionally separate from the production upload pipeline:
each model sees the same clips, writes raw tracking JSON, then runs the same
team-assignment stage so the scorer can compare outputs fairly.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.assign_teams import assign_teams
from src.config import EVAL_CLIPS_DIR, EVAL_MANIFESTS_DIR, EVAL_RUNS_DIR
from src.tracking import run_tracking


MODEL_PRESETS = {
    "football_players": {
        "description": "Production uisikdag YOLOv8 football weights; tracks ball, GK, and players.",
        "model": "models/football_players.pt",
        "track_classes": [0, 1, 2],
        "player_classes": [1, 2],
        "ball_classes": [0],
        "goalkeeper_classes": [1],
        "confidence": None,
        "role": "production",
    },
    "yolo11n": {
        "description": "Small generic COCO person detector; fastest fallback baseline.",
        "model": "yolo11n.pt",
        "track_classes": [0],
        "confidence": 0.25,
        "role": "fast_baseline",
    },
    "yolo11s": {
        "description": "Small-plus generic COCO person detector; balanced fallback baseline.",
        "model": "yolo11s.pt",
        "track_classes": [0],
        "confidence": 0.25,
        "role": "balanced_baseline",
    },
    "yolo11m": {
        "description": "Medium generic COCO person detector; tests whether scale beats domain tuning.",
        "model": "yolo11m.pt",
        "track_classes": [0],
        "confidence": 0.25,
        "role": "large_generic_baseline",
    },
}


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_extra_presets(path_str: str | None) -> dict:
    if not path_str:
        return {}
    obj = _load_json(Path(path_str))
    if not isinstance(obj, dict):
        raise ValueError("Preset file must be a JSON object keyed by preset name")
    return obj


def _resolve_video_path(path_str: str) -> Path:
    original = Path(path_str)
    local_eval_copy = EVAL_CLIPS_DIR / original.name
    if local_eval_copy.exists():
        return local_eval_copy
    if original.exists():
        return original
    raise FileNotFoundError(f"Could not resolve clip path: {path_str}")


def _unique_clips(manifest: dict) -> list[dict]:
    by_clip: OrderedDict[str, str] = OrderedDict()
    for frame in manifest["frames"]:
        by_clip.setdefault(frame["clip_id"], frame["source_video"])
    return [
        {"clip_id": clip_id, "video_path": str(_resolve_video_path(path_str))}
        for clip_id, path_str in by_clip.items()
    ]


def _check_model_file(model_name: str) -> None:
    if model_name.endswith(".pt"):
        path = ROOT / model_name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model weight {model_name}. Download it first or remove that preset."
            )


def _write_summary(summary_path: Path, summary: dict, existing_models: OrderedDict) -> None:
    summary["models"] = list(existing_models.values())
    summary["updated_at"] = datetime.now(timezone.utc).isoformat()
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--preset", dest="presets", action="append", default=[])
    parser.add_argument("--preset-file", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--skip-team-assignment", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    available_presets = dict(MODEL_PRESETS)
    available_presets.update(_load_extra_presets(args.preset_file))
    presets = args.presets or ["football_players", "yolo11n", "yolo11s", "yolo11m"]
    for preset in presets:
        if preset not in available_presets:
            raise ValueError(f"Unknown preset: {preset}")

    manifest = _load_json(EVAL_MANIFESTS_DIR / f"{args.manifest_id}.json")
    clips = _unique_clips(manifest)
    run_id = args.run_id or f"{args.manifest_id}_{_now_stamp()}"
    run_dir = EVAL_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"

    if summary_path.exists():
        summary = _load_json(summary_path)
        if summary.get("manifest_id") != args.manifest_id:
            raise ValueError(
                f"Existing run {run_id} is for manifest {summary.get('manifest_id')}, "
                f"not {args.manifest_id}"
            )
        existing_models = OrderedDict((m["preset"], m) for m in summary.get("models", []))
        summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        summary["vid_stride"] = args.vid_stride
    else:
        summary = {
            "run_id": run_id,
            "manifest_id": args.manifest_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vid_stride": args.vid_stride,
            "clips": clips,
            "models": [],
        }
        existing_models = OrderedDict()

    print(f"[benchmark] run_id={run_id}")
    for preset in presets:
        cfg = available_presets[preset]
        _check_model_file(cfg["model"])
        model_dir = run_dir / preset
        tracking_dir = model_dir / "tracking_raw"
        teams_dir = model_dir / "tracking_with_teams"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        teams_dir.mkdir(parents=True, exist_ok=True)

        if preset in existing_models and not args.force:
            expected_raw = [tracking_dir / f"{clip['clip_id']}.json" for clip in clips]
            expected_teams = [teams_dir / f"{clip['clip_id']}.json" for clip in clips]
            if all(path.exists() for path in expected_raw + expected_teams):
                print(f"[benchmark] preserve existing complete summary for {preset}")
                continue

        model_summary = {
            "preset": preset,
            "description": cfg.get("description", ""),
            "role": cfg.get("role", ""),
            "model": cfg["model"],
            "track_classes": cfg["track_classes"],
            "player_classes": cfg.get("player_classes"),
            "ball_classes": cfg.get("ball_classes"),
            "goalkeeper_classes": cfg.get("goalkeeper_classes"),
            "confidence": cfg.get("confidence"),
            "clips": [],
        }

        for clip in clips:
            clip_id = clip["clip_id"]
            video_path = Path(clip["video_path"])
            raw_path = tracking_dir / f"{clip_id}.json"
            team_path = teams_dir / f"{clip_id}.json"

            if raw_path.exists() and team_path.exists() and not args.force:
                print(f"[benchmark] skip existing {preset} / {clip_id}")
                model_summary["clips"].append({
                    "clip_id": clip_id,
                    "video_path": str(video_path),
                    "tracking_path": str(raw_path),
                    "team_tracking_path": str(team_path),
                    "runtime_s": None,
                    "processed_fps": None,
                    "skipped": True,
                })
                continue

            print(f"[benchmark] {preset} -> {clip_id}")
            t0 = time.perf_counter()
            tracking = run_tracking(
                video_path,
                model_name=cfg["model"],
                vid_stride=args.vid_stride,
                track_classes=cfg["track_classes"],
                confidence=cfg.get("confidence"),
                player_classes=cfg.get("player_classes"),
                ball_classes=cfg.get("ball_classes"),
                goalkeeper_classes=cfg.get("goalkeeper_classes"),
            )
            runtime_s = time.perf_counter() - t0
            raw_path.write_text(json.dumps(tracking, indent=2, ensure_ascii=False), encoding="utf-8")
            shutil.copy2(raw_path, team_path)
            has_player_classes = bool(cfg.get("player_classes", cfg.get("track_classes", [])))
            if not args.skip_team_assignment and not cfg.get("skip_team_assignment") and has_player_classes:
                assign_teams(team_path, video_path)

            processed_frames = tracking.get("frame_count", 0)
            model_summary["clips"].append({
                "clip_id": clip_id,
                "video_path": str(video_path),
                "tracking_path": str(raw_path),
                "team_tracking_path": str(team_path),
                "runtime_s": round(runtime_s, 3),
                "processed_fps": round(processed_frames / runtime_s, 3) if runtime_s > 0 else None,
                "skipped": False,
            })

        existing_models[preset] = model_summary
        _write_summary(summary_path, summary, existing_models)

    _write_summary(summary_path, summary, existing_models)
    print(f"[benchmark] wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
