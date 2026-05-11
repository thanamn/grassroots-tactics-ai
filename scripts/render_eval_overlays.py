"""Render overlay videos for every clip in an evaluation run.

This reuses the saved benchmark tracking JSONs, so it is much cheaper than
rerunning detector inference. For each model preset in the run summary, it
renders the actual video with the corresponding tracking/team JSON into an
`overlays/` folder inside that model directory.

Usage:
    .venv\Scripts\python.exe scripts\render_eval_overlays.py --run-id psg_tracking_latest_v1
    .venv\Scripts\python.exe scripts\render_eval_overlays.py --run-id psg_tracking_latest_v1 --preset yolo11n --preset football_players
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import EVAL_RUNS_DIR
from src.visualizer import render_overlay


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _render_run(run_id: str, presets: set[str] | None, force: bool) -> tuple[int, int, int]:
    run_dir = EVAL_RUNS_DIR / run_id
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path)

    rendered = 0
    skipped = 0
    failed = 0

    print(f"[overlay] run_id={run_id}")
    for model in summary.get("models", []):
        preset = model["preset"]
        if presets and preset not in presets:
            continue

        model_dir = run_dir / preset
        overlay_dir = model_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        clip_count = 0
        for clip in model.get("clips", []):
            clip_id = clip["clip_id"]
            video_path = Path(clip["video_path"])
            tracking_path = Path(clip.get("team_tracking_path") or clip["tracking_path"])
            output_path = overlay_dir / f"{clip_id}_overlay.mp4"

            if output_path.exists() and not force:
                print(f"[overlay] skip existing {preset} / {clip_id}")
                skipped += 1
                clip_count += 1
                continue

            print(f"[overlay] render {preset} / {clip_id}")
            try:
                render_overlay(video_path, tracking_path, output_path)
                rendered += 1
                clip_count += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[overlay] FAILED {preset} / {clip_id}: {exc}")

        if clip_count == 0:
            print(f"[overlay] no rendered clips for preset {preset}")

    print(f"[overlay] done run_id={run_id} rendered={rendered} skipped={skipped} failed={failed}")
    return rendered, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", dest="run_ids", action="append", required=True)
    parser.add_argument("--preset", dest="presets", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    presets = set(args.presets) if args.presets else None
    total_rendered = 0
    total_skipped = 0
    total_failed = 0

    for run_id in args.run_ids:
        rendered, skipped, failed = _render_run(run_id, presets, args.force)
        total_rendered += rendered
        total_skipped += skipped
        total_failed += failed

    print(
        f"[overlay] overall rendered={total_rendered} skipped={total_skipped} failed={total_failed}"
    )


if __name__ == "__main__":
    main()
