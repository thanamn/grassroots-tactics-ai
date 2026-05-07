"""Re-run assign_teams → metrics → visualizer for one or all existing jobs.

Skips the tracking step (YOLO inference) because the tracking JSONs are
already fixed by fix_tracking_cls.py and re-tracking would be slow.
Updates data/jobs/<id>.json summary so the web UI shows correct numbers.

Usage:
    # All jobs:
    .venv\Scripts\python.exe scripts/rerun_pipeline.py

    # Single job:
    .venv\Scripts\python.exe scripts/rerun_pipeline.py --job 6a532b15a9ba
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.jobs import load_job, update_job
from src.config import CACHE_DIR, CLIPS_DIR, TRACKING_DIR


def rerun_job(job_id: str) -> bool:
    job = load_job(job_id)
    if not job:
        print(f"  [{job_id}] no job record found — skipping")
        return False

    video_path    = CLIPS_DIR   / f"{job_id}.mp4"
    tracking_path = TRACKING_DIR / f"{job_id}.json"

    if not video_path.exists():
        print(f"  [{job_id}] video missing: {video_path} — skipping")
        return False
    if not tracking_path.exists():
        print(f"  [{job_id}] tracking JSON missing — skipping")
        return False

    print(f"\n=== {job_id} ({job.get('filename', '?')}) ===")

    try:
        # Stage 1 — assign_teams (in-place update of tracking JSON)
        print("  [1/3] assign_teams …")
        from scripts.assign_teams import assign_teams
        assign_teams(tracking_path, video_path)

        # Stage 2 — metrics
        print("  [2/3] metrics …")
        from src.metrics import compute_metrics
        tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
        metrics  = compute_metrics(tracking)
        metrics_path = CACHE_DIR / f"{job_id}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        update_job(job_id,
                   duration_s=metrics.get("duration_s"),
                   fps=metrics.get("fps"),
                   events_count=len(metrics.get("events", [])),
                   summary=metrics.get("summary"))

        # Stage 3 — visualizer
        print("  [3/3] visualizer …")
        from src.visualizer import render_overlay
        overlay_path = CACHE_DIR / f"{job_id}_overlay.mp4"
        render_overlay(video_path, tracking_path, overlay_path)

        update_job(job_id, status="done", stage_message="Analysis complete.")
        print(f"  [{job_id}] done.")
        return True

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"  [{job_id}] FAILED: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default=None, help="Single job ID to re-run (omit for all)")
    args = parser.parse_args()

    if args.job:
        job_ids = [args.job]
    else:
        job_ids = sorted(p.stem for p in (ROOT / "data" / "jobs").glob("*.json"))

    print(f"Re-running pipeline (assign_teams -> metrics -> visualizer) for {len(job_ids)} job(s)")
    ok = failed = 0
    for jid in job_ids:
        if rerun_job(jid):
            ok += 1
        else:
            failed += 1

    print(f"\nFinished: {ok} succeeded, {failed} failed/skipped.")


if __name__ == "__main__":
    main()
