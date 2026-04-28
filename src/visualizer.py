"""Render a tactical overlay on a video: convex hull + centroid per team.

Usage:
    python -m src.visualizer \
        --video data/clips/sample.mp4 \
        --tracking data/tracking/sample.json \
        --output data/cache/sample_overlay.mp4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from src.config import CACHE_DIR, CLIPS_DIR, TRACKING_DIR

TEAM_COLORS = {
    "A": (0, 200, 255),   # amber (BGR)
    "B": (255, 120, 60),  # blue
}


def _draw_team(frame: np.ndarray, points: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
    if len(points) < 3:
        return
    pts = np.asarray(points, dtype=np.float32)
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices].astype(np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [hull_pts], color)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, dst=frame)
        cv2.polylines(frame, [hull_pts], isClosed=True, color=color, thickness=2)
    except QhullError:
        pass

    centroid = pts.mean(axis=0).astype(int)
    cv2.circle(frame, tuple(centroid), 6, color, -1)
    cv2.circle(frame, tuple(centroid), 6, (255, 255, 255), 1)


def render_overlay(video_path: Path, tracking_path: Path, output_path: Path) -> None:
    tracking = json.loads(tracking_path.read_text())
    frames_by_idx = {f["frame"]: f for f in tracking["frames"]}

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or tracking.get("fps", 25)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        record = frames_by_idx.get(frame_idx)
        if record:
            for team, color in TEAM_COLORS.items():
                pts = [(p["x"], p["y"]) for p in record["players"] if p.get("team") == team]
                _draw_team(frame, pts, color)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # mp4v is not browser-playable; re-encode to H.264 if ffmpeg is available.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        tmp = output_path.with_suffix(".h264.mp4")
        subprocess.run(
            [ffmpeg, "-y", "-i", str(output_path),
             "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
             "-pix_fmt", "yuv420p", str(tmp)],
            check=True, capture_output=True,
        )
        tmp.replace(output_path)

    print(f"Wrote {output_path} — {frame_idx} frames")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        video_path = CLIPS_DIR / video_path.name
    tracking_path = Path(args.tracking)
    if not tracking_path.exists():
        tracking_path = TRACKING_DIR / tracking_path.name
    output_path = Path(args.output) if args.output else CACHE_DIR / f"{video_path.stem}_overlay.mp4"

    render_overlay(video_path, tracking_path, output_path)


if __name__ == "__main__":
    main()
