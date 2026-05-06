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

# Temporal smoothing window (frames). Hull is built from the union of the last
# SMOOTH_W frames so that a player missing for 1–3 frames due to occlusion or
# a missed detection doesn't cause the hull to collapse and flicker.
# At 25 fps, 7 frames ≈ 0.28 s — enough to bridge short gaps, short enough
# that ghost positions don't linger visibly.
SMOOTH_W = 7


def _smoothed_teams(frames_by_idx: dict, frame_idx: int,
                    window: int = SMOOTH_W) -> dict[str, list[tuple[float, float]]]:
    """Return per-team player positions smoothed over the last ``window`` frames.

    For each track_id that appeared in any frame in [frame_idx-window+1,
    frame_idx], we take the average (x, y) over those appearances. This keeps
    each player at a stable position even when YOLO misses them for a frame or
    two, eliminating hull flicker without adding visible lag.

    When tracking was run with ``vid_stride > 1`` the caller passes a larger
    window (typically ``SMOOTH_W * vid_stride``) so the same number of
    tracked entries is covered despite the gaps between them.
    """
    lo = max(0, frame_idx - window + 1)
    team_tracks: dict[str, dict[int, list]] = {"A": {}, "B": {}}

    for fi in range(lo, frame_idx + 1):
        fd = frames_by_idx.get(fi)
        if not fd:
            continue
        for p in fd["players"]:
            t = p.get("team")
            if t not in ("A", "B"):
                continue
            team_tracks[t].setdefault(p["track_id"], []).append((p["x"], p["y"]))

    result: dict[str, list] = {}
    for team, tracks in team_tracks.items():
        pts = []
        for positions in tracks.values():
            xs = [pos[0] for pos in positions]
            ys = [pos[1] for pos in positions]
            pts.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        result[team] = pts
    return result


def _draw_team(frame: np.ndarray, points: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
    if len(points) < 3:
        return
    pts = np.asarray(points, dtype=np.float32)

    # Mirror metrics.py: drop the player furthest from the team centroid
    # (goalkeeper proxy). Keeps overlay consistent with what the metric numbers show.
    if len(pts) >= 4:
        c = pts.mean(axis=0)
        pts = pts[np.argsort(np.linalg.norm(pts - c, axis=1))[:-1]]

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

    # Scale the smoothing window by the tracking stride so sparse frames
    # still produce a stable hull. With stride=5 and SMOOTH_W=7 we'd
    # otherwise only have ~1 tracked entry per window — flickery hull.
    # Scaling to SMOOTH_W * stride keeps ~SMOOTH_W tracked entries in
    # the average regardless of stride.
    stride = max(1, int(tracking.get("vid_stride", 1)))
    window = SMOOTH_W * stride

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or tracking.get("fps", 25)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # With stride > 1 only every Nth frame has a tracking entry, but we
    # want every output frame to carry the hull overlay (otherwise the
    # video flashes between tracked and bare frames). So render the
    # overlay on EVERY video frame, using the smoothing window to fill
    # in player positions from the nearest tracked entries.
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        smoothed = _smoothed_teams(frames_by_idx, frame_idx, window=window)
        for team, color in TEAM_COLORS.items():
            _draw_team(frame, smoothed.get(team, []), color)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # mp4v is not browser-playable; re-encode to H.264 if ffmpeg is available.
    # Falls back to the static binary that imageio-ffmpeg ships, so the
    # pipeline produces a playable file even when system ffmpeg is missing.
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = None
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
