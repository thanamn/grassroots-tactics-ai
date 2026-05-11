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
                    window: int = SMOOTH_W) -> dict[str, dict]:
    """Return per-team player positions smoothed over the last ``window`` frames.

    Returns a dict keyed by team ("A"/"B"), each value being:
        {"outfield": [(x, y), ...], "gks": [(x, y), ...]}

    GK identification uses the ``cls`` field saved by tracking.py (1 = GK,
    2 = outfield player). A track_id is treated as GK when >40 % of its
    detections in the smoothing window were class 1. If the JSON pre-dates
    the cls field (legacy), all players land in "outfield" and the caller
    falls back to the old furthest-from-centroid heuristic.
    """
    lo = max(0, frame_idx - window + 1)
    # track_id → {positions: [(x,y)...], gk_count: int, total: int}
    team_tracks: dict[str, dict[int, dict]] = {"A": {}, "B": {}}

    for fi in range(lo, frame_idx + 1):
        fd = frames_by_idx.get(fi)
        if not fd:
            continue
        for p in fd["players"]:
            t = p.get("team")
            if t not in ("A", "B"):
                continue
            tid = p["track_id"]
            d = team_tracks[t].setdefault(tid, {"pos": [], "gk": 0, "n": 0})
            d["pos"].append((p["x"], p["y"]))
            d["n"] += 1
            if p.get("cls") == 1:
                d["gk"] += 1

    result: dict[str, dict] = {}
    for team, tracks in team_tracks.items():
        outfield: list[tuple[float, float]] = []
        gks: list[tuple[float, float]] = []
        for d in tracks.values():
            avg_x = sum(x for x, _ in d["pos"]) / len(d["pos"])
            avg_y = sum(y for _, y in d["pos"]) / len(d["pos"])
            if d["n"] >= 3 and d["gk"] / d["n"] > 0.50:
                gks.append((avg_x, avg_y))
            else:
                outfield.append((avg_x, avg_y))
        result[team] = {"outfield": outfield, "gks": gks}
    return result



def _draw_team(frame: np.ndarray,
               outfield: list[tuple[float, float]],
               gks: list[tuple[float, float]],
               color: tuple[int, int, int]) -> None:
    """Draw tactical overlay for one team.

    - Light convex hull fill shows the team's occupied zone.
    - Delaunay triangulation edges connect every pair of nearby outfield
      players, showing the actual formation shape (not just the outer hull).
    - GKs (identified via cls=1 from the football model) are drawn with a
      distinct marker and excluded from the hull / Delaunay so they don't
      distort the outfield shape metrics.
    - Falls back to the old "drop furthest player" heuristic when no cls
      info is available (legacy JSON produced before this fix).
    """
    pts = np.asarray(outfield, dtype=np.float32) if outfield else np.empty((0, 2), dtype=np.float32)

    # Legacy fallback: if no GK was identified via cls, drop the player
    # furthest from the team centroid (the old proxy).
    if len(gks) == 0 and len(pts) >= 4:
        c = pts.mean(axis=0)
        pts = pts[np.argsort(np.linalg.norm(pts - c, axis=1))[:-1]]

    h, w = frame.shape[:2]
    # Scale dot/line sizes to frame resolution so they're visible on any input.
    # Base calibrated for 1280×720; scale proportionally for other resolutions.
    scale = max(w, h) / 1280.0
    dot_r   = max(4, int(8  * scale))   # outfield player dot radius
    gk_r    = max(6, int(11 * scale))   # GK circle radius
    line_w  = max(2, int(2  * scale))   # formation line thickness
    cross_d = max(5, int(7  * scale))   # GK cross arm length

    # Connect players within 35 % of the longer frame dimension so wide
    # fullbacks/wingers still get lines to their nearest teammates.
    max_edge_px = max(w, h) * 0.35

    # Light convex-hull fill — keeps the team-zone feel without cluttering lines
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices].astype(np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [hull_pts], color)
            cv2.addWeighted(overlay, 0.13, frame, 0.87, 0, dst=frame)
        except QhullError:
            pass

    # Draw lines between every pair of outfield players within max_edge_px.
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            edge_len = float(np.linalg.norm(pts[i] - pts[j]))
            if edge_len <= max_edge_px:
                p1 = (int(pts[i, 0]), int(pts[i, 1]))
                p2 = (int(pts[j, 0]), int(pts[j, 1]))
                cv2.line(frame, p1, p2, color, line_w, cv2.LINE_AA)

    # Outfield player dots — draw filled circle then black outline for contrast
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), dot_r, color, -1)
        cv2.circle(frame, (int(x), int(y)), dot_r, (0, 0, 0), 1)

    # GK: larger circle + white cross so it's visually distinct
    for x, y in gks:
        ix, iy = int(x), int(y)
        cv2.circle(frame, (ix, iy), gk_r, color, -1)
        cv2.circle(frame, (ix, iy), gk_r, (255, 255, 255), 2)
        cv2.line(frame, (ix - cross_d, iy - cross_d), (ix + cross_d, iy + cross_d), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (ix + cross_d, iy - cross_d), (ix - cross_d, iy + cross_d), (255, 255, 255), 2, cv2.LINE_AA)

    # Team centroid dot
    if len(pts) > 0:
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.circle(frame, (int(cx), int(cy)), gk_r, color, -1)
        cv2.circle(frame, (int(cx), int(cy)), gk_r, (255, 255, 255), 2)


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
            data = smoothed.get(team, {"outfield": [], "gks": []})
            _draw_team(frame, data["outfield"], data["gks"], color)
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
