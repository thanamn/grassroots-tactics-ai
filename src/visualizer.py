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
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from src.config import CACHE_DIR, CLIPS_DIR, TRACKING_DIR

TEAM_COLORS = {
    "A": (0, 200, 255),   # amber (BGR)
    "B": (255, 120, 60),  # blue
}

# Temporal smoothing window (frames). Hull is built from the union of the last
# SMOOTH_W frames so that a player missing for a few frames due to occlusion
# or a missed detection doesn't cause the hull to collapse and flicker.
# At 25 fps, 11 frames ≈ 0.44 s — bridges a longer gap than the previous
# 7-frame value while staying short enough that the dot does not linger
# after a player has genuinely left the area. Pairs with the longer tracker
# emit window (max_flow_emit_lost=25) and offline stitch/gap-interp in
# src/tracking.py — together they keep one player's dot from blinking in
# and out across short detector dropouts.
SMOOTH_W = 11
MAX_HULL_PLAYERS_PER_TEAM = 10


def _select_hull_points(points: np.ndarray,
                        max_points: int = MAX_HULL_PLAYERS_PER_TEAM) -> np.ndarray:
    """Choose stable hull inputs without hiding tracked player dots.

    If tracking has extra team-coloured candidates, remove interior/duplicate
    points first. Extreme players that define the actual team width/depth are
    preserved unless there are still too many hull vertices.
    """
    if len(points) <= max_points:
        return points

    pts = points.astype(np.float32).copy()
    while len(pts) > max_points:
        removed_idx: int | None = None
        if len(pts) >= 4:
            try:
                hull_indices = set(int(i) for i in ConvexHull(pts).vertices)
                interior = [i for i in range(len(pts)) if i not in hull_indices]
                if interior:
                    centre = np.median(pts, axis=0)
                    removed_idx = min(
                        interior,
                        key=lambda i: float(np.linalg.norm(pts[i] - centre)),
                    )
            except QhullError:
                pass

        if removed_idx is None:
            # Fall back to removing one point from the closest pair, which is
            # usually a duplicate/track split rather than a real shape edge.
            best_pair = None
            best_dist = float("inf")
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = float(np.linalg.norm(pts[i] - pts[j]))
                    if d < best_dist:
                        best_dist = d
                        best_pair = (i, j)
            if best_pair is None:
                break
            centre = np.median(pts, axis=0)
            i, j = best_pair
            removed_idx = i if np.linalg.norm(pts[i] - centre) < np.linalg.norm(pts[j] - centre) else j

        pts = np.delete(pts, removed_idx, axis=0)
    return pts


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
    has_shape_flags: dict[str, bool] = {"A": False, "B": False}

    for fi in range(lo, frame_idx + 1):
        fd = frames_by_idx.get(fi)
        if not fd:
            continue
        for p in fd["players"]:
            t = p.get("team")
            if t not in ("A", "B"):
                continue
            has_shape_flags[t] = has_shape_flags[t] or "include_in_shape" in p
            if p.get("include_in_shape") is False:
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
        result[team] = {
            "outfield": outfield,
            "gks": gks,
            "legacy_no_shape_flags": not has_shape_flags[team],
        }
    return result


def _draw_team(frame: np.ndarray,
               outfield: list[tuple[float, float]],
               gks: list[tuple[float, float]],
               color: tuple[int, int, int],
               legacy_no_shape_flags: bool = False) -> None:
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
    hull_pts_src = _select_hull_points(pts)

    # Legacy fallback: if no GK was identified via cls, drop the player
    # furthest from the team centroid (the old proxy).
    if legacy_no_shape_flags and len(gks) == 0 and len(hull_pts_src) >= 4:
        c = hull_pts_src.mean(axis=0)
        hull_pts_src = hull_pts_src[np.argsort(np.linalg.norm(hull_pts_src - c, axis=1))[:-1]]

    h, w = frame.shape[:2]
    # Max edge length: 30 % of the longer frame dimension.  Connects players
    # within the same unit (defensive / midfield line) without drawing wild
    # diagonals across the entire pitch.
    max_edge_px = max(w, h) * 0.25

    # Light convex-hull fill — keeps the team-zone feel without cluttering lines
    if len(hull_pts_src) >= 3:
        try:
            hull = ConvexHull(hull_pts_src)
            hull_pts = hull_pts_src[hull.vertices].astype(np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [hull_pts], color)
            cv2.addWeighted(overlay, 0.13, frame, 0.87, 0, dst=frame)
        except QhullError:
            pass

    # Draw lines between every pair of outfield players within max_edge_px.
    # Simpler than Delaunay: avoids the problematic long outer-boundary edges
    # that Delaunay forces between isolated players. O(n²) is fine for n≤11.
    for i in range(len(hull_pts_src)):
        for j in range(i + 1, len(hull_pts_src)):
            edge_len = float(np.linalg.norm(hull_pts_src[i] - hull_pts_src[j]))
            if edge_len <= max_edge_px:
                p1 = (int(hull_pts_src[i, 0]), int(hull_pts_src[i, 1]))
                p2 = (int(hull_pts_src[j, 0]), int(hull_pts_src[j, 1]))
                cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

    # Outfield player dots
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 0), 1)

    # Team centroid dot
    if len(hull_pts_src) > 0:
        cx, cy = hull_pts_src.mean(axis=0).astype(int)
        cv2.circle(frame, (int(cx), int(cy)), 7, color, -1)
        cv2.circle(frame, (int(cx), int(cy)), 7, (255, 255, 255), 2)


def render_overlay(
    video_path: Path,
    tracking_path: Path,
    output_path: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or tracking.get("frame_count_meta") or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # With stride > 1 only every Nth frame has a tracking entry, but we
    # want every output frame to carry the hull overlay (otherwise the
    # video flashes between tracked and bare frames). So render the
    # overlay on EVERY video frame, using the smoothing window to fill
    # in player positions from the nearest tracked entries.
    frame_idx = 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        smoothed = _smoothed_teams(frames_by_idx, frame_idx, window=window)
        for team, color in TEAM_COLORS.items():
            data = smoothed.get(team, {"outfield": [], "gks": [], "legacy_no_shape_flags": False})
            _draw_team(frame, data["outfield"], data["gks"], color, data["legacy_no_shape_flags"])
        writer.write(frame)
        frame_idx += 1
        if progress_callback and (frame_idx == 1 or frame_idx % 100 == 0):
            elapsed = max(0.001, time.perf_counter() - started)
            actual_fps = frame_idx / elapsed
            remaining_frames = max(0, total_frames - frame_idx) if total_frames else 0
            progress_callback({
                "processed": frame_idx,
                "total": total_frames or None,
                "progress": min(0.90, (frame_idx / max(1, total_frames)) * 0.90) if total_frames else 0.0,
                "elapsed_s": elapsed,
                "estimated_remaining_s": (remaining_frames / max(0.001, actual_fps)) + max(5.0, elapsed * 0.10),
            })

    cap.release()
    writer.release()
    if progress_callback:
        elapsed = max(0.001, time.perf_counter() - started)
        progress_callback({
            "processed": frame_idx,
            "total": total_frames or None,
            "progress": 0.93,
            "elapsed_s": elapsed,
            "estimated_remaining_s": max(5.0, elapsed * 0.08),
        })

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
        if progress_callback:
            elapsed = max(0.001, time.perf_counter() - started)
            progress_callback({
                "processed": frame_idx,
                "total": total_frames or None,
                "progress": 1.0,
                "elapsed_s": elapsed,
                "estimated_remaining_s": 0.0,
            })

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
