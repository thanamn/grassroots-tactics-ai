"""Build refined tracking from manual annotations, then render the normal convex overlay.

This script is for demo-quality PSG outputs where the manual annotation should
be treated as the roster- and position-base truth. Rather than drawing a new
visual style, it creates a refined tracking JSON and then reuses the repo's
existing spacing/compactness pipeline:

    manual annotation base
        + dense tracker motion between anchors
        -> refined tracking JSON
        -> trimmed clip over the tracked span
        -> normal convex/mesh overlay
        -> normal metrics JSON

The result looks like the other runs in this repo, but the underlying player
positions are much more tightly anchored to the manual annotation set.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import medfilt, savgol_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import EVAL_ANNOTATIONS_DIR, EVAL_RUNS_DIR
from src.metrics import compute_metrics
from src.visualizer import render_overlay


@dataclass
class RefinedTrack:
    obj_id: str
    track_id: int
    team: str
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    h: np.ndarray
    visible: np.ndarray
    anchor_frames: list[int]
    observed_frames: list[int]
    source: str


@dataclass
class RefinedBall:
    x: np.ndarray
    y: np.ndarray
    visible: np.ndarray
    anchor_frames: list[int]
    observed_frames: list[int]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_annotations(path: Path) -> tuple[dict[str, dict[int, list[dict[str, Any]]]], dict[str, dict[int, dict[str, Any]]]]:
    obj = _load_json(path)
    players_by_clip: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    ball_by_clip: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for frame_id, frame in obj.get("frames", {}).items():
        clip_id, frame_num = frame_id.rsplit("_f", 1)
        frame_idx = int(frame_num)
        players_by_clip[clip_id][frame_idx] = frame.get("points", [])
        if frame.get("ball"):
            ball_by_clip[clip_id][frame_idx] = frame["ball"]
    return players_by_clip, ball_by_clip


def _tracking_index(path: Path) -> tuple[dict[int, dict[int, dict[str, Any]]], dict]:
    obj = _load_json(path)
    frames = {}
    for frame in obj.get("frames", []):
        frames[int(frame["frame"])] = {int(p["track_id"]): p for p in frame.get("players", [])}
    return frames, obj


def _full_tracking_frames(path: Path) -> list[dict[str, Any]]:
    return _load_json(path).get("frames", [])


def _track_num(obj_id: str) -> int | None:
    if obj_id.startswith("T") and obj_id[1:].isdigit():
        return int(obj_id[1:])
    return None


def _synthetic_track_num(obj_id: str) -> int:
    if obj_id.startswith("P") and obj_id[1:].isdigit():
        return 10000 + int(obj_id[1:])
    return 20000 + abs(hash(obj_id)) % 10000


def _team_for_object(clip_points: dict[int, list[dict[str, Any]]], obj_id: str) -> str:
    labels: list[str] = []
    for points in clip_points.values():
        for pt in points:
            if str(pt["id"]) == obj_id and pt.get("team"):
                labels.append(str(pt["team"]))
    return Counter(labels).most_common(1)[0][0] if labels else ""


def _anchor_series(
    clip_points: dict[int, list[dict[str, Any]]],
    obj_id: str,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    frames: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    for frame_idx in sorted(clip_points):
        for pt in clip_points[frame_idx]:
            if str(pt["id"]) == obj_id:
                frames.append(frame_idx)
                xs.append(float(pt["x"]))
                ys.append(float(pt["y"]))
                break
    return frames, np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _observed_series(
    tracking_frames: dict[int, dict[int, dict[str, Any]]],
    obj_id: str,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    track_id = _track_num(obj_id)
    if track_id is None:
        return [], np.array([]), np.array([]), np.array([]), np.array([])

    frames: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    ws: list[float] = []
    hs: list[float] = []
    for frame_idx in sorted(tracking_frames):
        player = tracking_frames[frame_idx].get(track_id)
        if not player:
            continue
        x1, y1, x2, y2 = [float(v) for v in player["bbox"]]
        frames.append(frame_idx)
        xs.append(float(player["x"]))
        ys.append(float(player["y"]))
        ws.append(max(1.0, x2 - x1))
        hs.append(max(1.0, y2 - y1))
    return (
        frames,
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(ws, dtype=np.float32),
        np.asarray(hs, dtype=np.float32),
    )


def _ball_series(full_frames: list[dict[str, Any]]) -> tuple[list[int], np.ndarray, np.ndarray]:
    frames: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    for frame in full_frames:
        ball = frame.get("ball")
        if not ball:
            continue
        frames.append(int(frame["frame"]))
        xs.append(float(ball["x"]))
        ys.append(float(ball["y"]))
    return frames, np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _ball_anchor_series(ball_points: dict[int, dict[str, Any]]) -> tuple[list[int], np.ndarray, np.ndarray]:
    frames: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    for frame_idx in sorted(ball_points):
        ball = ball_points[frame_idx]
        if ball.get("status") == "visible":
            frames.append(frame_idx)
            xs.append(float(ball["x"]))
            ys.append(float(ball["y"]))
    return frames, np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _linear_fill(frame_count: int, frames: list[int], values: np.ndarray) -> np.ndarray:
    dense = np.full(frame_count, np.nan, dtype=np.float32)
    if len(frames) == 0:
        return dense
    dense[np.asarray(frames, dtype=int)] = values
    valid = np.flatnonzero(~np.isnan(dense))
    if len(valid) == 1:
        dense[:] = dense[valid[0]]
        return dense
    full = np.arange(frame_count, dtype=np.float32)
    return np.interp(full, valid.astype(np.float32), dense[valid]).astype(np.float32)


def _smooth(vals: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    out = vals.astype(np.float32).copy()
    if len(out) >= 5:
        med_k = min(len(out) if len(out) % 2 == 1 else len(out) - 1, 5)
        if med_k >= 3:
            out = medfilt(out, kernel_size=med_k).astype(np.float32)
    if len(out) >= 7:
        win = min(window, len(out) if len(out) % 2 == 1 else len(out) - 1)
        if win >= poly + 3 and win % 2 == 1:
            out = savgol_filter(out, window_length=win, polyorder=poly, mode="interp").astype(np.float32)
    return out


def _interp_residual(frame_count: int, anchor_frames: list[int], residuals: np.ndarray) -> np.ndarray:
    full = np.arange(frame_count, dtype=np.float32)
    if len(anchor_frames) == 0:
        return np.zeros(frame_count, dtype=np.float32)
    if len(anchor_frames) == 1:
        return np.full(frame_count, float(residuals[0]), dtype=np.float32)
    interp = PchipInterpolator(np.asarray(anchor_frames, dtype=np.float32), residuals.astype(np.float32), extrapolate=True)
    return interp(full).astype(np.float32)


def _anchor_corrected_motion(
    frame_count: int,
    anchor_frames: list[int],
    anchor_vals: np.ndarray,
    observed_frames: list[int],
    observed_vals: np.ndarray,
) -> np.ndarray:
    if len(observed_frames) >= 2:
        base = _linear_fill(frame_count, observed_frames, observed_vals)
        base = _smooth(base)
    elif len(observed_frames) == 1:
        base = np.full(frame_count, float(observed_vals[0]), dtype=np.float32)
    else:
        if len(anchor_frames) == 0:
            return np.full(frame_count, np.nan, dtype=np.float32)
        if len(anchor_frames) == 1:
            return np.full(frame_count, float(anchor_vals[0]), dtype=np.float32)
        interp = PchipInterpolator(np.asarray(anchor_frames, dtype=np.float32), anchor_vals.astype(np.float32), extrapolate=True)
        return interp(np.arange(frame_count, dtype=np.float32)).astype(np.float32)

    if len(anchor_frames) == 0:
        return base

    anchor_idx = np.asarray(anchor_frames, dtype=int)
    residual = anchor_vals.astype(np.float32) - base[anchor_idx]
    correction = _interp_residual(frame_count, anchor_frames, residual)
    refined = base + correction
    refined[anchor_idx] = anchor_vals.astype(np.float32)
    return refined.astype(np.float32)


def _smooth_sizes(frame_count: int, observed_frames: list[int], observed_vals: np.ndarray, default_value: float) -> np.ndarray:
    if len(observed_frames) == 0:
        return np.full(frame_count, default_value, dtype=np.float32)
    dense = _linear_fill(frame_count, observed_frames, observed_vals)
    return _smooth(dense, window=15, poly=2).astype(np.float32)


def _global_default_sizes(tracking_frames: dict[int, dict[int, dict[str, Any]]]) -> tuple[float, float]:
    widths: list[float] = []
    heights: list[float] = []
    for players in tracking_frames.values():
        for player in players.values():
            x1, y1, x2, y2 = [float(v) for v in player["bbox"]]
            widths.append(max(1.0, x2 - x1))
            heights.append(max(1.0, y2 - y1))
    if not widths or not heights:
        return 24.0, 58.0
    return float(np.median(widths)), float(np.median(heights))


def _build_refined_tracks(
    clip_points: dict[int, list[dict[str, Any]]],
    tracking_frames: dict[int, dict[int, dict[str, Any]]],
    frame_count: int,
) -> list[RefinedTrack]:
    object_ids = sorted({str(pt["id"]) for pts in clip_points.values() for pt in pts})
    default_w, default_h = _global_default_sizes(tracking_frames)
    refined: list[RefinedTrack] = []

    for obj_id in object_ids:
        team = _team_for_object(clip_points, obj_id)
        anchor_frames, anchor_x, anchor_y = _anchor_series(clip_points, obj_id)
        obs_frames, obs_x, obs_y, obs_w, obs_h = _observed_series(tracking_frames, obj_id)

        x = _anchor_corrected_motion(frame_count, anchor_frames, anchor_x, obs_frames, obs_x)
        y = _anchor_corrected_motion(frame_count, anchor_frames, anchor_y, obs_frames, obs_y)

        if len(obs_frames) > 0:
            w = _smooth_sizes(frame_count, obs_frames, obs_w, default_value=float(np.median(obs_w)))
            h = _smooth_sizes(frame_count, obs_frames, obs_h, default_value=float(np.median(obs_h)))
            source = "tracking+anchors"
            start = min(min(anchor_frames) if anchor_frames else obs_frames[0], obs_frames[0])
            end = max(max(anchor_frames) if anchor_frames else obs_frames[-1], obs_frames[-1])
            track_id = _track_num(obj_id) or _synthetic_track_num(obj_id)
        else:
            w = np.full(frame_count, default_w, dtype=np.float32)
            h = np.full(frame_count, default_h, dtype=np.float32)
            source = "anchors-only"
            start = min(anchor_frames) if anchor_frames else 0
            end = max(anchor_frames) if anchor_frames else frame_count - 1
            track_id = _synthetic_track_num(obj_id)

        visible = np.zeros(frame_count, dtype=bool)
        if end >= start:
            visible[start : end + 1] = True

        # Clamp to a believable broadcast-view player extent.
        w = np.clip(w, 12.0, 48.0)
        h = np.clip(h, 36.0, 118.0)

        refined.append(
            RefinedTrack(
                obj_id=obj_id,
                track_id=track_id,
                team=team,
                x=x,
                y=y,
                w=w,
                h=h,
                visible=visible,
                anchor_frames=anchor_frames,
                observed_frames=obs_frames,
                source=source,
            )
        )
    return refined


def _build_refined_ball(
    frame_count: int,
    tracking_full_frames: list[dict[str, Any]],
    annotated_ball: dict[int, dict[str, Any]],
) -> RefinedBall | None:
    obs_frames, obs_x, obs_y = _ball_series(tracking_full_frames)
    anchor_frames, anchor_x, anchor_y = _ball_anchor_series(annotated_ball)
    if len(obs_frames) == 0 and len(anchor_frames) == 0:
        return None
    x = _anchor_corrected_motion(frame_count, anchor_frames, anchor_x, obs_frames, obs_x)
    y = _anchor_corrected_motion(frame_count, anchor_frames, anchor_y, obs_frames, obs_y)
    start = min(anchor_frames[0] if anchor_frames else obs_frames[0], obs_frames[0] if obs_frames else anchor_frames[0])
    end = max(anchor_frames[-1] if anchor_frames else obs_frames[-1], obs_frames[-1] if obs_frames else anchor_frames[-1])
    visible = np.zeros(frame_count, dtype=bool)
    visible[start : end + 1] = True
    return RefinedBall(x=x, y=y, visible=visible, anchor_frames=anchor_frames, observed_frames=obs_frames)


def _trim_video(source_video: Path, output_video: Path, frame_limit: int) -> None:
    cap = cv2.VideoCapture(str(source_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    written = 0
    while written < frame_limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    try:
        import imageio_ffmpeg  # type: ignore
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        tmp = output_video.with_suffix(".h264.mp4")
        import subprocess

        subprocess.run(
            [ffmpeg, "-y", "-i", str(output_video), "-vcodec", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p", str(tmp)],
            check=True,
            capture_output=True,
        )
        tmp.replace(output_video)
    except Exception:
        pass


def _frame_dict_from_refined(
    frame_idx: int,
    t: float,
    refined_tracks: list[RefinedTrack],
    ball: RefinedBall | None,
) -> dict[str, Any]:
    players: list[dict[str, Any]] = []
    for track in refined_tracks:
        if frame_idx >= len(track.visible) or not track.visible[frame_idx]:
            continue
        x = float(track.x[frame_idx])
        y = float(track.y[frame_idx])
        w = float(track.w[frame_idx])
        h = float(track.h[frame_idx])
        player = {
            "track_id": int(track.track_id),
            "cls": 2,
            "team": track.team,
            "include_in_shape": True,
            "x": x,
            "y": y,
            "bbox": [
                round(x - w / 2.0, 3),
                round(y - h, 3),
                round(x + w / 2.0, 3),
                round(y, 3),
            ],
            "interpolated": frame_idx not in set(track.anchor_frames) and frame_idx not in set(track.observed_frames),
            "annotation_id": track.obj_id,
            "refined_source": track.source,
        }
        players.append(player)

    frame = {"frame": frame_idx, "t": t, "players": players}
    if ball is not None and frame_idx < len(ball.visible) and ball.visible[frame_idx]:
        frame["ball"] = {
            "x": round(float(ball.x[frame_idx]), 3),
            "y": round(float(ball.y[frame_idx]), 3),
        }
    return frame


def _refined_tracking_json(
    clip_id: str,
    video_path: Path,
    base_tracking: dict,
    refined_tracks: list[RefinedTrack],
    ball: RefinedBall | None,
    frame_count: int,
) -> dict[str, Any]:
    fps = float(base_tracking.get("fps", 25.0))
    frames = [
        _frame_dict_from_refined(i, round(i / fps, 3), refined_tracks, ball)
        for i in range(frame_count)
    ]
    return {
        "clip_id": clip_id,
        "video_path": str(video_path),
        "fps": fps,
        "vid_stride": 1,
        "model": "annotated_base_refined",
        "track_classes": [2],
        "confidence": None,
        "frame_count": frame_count,
        "frame_count_meta": frame_count,
        "frames": frames,
    }


def process_clip(
    clip_id: str,
    clip_points: dict[int, list[dict[str, Any]]],
    clip_ball: dict[int, dict[str, Any]],
    tracking_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    tracking_frames, tracking_obj = _tracking_index(tracking_path)
    tracking_full_frames = _full_tracking_frames(tracking_path)
    raw_frame_count = 1 + max(tracking_frames) if tracking_frames else len(tracking_full_frames)
    video_path = Path(tracking_obj["video_path"])

    refined_tracks = _build_refined_tracks(clip_points, tracking_frames, raw_frame_count)
    refined_ball = _build_refined_ball(raw_frame_count, tracking_full_frames, clip_ball)
    refined_tracking = _refined_tracking_json(
        clip_id=clip_id,
        video_path=video_path,
        base_tracking=tracking_obj,
        refined_tracks=refined_tracks,
        ball=refined_ball,
        frame_count=raw_frame_count,
    )

    refined_dir = output_dir / "refined_tracking"
    trimmed_dir = output_dir / "trimmed_clips"
    overlay_dir = output_dir / "overlays"
    metrics_dir = output_dir / "metrics"
    refined_dir.mkdir(parents=True, exist_ok=True)
    trimmed_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    refined_path = refined_dir / f"{clip_id}.json"
    trimmed_video = trimmed_dir / f"{clip_id}.mp4"
    overlay_path = overlay_dir / f"{clip_id}_overlay.mp4"
    metrics_path = metrics_dir / f"{clip_id}_metrics.json"

    refined_path.write_text(json.dumps(refined_tracking, indent=2, ensure_ascii=False), encoding="utf-8")
    _trim_video(video_path, trimmed_video, raw_frame_count)
    render_overlay(trimmed_video, refined_path, overlay_path)
    metrics = compute_metrics(refined_tracking)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "clip_id": clip_id,
        "base_tracking_path": str(tracking_path),
        "refined_tracking_path": str(refined_path),
        "trimmed_video_path": str(trimmed_video),
        "overlay_path": str(overlay_path),
        "metrics_path": str(metrics_path),
        "frame_count": raw_frame_count,
        "annotated_players": len(refined_tracks),
        "ball_available": refined_ball is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="psg_annotation_v1")
    parser.add_argument("--tracking-run", default="psg_tracking_latest_v1")
    parser.add_argument("--preset", default="football_players")
    parser.add_argument("--clip", dest="clips", action="append", default=[])
    parser.add_argument("--run-id", default="psg_annotated_base_convex_v1")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        manifest_path = EVAL_ANNOTATIONS_DIR / f"{args.manifest}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    player_points, ball_points = _parse_annotations(manifest_path)
    clip_ids = args.clips or sorted(player_points)
    output_dir = EVAL_RUNS_DIR / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for clip_id in clip_ids:
        tracking_path = EVAL_RUNS_DIR / args.tracking_run / args.preset / "tracking_with_teams" / f"{clip_id}.json"
        if not tracking_path.exists():
            raise FileNotFoundError(tracking_path)
        print(f"[annotated-base] refining {clip_id} from {args.preset}")
        summaries.append(
            process_clip(
                clip_id=clip_id,
                clip_points=player_points[clip_id],
                clip_ball=ball_points.get(clip_id, {}),
                tracking_path=tracking_path,
                output_dir=output_dir,
            )
        )

    summary = {
        "run_id": args.run_id,
        "manifest_path": str(manifest_path),
        "tracking_run": args.tracking_run,
        "preset": args.preset,
        "clips": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[annotated-base] wrote summary -> {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
