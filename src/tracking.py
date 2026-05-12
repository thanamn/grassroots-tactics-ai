"""Run YOLOv8 detection + centroid tracker on a clip, save tracking JSON.

Usage:
    python -m src.tracking --input data/clips/sample.mp4

Output: data/tracking/<clip_stem>.json

Notes
-----
- Team assignment is NOT done here. Each player gets a numeric track_id only.
  Edit the JSON afterwards to label each track_id as "A" or "B". Manual
  labelling of 2 clips takes ~2 min each — fine for Wizard-of-Oz.
- We use model.predict() (detection-only) + a small motion-aware tracker
  instead of model.track() (ByteTrack). ByteTrack's lap DLL crashes silently
  on this Windows environment after the first frame. The local tracker uses
  optical flow, predicted boxes, centre-distance gating, and bridge frames to
  avoid ID fragmentation when YOLO misses a running player for a moment.
- Player (x, y) is the bottom-centre of the bounding box, not the geometric
  centre. Bottom-centre approximates where the player's feet contact the
  pitch — the ground-plane point that spacing/hull metrics actually want to
  model. Centroid would over-weight tall players and skew the hull whenever
  bbox heights vary (camera zoom, motion blur).
- Class filtering is delegated to the football-pretrained model: TRACK_CLASSES
  picks only goalkeeper + player IDs (1, 2). The previous COCO-based pipeline
  needed an extra confidence and bbox-area filter to drop spectators/refs/
  graphics — the football-specific weights don't detect those, so the filter
  is gone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

import math

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from src.config import (
    CLIPS_DIR,
    DEFAULT_FPS,
    TRACK_CLASSES,
    TRACKING_DIR,
    YOLO_MODEL,
)

# Auto-detect CUDA. On the dev laptop (RTX 4070) this gives ~10–15× the
# throughput of CPU inference; on a machine without an NVIDIA GPU this
# silently falls back to "cpu" rather than failing. We deliberately do
# NOT raise on CPU — the fallback is the documented mode for non-GPU
# users (Colab, paper-writing on a laptop without a GPU, etc.).
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ── motion-aware tracker ────────────────────────────────────────────────────

def _iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """IoU between two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (aA + aB - inter)


def _centre(box: np.ndarray) -> np.ndarray:
    """Centre point for a box [x1,y1,x2,y2]."""
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=float)


def _diag(box: np.ndarray) -> float:
    """Diagonal length for a box, with a small floor for tiny/partial boxes."""
    w = max(1.0, float(box[2] - box[0]))
    h = max(1.0, float(box[3] - box[1]))
    return float(math.hypot(w, h))


def _clip_box(box: np.ndarray, frame_shape: tuple[int, int] | None) -> np.ndarray:
    """Clamp a box to frame boundaries while preserving positive size."""
    if frame_shape is None:
        return np.asarray(box, dtype=float)
    h, w = frame_shape
    x1 = float(np.clip(box[0], 0, max(1, w - 2)))
    y1 = float(np.clip(box[1], 0, max(1, h - 2)))
    x2 = float(np.clip(box[2], x1 + 1, max(2, w - 1)))
    y2 = float(np.clip(box[3], y1 + 1, max(2, h - 1)))
    return np.array([x1, y1, x2, y2], dtype=float)


def _points_in_box(gray: np.ndarray, box: np.ndarray,
                   max_points: int = 18) -> np.ndarray | None:
    """Seed optical-flow points from the central player box region."""
    h_img, w_img = gray.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, (h_img, w_img))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 4 or bh < 8:
        return None

    # Use the central body area rather than the whole box; the full box often
    # includes grass, which optical flow can happily follow instead of the
    # player.
    ix1 = int(max(0, x1 + bw * 0.15))
    ix2 = int(min(w_img - 1, x2 - bw * 0.15))
    iy1 = int(max(0, y1 + bh * 0.10))
    iy2 = int(min(h_img - 1, y2 - bh * 0.08))
    if ix2 <= ix1 + 2 or iy2 <= iy1 + 2:
        return None

    roi = gray[iy1:iy2, ix1:ix2]
    pts = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=3,
        blockSize=3,
    )
    if pts is not None and len(pts) >= 4:
        pts = pts.reshape(-1, 2)
        pts[:, 0] += ix1
        pts[:, 1] += iy1
        return pts.reshape(-1, 1, 2).astype(np.float32)

    # Tiny distant players often lack enough corners. A small interior grid
    # gives LK a chance to follow jersey/edge texture without adding many
    # points.
    xs = np.linspace(ix1 + 1, ix2 - 1, num=3)
    ys = np.linspace(iy1 + 1, iy2 - 1, num=4)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    return grid.reshape(-1, 1, 2)


class _CentroidTracker:
    """Lightweight motion-aware tracker; no native DLLs required.

    The first implementation matched only by IoU. That fragments football
    tracking badly because a player missed for even one frame has no overlap
    with the stale box once the camera pans or the player sprints. This keeps a
    per-track optical-flow points and velocity estimate, matches by predicted
    IoU plus centre distance, and emits moving bridge positions while a
    confirmed track is temporarily lost.
    """

    def __init__(self, iou_thresh: float = 0.10, max_lost: int = 40,
                 max_emit_lost: int = 7, max_flow_emit_lost: int = 25,
                 centre_gate_px: float = 90.0,
                 centre_gate_scale: float = 1.75, min_confirmed_hits: int = 2):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.max_emit_lost = max_emit_lost
        self.max_flow_emit_lost = max_flow_emit_lost
        self.centre_gate_px = centre_gate_px
        self.centre_gate_scale = centre_gate_scale
        self.min_confirmed_hits = min_confirmed_hits
        self._next_id = 1
        self._tracks: dict[int, dict] = {}
        self._prev_gray: np.ndarray | None = None
        self._frame_shape: tuple[int, int] | None = None

    def _dominant(self, track: dict) -> int:
        total = track.get("total_count", 1)
        gk = track.get("gk_count", 0)
        # Require ≥5 frames before committing to GK label — prevents a
        # linesman seen for 1–3 frames from getting a GK marker.
        return 1 if total >= 5 and gk / total > 0.50 else 2

    def _predicted_box(self, track: dict, steps: int | None = None) -> np.ndarray:
        steps = track.get("lost", 0) + 1 if steps is None else steps
        return np.asarray(track["box"], dtype=float) + np.asarray(track["velocity"], dtype=float) * steps

    def _seed_points(self, tid: int, gray: np.ndarray | None) -> None:
        if gray is None or tid not in self._tracks:
            return
        self._tracks[tid]["points"] = _points_in_box(gray, self._tracks[tid]["box"])

    def _new_track(self, box: np.ndarray, cls: int, gray: np.ndarray | None = None) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "box": _clip_box(np.asarray(box, dtype=float), self._frame_shape),
            "velocity": np.zeros(4, dtype=float),
            "lost": 0,
            "hits": 1,
            "flow_ok": False,
            "points": None,
            "gk_count": int(cls == 1),
            "total_count": 1,
        }
        self._seed_points(tid, gray)
        return tid

    def _advance_tracks_by_flow(self, frame: np.ndarray | None) -> np.ndarray | None:
        """Move existing tracks from the previous frame to this frame."""
        if frame is None:
            for track in self._tracks.values():
                velocity = np.asarray(track.get("velocity", np.zeros(4)), dtype=float)
                if track.get("lost", 0) > 0 or np.linalg.norm(velocity[:2]) > 0:
                    track["box"] = np.asarray(track["box"], dtype=float) + velocity
                track["flow_ok"] = False
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self._frame_shape = gray.shape[:2]
        if self._prev_gray is None:
            for tid in self._tracks:
                self._seed_points(tid, gray)
                self._tracks[tid]["flow_ok"] = False
            return gray

        for tid, track in self._tracks.items():
            old_box = np.asarray(track["box"], dtype=float)
            flow_ok = False
            pts = track.get("points")
            if pts is not None and len(pts) >= 3:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray,
                    gray,
                    pts.astype(np.float32),
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                )
                if next_pts is not None and status is not None:
                    ok = status.reshape(-1) == 1
                    old_pts = pts.reshape(-1, 2)[ok]
                    new_pts = next_pts.reshape(-1, 2)[ok]
                    if len(new_pts) >= 3:
                        deltas = new_pts - old_pts
                        med = np.median(deltas, axis=0)
                        residual = np.linalg.norm(deltas - med, axis=1)
                        keep = residual <= max(5.0, float(np.percentile(residual, 75)) * 2.0)
                        if int(np.sum(keep)) >= 3:
                            new_pts = new_pts[keep]
                            deltas = deltas[keep]
                            med = np.median(deltas, axis=0)
                        max_step = max(70.0, _diag(old_box) * 1.25)
                        if float(np.linalg.norm(med)) <= max_step:
                            shift = np.array([med[0], med[1], med[0], med[1]], dtype=float)
                            track["box"] = _clip_box(old_box + shift, self._frame_shape)
                            track["velocity"] = (
                                np.asarray(track.get("velocity", np.zeros(4)), dtype=float) * 0.35
                                + shift * 0.65
                            )
                            track["points"] = new_pts.reshape(-1, 1, 2).astype(np.float32)
                            flow_ok = True

            if not flow_ok:
                velocity = np.asarray(track.get("velocity", np.zeros(4)), dtype=float)
                if track.get("lost", 0) > 0 or np.linalg.norm(velocity[:2]) > 0:
                    track["box"] = _clip_box(old_box + velocity, self._frame_shape)
                if track.get("lost", 0) == 0:
                    self._seed_points(tid, gray)
            track["flow_ok"] = flow_ok

        return gray

    def _finish_frame(self, gray: np.ndarray | None) -> None:
        if gray is not None:
            self._prev_gray = gray

    def _lost_results(self, suppress_boxes: np.ndarray | None = None) -> list[tuple[int, np.ndarray, int, bool]]:
        result = []
        for tid, track in self._tracks.items():
            emit_limit = self.max_flow_emit_lost if track.get("flow_ok") else self.max_emit_lost
            if 0 < track["lost"] <= emit_limit and track.get("hits", 0) >= self.min_confirmed_hits:
                pred_box = np.asarray(track["box"], dtype=float)
                if suppress_boxes is not None and len(suppress_boxes) > 0:
                    pred_centre = _centre(pred_box)
                    too_close_to_detection = False
                    for det_box in suppress_boxes:
                        near_gate = max(25.0, 0.75 * max(_diag(pred_box), _diag(det_box)))
                        if float(np.linalg.norm(pred_centre - _centre(det_box))) <= near_gate:
                            too_close_to_detection = True
                            break
                    if too_close_to_detection:
                        continue
                result.append((tid, pred_box, self._dominant(track), True))
        return result

    def update(self, boxes: np.ndarray,
               classes: np.ndarray | None = None,
               frame: np.ndarray | None = None) -> list[tuple[int, np.ndarray, int, bool]]:
        """Match detections to tracks; return [(track_id, box, dominant_cls, interpolated), ...].

        ``classes`` is a parallel array of YOLO class IDs (1=GK, 2=player).
        We accumulate class votes per track across frames so a player who is
        occasionally mis-classified for one frame keeps its stable label.
        Returns ``dominant_cls=1`` when >40 % of votes were class-1 (GK).
        """
        if classes is None:
            classes = np.full(len(boxes), 2, dtype=int)

        gray = self._advance_tracks_by_flow(frame)

        if len(boxes) == 0:
            for t in self._tracks.values():
                t["lost"] += 1
            self._tracks = {k: v for k, v in self._tracks.items()
                            if v["lost"] <= self.max_lost}
            result = self._lost_results()
            self._finish_frame(gray)
            return result

        track_ids = list(self._tracks.keys())
        if not track_ids:
            result = []
            for box, cls in zip(boxes, classes):
                tid = self._new_track(box, int(cls), gray)
                result.append((tid, box, int(cls), False))
            self._finish_frame(gray)
            return result

        # Build cost matrix from predicted IoU and centre distance. Distance
        # gates grow gently with lost age so a player can be reacquired after a
        # short occlusion without allowing wild cross-pitch ID swaps.
        track_boxes = np.array([
            np.asarray(self._tracks[i]["box"], dtype=float)
            if gray is not None else self._predicted_box(self._tracks[i])
            for i in track_ids
        ])
        cost = np.full((len(track_ids), len(boxes)), 1e6, dtype=float)
        for ti, tb in enumerate(track_boxes):
            track = self._tracks[track_ids[ti]]
            tc = _centre(tb)
            gate = max(self.centre_gate_px, self.centre_gate_scale * _diag(tb))
            gate *= 1.0 + 0.35 * track.get("lost", 0)
            for di, db in enumerate(boxes):
                iou = _iou(tb, db)
                centre_dist = float(np.linalg.norm(tc - _centre(db)))
                if iou < self.iou_thresh and centre_dist > gate:
                    continue
                centre_cost = min(centre_dist / gate, 1.0)
                cost[ti, di] = 0.65 * (1.0 - iou) + 0.35 * centre_cost

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_tracks = set(); matched_dets = set()
        result = []

        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] >= 1e5:
                continue
            tid = track_ids[ri]
            old = self._tracks[tid]
            new_cls = int(classes[ci])
            gap = old.get("lost", 0) + 1
            observed_velocity = (boxes[ci] - old["box"]) / max(1, gap)
            velocity = old.get("velocity", np.zeros(4, dtype=float)) * 0.80 + observed_velocity * 0.20
            self._tracks[tid] = {
                "box": _clip_box(np.asarray(boxes[ci], dtype=float), self._frame_shape),
                "velocity": velocity,
                "lost": 0,
                "hits": old.get("hits", 1) + 1,
                "flow_ok": old.get("flow_ok", False),
                "points": old.get("points"),
                "gk_count": old.get("gk_count", 0) + (1 if new_cls == 1 else 0),
                "total_count": old.get("total_count", 0) + 1,
            }
            self._seed_points(tid, gray)
            result.append((tid, boxes[ci], self._dominant(self._tracks[tid]), False))
            matched_tracks.add(ri); matched_dets.add(ci)

        # Unmatched detections → new tracks
        for ci, (box, cls) in enumerate(zip(boxes, classes)):
            if ci not in matched_dets:
                tid = self._new_track(box, int(cls), gray)
                result.append((tid, box, int(cls), False))

        # Unmatched tracks → increment lost counter
        for ri, tid in enumerate(track_ids):
            if ri not in matched_tracks:
                self._tracks[tid]["lost"] += 1
        self._tracks = {k: v for k, v in self._tracks.items()
                        if v["lost"] <= self.max_lost}
        result.extend(self._lost_results(boxes))
        self._finish_frame(gray)
        return result


def _player_point(player: dict) -> np.ndarray:
    return np.array([float(player["x"]), float(player["y"])], dtype=float)


def _interp_player(prev: dict, nxt: dict, alpha: float) -> dict:
    prev_box = np.asarray(prev["bbox"], dtype=float)
    next_box = np.asarray(nxt["bbox"], dtype=float)
    box = prev_box + (next_box - prev_box) * alpha
    x1, y1, x2, y2 = box
    cls = prev.get("cls", 2) if prev.get("cls") == nxt.get("cls") else prev.get("cls", 2)
    return {
        "track_id": int(prev["track_id"]),
        "cls": int(cls),
        "team": None,
        "interpolated": True,
        "interpolation": "gap",
        "x": float((x1 + x2) / 2.0),
        "y": float(y2),
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
    }


def _has_nearby_player(players: list[dict], candidate: dict) -> bool:
    for player in players:
        if int(player.get("track_id", -1)) == int(candidate["track_id"]):
            return True
    return False


def _interpolate_player_gaps(frames: list[dict], fps: float, vid_stride: int) -> int:
    """Fill short within-track player gaps after the whole clip is known.

    Optical flow helps during the forward pass, but the offline JSON can do one
    thing a live tracker cannot: use a later reappearance of the same track to
    bridge the missing frames. This prevents the overlay from blinking a runner
    out during short detector dropouts while still leaving leading/trailing
    off-screen gaps empty.

    Runs after track stitching, so most ID-switch gaps have already been
    merged onto a single track_id and become eligible for in-track interp.
    """
    if not frames:
        return 0

    # Bridge gaps up to 2 s long — matches the stitcher's window so the two
    # passes share one budget. Track stitching has already merged most
    # split tracks; this fills the *within-track* gaps the tracker did not
    # emit a bridge for (e.g. occlusion longer than max_flow_emit_lost).
    max_gap_frames = max(5, int(round(fps * 2.0)))
    # Cap per-source-frame jump to ~38 px. A genuine sprint covers roughly
    # 25 px/source-frame in broadcast pixel space; the headroom absorbs
    # camera panning. NOTE this is per *source* frame, not per processed
    # step — the previous formulation multiplied by ``vid_stride`` and
    # silently over-permitted at higher strides.
    max_per_frame_px = 38.0
    frames_by_no = {int(frame["frame"]): frame for frame in frames}
    track_obs: dict[int, list[tuple[int, dict]]] = {}

    for frame in frames:
        frame_no = int(frame["frame"])
        for player in frame.get("players", []):
            if player.get("interpolation") == "gap":
                continue
            track_obs.setdefault(int(player["track_id"]), []).append((frame_no, player))

    inserted = 0
    for tid, observations in track_obs.items():
        observations.sort(key=lambda item: item[0])
        for (f0, p0), (f1, p1) in zip(observations, observations[1:]):
            gap = f1 - f0
            if gap <= vid_stride or gap > max_gap_frames:
                continue
            p0_xy = _player_point(p0)
            p1_xy = _player_point(p1)
            if float(np.linalg.norm(p1_xy - p0_xy)) / max(1, gap) > max_per_frame_px:
                continue

            frame_no = f0 + vid_stride
            while frame_no < f1:
                frame = frames_by_no.get(frame_no)
                if frame is not None:
                    alpha = (frame_no - f0) / gap
                    candidate = _interp_player(p0, p1, alpha)
                    players = frame.setdefault("players", [])
                    if not _has_nearby_player(players, candidate):
                        players.append(candidate)
                        inserted += 1
                frame_no += vid_stride

    if inserted:
        for frame in frames:
            frame["players"].sort(key=lambda player: int(player["track_id"]))
    return inserted


def _stitch_broken_tracks(frames: list[dict], fps: float,
                          max_gap_s: float = 2.0,
                          max_dist_scale: float = 1.8) -> int:
    """Merge tracks that look like the same player split by an ID switch.

    The centroid tracker spawns a new ID whenever a player is lost beyond its
    re-emit window. Most splits are short — a missed YOLO detection during
    fast motion, a brief occlusion behind another player, a sudden camera
    pan. Stitching them back is what keeps the per-frame hull stable: every
    unrepaired ID swap looks to ``src/metrics.py`` like the player momentarily
    vanished and can fire a false ``compactness_spike`` event.

    The pass is intentionally conservative because the cost of fusing two
    different players is much worse than the cost of leaving a hard case
    unstitched (which downstream interpolation may still bridge anyway):

    - Only stitch when the gap between A's last frame and B's first frame is
      ≤ ``max_gap_s`` seconds (capped at 60 source frames absolute).
    - Only stitch when B's start position lies within
      ``max_dist_scale × avg_bbox_diag`` of A's predicted endpoint (last
      observed position + estimated velocity × gap).
    - Tracks must share the same dominant class (no GK ↔ outfield merges).
    - When two candidate predecessors compete for the same successor with
      similar scores (runner-up within 1.5× best), neither is stitched.
      Ambiguous succession is exactly the case where we are most likely to
      fuse two different players.

    Returns the number of stitches applied (i.e. number of merged endings).
    """
    if not frames:
        return 0

    obs: dict[int, list[tuple[int, dict]]] = {}
    for frame in frames:
        fno = int(frame["frame"])
        for p in frame.get("players", []):
            if p.get("interpolation") == "gap":
                continue
            obs.setdefault(int(p["track_id"]), []).append((fno, p))
    for tid in obs:
        obs[tid].sort(key=lambda item: item[0])

    max_gap_frames = min(60, int(round(fps * max_gap_s)))

    def _vel(track_obs: list[tuple[int, dict]]) -> tuple[float, float]:
        if len(track_obs) < 2:
            return 0.0, 0.0
        tail = track_obs[-min(5, len(track_obs)):]
        f0, p0 = tail[0]
        f1, p1 = tail[-1]
        df = max(1, f1 - f0)
        return (p1["x"] - p0["x"]) / df, (p1["y"] - p0["y"]) / df

    def _diag(player: dict) -> float:
        x1, y1, x2, y2 = player["bbox"]
        return max(8.0, math.hypot(x2 - x1, y2 - y1))

    def _cls(track_obs: list[tuple[int, dict]]) -> int:
        n_gk = sum(1 for _, p in track_obs if int(p.get("cls", 2)) == 1)
        return 1 if len(track_obs) >= 3 and n_gk >= len(track_obs) * 0.5 else 2

    track_ids = list(obs.keys())
    track_start = {tid: obs[tid][0][0] for tid in track_ids}
    track_end = {tid: obs[tid][-1][0] for tid in track_ids}

    # successor_tid -> [(predecessor_tid, score)]
    cands: dict[int, list[tuple[int, float]]] = {}

    for a_tid in track_ids:
        a_obs = obs[a_tid]
        a_end_player = a_obs[-1][1]
        a_cls = _cls(a_obs)
        vx, vy = _vel(a_obs)
        a_x, a_y = a_end_player["x"], a_end_player["y"]
        a_diag = _diag(a_end_player)

        for b_tid in track_ids:
            if b_tid == a_tid:
                continue
            gap = track_start[b_tid] - track_end[a_tid]
            if gap <= 0 or gap > max_gap_frames:
                continue
            b_obs = obs[b_tid]
            if _cls(b_obs) != a_cls:
                continue
            b_first = b_obs[0][1]
            avg_diag = (a_diag + _diag(b_first)) / 2.0
            pred_x = a_x + vx * gap
            pred_y = a_y + vy * gap
            dist = math.hypot(b_first["x"] - pred_x, b_first["y"] - pred_y)
            if dist > max_dist_scale * avg_diag:
                continue
            # Lower score = better. Distance dominates; small gap penalty
            # breaks ties toward earlier reacquisition.
            cands.setdefault(b_tid, []).append((a_tid, dist + 4.0 * gap))

    parent: dict[int, int] = {}
    used_predecessors: set[int] = set()

    # Process successors in start-frame order so chains build left to right.
    for b_tid in sorted(cands.keys(), key=lambda t: track_start[t]):
        ranked = sorted(cands[b_tid], key=lambda item: item[1])
        best: tuple[int, float] | None = None
        runner_score: float | None = None
        for a_tid, score in ranked:
            if a_tid in used_predecessors:
                continue
            if best is None:
                best = (a_tid, score)
            else:
                runner_score = score
                break
        if best is None:
            continue
        # Refuse ambiguous matches: if the runner-up is within 1.5× of the
        # best score, leave the case unstitched. The downstream gap-fill
        # may still bridge within-track gaps, and we have not actively
        # made the wrong call.
        if runner_score is not None and runner_score < best[1] * 1.5:
            continue
        a_tid = best[0]
        root = a_tid
        while root in parent and parent[root] != root:
            root = parent[root]
        parent[b_tid] = root
        used_predecessors.add(a_tid)

    if not parent:
        return 0

    def _resolve(tid: int) -> int:
        r = tid
        while r in parent and parent[r] != r:
            r = parent[r]
        return r

    for frame in frames:
        for p in frame.get("players", []):
            tid = int(p["track_id"])
            new_tid = _resolve(tid)
            if new_tid != tid:
                p["track_id"] = new_tid

    return len(parent)


def _consolidate_track_cls(frames: list[dict]) -> int:
    """Rewrite every frame's ``cls`` to the (merged) track's dominant class.

    The live tracker can't commit a track to GK until it has accumulated
    ≥5 frames and >50 % cls=1 evidence. So the first 1–4 frames of every
    goalkeeper track go in as cls=2, then flip once the votes pile up.
    The downstream team-assignment stage reads cls per-frame, so without
    this consolidation those early frames slip into the outfield team and
    pollute the convex hull / spread metrics.

    Runs after track stitching so the vote is taken over the full merged
    track, which is also wider evidence than any one of the sub-tracks
    would have had on its own.

    Returns the number of frames where ``cls`` was actually rewritten.
    """
    votes: dict[int, dict[int, int]] = {}
    for frame in frames:
        for p in frame.get("players", []):
            tid = int(p["track_id"])
            c = int(p.get("cls", 2))
            counts = votes.setdefault(tid, {1: 0, 2: 0})
            counts[c] = counts.get(c, 0) + 1

    final_cls: dict[int, int] = {}
    for tid, counts in votes.items():
        n_gk = counts.get(1, 0)
        total = n_gk + counts.get(2, 0)
        # Same decision rule as the live tracker's ``_dominant``, applied
        # retroactively to the merged track's full vote.
        final_cls[tid] = 1 if total >= 5 and n_gk / total > 0.50 else 2

    rewritten = 0
    for frame in frames:
        for p in frame.get("players", []):
            tid = int(p["track_id"])
            target = final_cls.get(tid)
            if target is None:
                continue
            if int(p.get("cls", 2)) != target:
                p["cls"] = target
                rewritten += 1
    return rewritten



def run_tracking(video_path: Path, model_name: str | None = None,
                 vid_stride: int = 1,
                 track_classes: list[int] | None = None,
                 confidence: float | None = None,
                 player_classes: list[int] | None = None,
                 ball_classes: list[int] | None = None,
                 goalkeeper_classes: list[int] | None = None,
                 progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict:
    """Detect + track persons across a video, return JSON-serialisable dict.

    Parameters
    ----------
    model_name : str | None
        Path/name of the YOLO weights. ``None`` falls back to ``YOLO_MODEL``
        from config. The upload pipeline keeps the football-specific model
        for long clips and increases ``vid_stride`` when it needs speed.
    vid_stride : int
        Process every ``vid_stride``-th frame of the video. ``1`` (default)
        keeps full temporal resolution; long clips use stride 3–5 and let
        the visualiser's smoothing window stretch to cover the gaps.
        Ultralytics' built-in ``vid_stride`` is fed straight through so
        we don't pay the cost of decoding skipped frames.
    track_classes : list[int] | None
        Optional class IDs for benchmark presets. ``None`` uses the production
        default from config.
    confidence : float | None
        Optional detector confidence override for model comparison runs.
    player_classes, ball_classes, goalkeeper_classes : list[int] | None
        Optional source-model class maps. Football-finetuned checkpoints do
        not all use the same class order, so benchmark presets pass these
        explicitly. Player output classes are normalized to 1=GK and
        2=outfield player for downstream team-assignment compatibility.
    """
    model_name = model_name or YOLO_MODEL
    track_classes = TRACK_CLASSES if track_classes is None else track_classes
    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frame_count_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    processed_fps = fps / max(1, vid_stride)
    tracker = _CentroidTracker(
        max_lost=max(10, int(round(processed_fps * 1.0))),
        max_emit_lost=max(1, min(3, int(round(processed_fps * 0.15)))),
        max_flow_emit_lost=max(5, min(15, int(round(processed_fps * 0.60)))),
    )

    model_name_l = str(model_name).lower()
    if player_classes is None and ball_classes is None and "football" in model_name_l:
        player_classes = [1, 2]
        ball_classes = [0]
        goalkeeper_classes = [1] if goalkeeper_classes is None else goalkeeper_classes
    elif player_classes is None:
        player_classes = track_classes
    if ball_classes is None:
        ball_classes = []
    if goalkeeper_classes is None:
        goalkeeper_classes = []

    player_classes = list(player_classes)
    ball_classes = list(ball_classes)
    goalkeeper_classes = list(goalkeeper_classes)

    gpu_name = torch.cuda.get_device_name(0) if DEVICE != 'cpu' else 'CPU only'
    print(f"[tracking] device={DEVICE!r} ({gpu_name}) "
          f"model={model_name} vid_stride={vid_stride} "
          f"players={player_classes} ball={ball_classes}")
    predict_kwargs = {
        "source": str(video_path),
        "classes": track_classes,
        "stream": True,
        "verbose": False,
        "device": DEVICE,
        "vid_stride": vid_stride,
    }
    if confidence is not None:
        predict_kwargs["conf"] = confidence
    results = model.predict(**predict_kwargs)

    frames: list[dict] = []
    ball_raw: list[dict | None] = []   # one entry per processed frame, None = not seen
    started = time.perf_counter()
    total_processed_est = max(1, math.ceil(frame_count_meta / max(1, vid_stride))) if frame_count_meta else 1

    # When vid_stride > 1, ultralytics yields one result per *processed*
    # frame, so the loop index i maps back to the original video frame
    # number via ``i * vid_stride``. We store the original frame number
    # so downstream stages keep their existing per-frame timestamp logic.
    for i, result in enumerate(results):
        frame_idx = i * vid_stride

        # Split detections using the source model's explicit class map.
        if result.boxes is not None and len(result.boxes) > 0:
            det_cls  = result.boxes.cls.cpu().numpy().astype(int)
            xyxy = result.boxes.xyxy.cpu().numpy()
            player_mask   = np.isin(det_cls, player_classes)
            player_boxes  = xyxy[player_mask]
            source_player_classes = det_cls[player_mask]
            normalized_player_classes = np.array([
                1 if int(cls) in goalkeeper_classes else 2
                for cls in source_player_classes
            ], dtype=int)
            ball_boxes = xyxy[np.isin(det_cls, ball_classes)]
        else:
            player_boxes = np.empty((0, 4))
            normalized_player_classes = np.empty((0,), dtype=int)
            ball_boxes = np.empty((0, 4))

        tracks = tracker.update(player_boxes, normalized_player_classes, frame=result.orig_img)
        players = []
        for tid, (x1, y1, x2, y2), dominant_cls, interpolated in tracks:
            players.append({
                "track_id": int(tid),
                "cls":  int(dominant_cls),   # 1=GK, 2=outfield player
                "team": None,
                "interpolated": bool(interpolated),
                # Bottom-centre = where the player's feet meet the pitch.
                # See module docstring for why this beats geometric centroid
                # for spacing/hull metrics.
                "x": float((x1 + x2) / 2),
                "y": float(y2),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        # Ball: take highest-confidence detection (first box) when visible.
        if len(ball_boxes) > 0:
            bx1, by1, bx2, by2 = ball_boxes[0]
            ball_raw.append({"x": float((bx1 + bx2) / 2), "y": float((by1 + by2) / 2)})
        else:
            ball_raw.append(None)

        frames.append({"frame": frame_idx, "t": frame_idx / fps, "players": players})

        # Print every 100 processed frames, not every 100 source frames.
        if i % 100 == 0:
            print(f"  frame {frame_idx}/{frame_count_meta} — {len(players)} players")
            if progress_callback:
                processed = i + 1
                elapsed = max(0.001, time.perf_counter() - started)
                actual_fps = processed / elapsed
                remaining_processed = max(0, total_processed_est - processed)
                progress_callback({
                    "processed": processed,
                    "total": total_processed_est,
                    "progress": min(0.94, (processed / total_processed_est) * 0.94),
                    "elapsed_s": elapsed,
                    "estimated_remaining_s": (remaining_processed / max(0.001, actual_fps)) + max(3.0, elapsed * 0.06),
                })

    # Interpolate short ball-detection gaps (e.g. brief occlusions) so
    # downstream possession metrics don't see artificial dropouts. Gaps longer
    # than BALL_LOST_PATIENCE frames stay null — the ball is genuinely lost.
    BALL_LOST_PATIENCE = 30
    _nan = {"x": float("nan"), "y": float("nan")}
    if ball_raw and any(b is not None for b in ball_raw):
        # Replace None with NaN-dict so pandas gets a uniform list of dicts.
        df_ball = pd.DataFrame([b if b is not None else _nan for b in ball_raw])
        # Only interpolate within runs of detections — don't fill leading/
        # trailing nulls (ball off-screen). limit= caps gap fill at patience.
        df_ball = df_ball.interpolate(method="linear", limit=BALL_LOST_PATIENCE)
        ball_interp = df_ball.to_dict("records")
    else:
        ball_interp = [None] * len(ball_raw)

    for idx, frame in enumerate(frames):
        b = ball_interp[idx] if idx < len(ball_interp) else None
        if b and not math.isnan(b.get("x", float("nan"))):
            frame["ball"] = b
        else:
            frame["ball"] = None

    # Three-pass offline cleanup, in order:
    #   1. Stitching merges ID-switched fragments of the same player so the
    #      subsequent gap-fill has a single track to interpolate within and
    #      the visualizer's smoothing window stays attached to one ID.
    #   2. cls consolidation backfills the early-track frames of every
    #      goalkeeper so they don't leak into the outfield assignment.
    #   3. Within-track gap interpolation fills the remaining holes for
    #      cases where the tracker emitted no bridge (long occlusion).
    if progress_callback:
        elapsed = max(0.001, time.perf_counter() - started)
        progress_callback({
            "processed": total_processed_est,
            "total": total_processed_est,
            "progress": 0.96,
            "elapsed_s": elapsed,
            "estimated_remaining_s": max(4.0, elapsed * 0.04),
        })
    stitched_tracks = _stitch_broken_tracks(frames, fps)
    cls_corrections = _consolidate_track_cls(frames)
    player_gap_interpolations = _interpolate_player_gaps(frames, fps, vid_stride)
    if progress_callback:
        elapsed = max(0.001, time.perf_counter() - started)
        progress_callback({
            "processed": total_processed_est,
            "total": total_processed_est,
            "progress": 1.0,
            "elapsed_s": elapsed,
            "estimated_remaining_s": 0.0,
        })
    print(
        f"[tracking] post-pass: stitched={stitched_tracks} "
        f"cls_corrected={cls_corrections} gap_interp={player_gap_interpolations}"
    )

    return {
        "clip_id": video_path.stem,
        "video_path": str(video_path),
        "fps": fps,
        "vid_stride": vid_stride,
        "model": model_name,
        "track_classes": track_classes,
        "player_classes": player_classes,
        "ball_classes": ball_classes,
        "goalkeeper_classes": goalkeeper_classes,
        "confidence": confidence,
        "stitched_tracks": stitched_tracks,
        "cls_corrections": cls_corrections,
        "player_gap_interpolations": player_gap_interpolations,
        "frame_count": len(frames),
        "frame_count_meta": frame_count_meta,
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to video clip (mp4 etc.)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        alt = CLIPS_DIR / video_path.name
        if alt.exists():
            video_path = alt
        else:
            raise FileNotFoundError(video_path)

    output_path = (Path(args.output) if args.output
                   else TRACKING_DIR / f"{video_path.stem}.json")
    data = run_tracking(video_path)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {output_path} — {data['frame_count']} frames")


if __name__ == "__main__":
    main()
