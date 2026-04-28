"""Run YOLOv8 detection + centroid tracker on a clip, save tracking JSON.

Usage:
    python -m src.tracking --input data/clips/sample.mp4

Output: data/tracking/<clip_stem>.json

Notes
-----
- Team assignment is NOT done here. Each player gets a numeric track_id only.
  Edit the JSON afterwards to label each track_id as "A" or "B". Manual
  labelling of 2 clips takes ~2 min each — fine for Wizard-of-Oz.
- We use model.predict() (detection-only) + a simple IoU-based centroid
  tracker instead of model.track() (ByteTrack). ByteTrack's lap DLL crashes
  silently on this Windows environment after the first frame. The centroid
  tracker achieves the same result for short broadcast clips and requires
  only scipy — no native extensions with manifest issues.
- Bounding-box centroid is used as player (x, y). Bottom-centre would be
  more accurate (ground-plane) and is a v2 TODO.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from src.config import (
    BORDER_BOT,
    BORDER_TOP,
    BORDER_X,
    CLIPS_DIR,
    DEFAULT_FPS,
    PERSON_CLASS_ID,
    TRACKING_DIR,
    YOLO_MODEL,
)


# ── centroid tracker ────────────────────────────────────────────────────────

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


class _CentroidTracker:
    """Lightweight IoU-based tracker; no native DLLs required."""

    def __init__(self, iou_thresh: float = 0.25, max_lost: int = 10):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self._next_id = 1
        self._tracks: dict[int, dict] = {}  # id → {box, lost}

    def update(self, boxes: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Match detections to existing tracks; return [(track_id, box), ...]."""
        if len(boxes) == 0:
            for t in self._tracks.values():
                t["lost"] += 1
            self._tracks = {k: v for k, v in self._tracks.items()
                            if v["lost"] <= self.max_lost}
            return []

        track_ids = list(self._tracks.keys())
        if not track_ids:
            result = []
            for box in boxes:
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = {"box": box, "lost": 0}
                result.append((tid, box))
            return result

        # Build cost matrix (1 - IoU)
        track_boxes = np.array([self._tracks[i]["box"] for i in track_ids])
        cost = np.ones((len(track_ids), len(boxes)))
        for ti, tb in enumerate(track_boxes):
            for di, db in enumerate(boxes):
                cost[ti, di] = 1.0 - _iou(tb, db)

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_tracks = set(); matched_dets = set()
        result = []

        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] > 1.0 - self.iou_thresh:
                continue  # poor match → new track
            tid = track_ids[ri]
            self._tracks[tid] = {"box": boxes[ci], "lost": 0}
            result.append((tid, boxes[ci]))
            matched_tracks.add(ri); matched_dets.add(ci)

        # Unmatched detections → new tracks
        for ci, box in enumerate(boxes):
            if ci not in matched_dets:
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = {"box": box, "lost": 0}
                result.append((tid, box))

        # Unmatched tracks → increment lost counter
        for ri, tid in enumerate(track_ids):
            if ri not in matched_tracks:
                self._tracks[tid]["lost"] += 1
        self._tracks = {k: v for k, v in self._tracks.items()
                        if v["lost"] <= self.max_lost}
        return result


# ── pitch detection + filtering ──────────────────────────────────────────────

# HSV range for football pitch grass.
# H 30-85 covers yellow-green to pure green (avoids cyan/blue stands).
# S ≥ 40 avoids washed-out whites (line markings, shirts).
# V 30-210 excludes very dark shadows and very bright specular highlights.
_GRASS_LO = np.array([30,  40,  30], dtype=np.uint8)
_GRASS_HI = np.array([85, 255, 210], dtype=np.uint8)


def _build_pitch_hull(video_path: Path, n_samples: int = 12) -> np.ndarray | None:
    """Return the convex hull polygon of the grass playing surface.

    Samples n_samples frames spread across the clip, thresholds each frame
    for grass colour in HSV, then combines them via pixel-wise vote. The
    convex hull of the resulting green region gives a tight polygon around the
    pitch that automatically excludes advertising boards, coach technical areas,
    and ball-boy positions (which sit on concrete / artificial surfaces).

    Returns an (N, 2) float32 array of hull vertices in (x, y) pixel coords,
    or None if grass detection yields too few pixels to be reliable.
    """
    cap = cv2.VideoCapture(str(video_path))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    votes  = np.zeros((H, W), dtype=np.uint16)

    indices = np.linspace(0, total - 1, n_samples, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _GRASS_LO, _GRASS_HI)
        votes += (mask > 0).astype(np.uint16)
    cap.release()

    # Keep pixels green in at least half the samples (robust to camera cuts)
    binary = ((votes >= n_samples // 2) * 255).astype(np.uint8)

    # Close white lines and gaps so the pitch interior is solid green
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (10, 10)))

    # Collect all green pixel coordinates and compute their convex hull
    ys, xs = np.where(binary > 0)
    if len(xs) < 50:
        return None
    pts  = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(pts)          # shape (N, 1, 2)
    return hull.reshape(-1, 2)          # (N, 2)


def _filter_by_pitch(boxes: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Keep only detections whose bbox centre lies inside the pitch hull."""
    if len(boxes) == 0:
        return boxes
    hull_i32 = hull.astype(np.float32)
    keep = np.array([
        cv2.pointPolygonTest(hull_i32, (float((b[0]+b[2])/2), float((b[1]+b[3])/2)), False) >= 0
        for b in boxes
    ])
    return boxes[keep]


def _apply_border_filter(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fallback filter when pitch-hull detection fails."""
    if len(boxes) == 0:
        return boxes
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    mask = (
        (cx > w * BORDER_X) & (cx < w * (1 - BORDER_X)) &
        (cy > h * BORDER_TOP) & (cy < h * (1 - BORDER_BOT))
    )
    return boxes[mask]


def run_tracking(video_path: Path, model_name: str = YOLO_MODEL) -> dict:
    """Detect + track persons across a video, return JSON-serialisable dict."""
    model = YOLO(model_name)
    tracker = _CentroidTracker()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frame_count_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Build pitch mask once from sampled frames; fall back to border filter if it fails
    print("Building pitch mask from grass colour ...")
    pitch_hull = _build_pitch_hull(video_path)
    if pitch_hull is not None:
        print(f"  Pitch hull OK — {len(pitch_hull)} vertices")
    else:
        print("  Pitch hull failed — using border filter fallback")

    results = model.predict(
        source=str(video_path),
        classes=[PERSON_CLASS_ID],
        stream=True,
        verbose=False,
    )

    frames = []
    for frame_idx, result in enumerate(results):
        boxes_xyxy = (result.boxes.xyxy.cpu().numpy()
                      if result.boxes is not None and len(result.boxes) > 0
                      else np.empty((0, 4)))

        if pitch_hull is not None:
            boxes_xyxy = _filter_by_pitch(boxes_xyxy, pitch_hull)
        else:
            boxes_xyxy = _apply_border_filter(boxes_xyxy, W, H)
        tracks = tracker.update(boxes_xyxy)
        players = []
        for tid, (x1, y1, x2, y2) in tracks:
            players.append({
                "track_id": int(tid),
                "team": None,
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
        frames.append({"frame": frame_idx, "t": frame_idx / fps, "players": players})

        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{frame_count_meta} — {len(players)} players")

    return {
        "clip_id": video_path.stem,
        "video_path": str(video_path),
        "fps": fps,
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
