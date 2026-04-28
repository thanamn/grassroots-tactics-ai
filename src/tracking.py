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



def run_tracking(video_path: Path, model_name: str = YOLO_MODEL) -> dict:
    """Detect + track persons across a video, return JSON-serialisable dict."""
    model = YOLO(model_name)
    tracker = _CentroidTracker()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frame_count_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

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
