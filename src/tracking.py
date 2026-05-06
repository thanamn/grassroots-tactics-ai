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
- Player (x, y) is the bottom-centre of the bounding box, not the geometric
  centre. Bottom-centre approximates where the player's feet contact the
  pitch — the ground-plane point that spacing/hull metrics actually want to
  model. Centroid would over-weight tall players and skew the hull whenever
  bbox heights vary (camera zoom, motion blur).
- Detections are filtered by both confidence (CONF_THRESHOLD) and bbox area
  (MIN_BBOX_AREA) before reaching the tracker. Default YOLO confidence (0.25)
  lets in too many refs and partially-occluded sideline figures; tightening
  to 0.4 drops noise without losing real players. The area floor catches
  tiny detections of distant spectators or broadcast-graphic figurines that
  pollute the hull.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from src.config import (
    CLIPS_DIR,
    DEFAULT_FPS,
    PERSON_CLASS_ID,
    TRACKING_DIR,
    YOLO_MODEL,
)

# ── detection-quality knobs ─────────────────────────────────────────────────
# Tuned for broadcast / wide-angle football clips. Re-tune if grassroots
# footage has a very different scale (e.g. close-up handheld → raise
# MIN_BBOX_AREA, very-wide drone → lower it).
CONF_THRESHOLD = 0.4   # YOLO confidence floor; default 0.25 admits too many refs/sideline figures
MIN_BBOX_AREA  = 300   # px²; below this is almost certainly a spectator or graphic, not a player

# Auto-detect CUDA. On the dev laptop (RTX 4070) this gives ~10–15× the
# throughput of CPU inference; on a machine without an NVIDIA GPU this
# silently falls back to "cpu" rather than failing. We deliberately do
# NOT raise on CPU — the fallback is the documented mode for non-GPU
# users (Colab, paper-writing on a laptop without a GPU, etc.).
DEVICE = 0 if torch.cuda.is_available() else "cpu"


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



def run_tracking(video_path: Path, model_name: str | None = None,
                 vid_stride: int = 1) -> dict:
    """Detect + track persons across a video, return JSON-serialisable dict.

    Parameters
    ----------
    model_name : str | None
        Path/name of the YOLO weights. ``None`` falls back to ``YOLO_MODEL``
        from config. For long clips the pipeline overrides this with a
        nano model (``yolo11n.pt``) so tracking finishes in a session.
    vid_stride : int
        Process every ``vid_stride``-th frame of the video. ``1`` (default)
        keeps full temporal resolution; long clips use stride 3–5 and let
        the visualiser's smoothing window stretch to cover the gaps.
        Ultralytics' built-in ``vid_stride`` is fed straight through so
        we don't pay the cost of decoding skipped frames.
    """
    model_name = model_name or YOLO_MODEL
    model = YOLO(model_name)
    tracker = _CentroidTracker()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frame_count_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    gpu_name = torch.cuda.get_device_name(0) if DEVICE != 'cpu' else 'CPU only'
    print(f"[tracking] device={DEVICE!r} ({gpu_name}) "
          f"model={model_name} vid_stride={vid_stride}")
    results = model.predict(
        source=str(video_path),
        classes=[PERSON_CLASS_ID],
        conf=CONF_THRESHOLD,
        stream=True,
        verbose=False,
        device=DEVICE,
        vid_stride=vid_stride,
    )

    frames = []
    # When vid_stride > 1, ultralytics yields one result per *processed*
    # frame, so the loop index i maps back to the original video frame
    # number via ``i * vid_stride``. We store the original frame number
    # so downstream stages keep their existing per-frame timestamp logic.
    for i, result in enumerate(results):
        frame_idx = i * vid_stride
        boxes_xyxy = (result.boxes.xyxy.cpu().numpy()
                      if result.boxes is not None and len(result.boxes) > 0
                      else np.empty((0, 4)))

        # Min-area filter — drops distant spectators, graphics, and partial
        # detections. Applied here (not inside the tracker) so the tracker
        # never sees the noise and never spawns short-lived ghost IDs from it.
        if len(boxes_xyxy):
            areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            boxes_xyxy = boxes_xyxy[areas >= MIN_BBOX_AREA]

        tracks = tracker.update(boxes_xyxy)
        players = []
        for tid, (x1, y1, x2, y2) in tracks:
            players.append({
                "track_id": int(tid),
                "team": None,
                # Bottom-centre = where the player's feet meet the pitch.
                # See module docstring for why this beats geometric centroid
                # for spacing/hull metrics.
                "x": float((x1 + x2) / 2),
                "y": float(y2),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
        frames.append({"frame": frame_idx, "t": frame_idx / fps, "players": players})

        # Print every 100 processed frames, not every 100 source frames —
        # otherwise stride=5 only logs every 500-frame block which is
        # unhelpfully sparse.
        if i % 100 == 0:
            print(f"  frame {frame_idx}/{frame_count_meta} — {len(players)} players")

    return {
        "clip_id": video_path.stem,
        "video_path": str(video_path),
        "fps": fps,
        "vid_stride": vid_stride,
        "model": model_name,
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
