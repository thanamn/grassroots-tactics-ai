"""Auto-assign team labels (A / B) to each player in tracking JSON using jersey colour.

Usage:
    python scripts/assign_teams.py --tracking data/tracking/clip1.json \
                                   --video    data/clips/clip1.mp4

How it works
------------
1. Sample every SAMPLE_EVERY frames from the video.
2. For each detected player, crop the *jersey zone* — the middle vertical
   third of the bounding box, full width. This avoids hair/skin at the top
   and shorts/boots at the bottom.
3. Compute the median BGR colour of that crop, then convert to CIE L*a*b*.
   L*a*b* is perceptually uniform, so Euclidean distance matches human
   colour perception better than RGB.
4. Stack all (a*, b*) pairs and run k-means with k=2.  Hue lives in a*b*;
   dropping L* makes the cluster robust to lighting variation across the
   pitch.
5. Assign every player in every frame to the nearest cluster centre and
   label them "A" or "B".  The frame-by-frame assignment is independent so
   there is no label-flip risk between frames.
6. Write the updated JSON back to disk (same file, in-place).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TRACKING_DIR, CLIPS_DIR

SAMPLE_EVERY = 5          # use every Nth frame for k-means fitting
JERSEY_TOP    = 0.25      # skip top 25 % of bbox (head)
JERSEY_BOTTOM = 0.65      # stop at 65 % of bbox (waist)
MIN_CROP_PX   = 4         # ignore tiny boxes

# Players whose jersey a*b* is farther than this from the nearest cluster
# centre are treated as non-players (goalkeeper different colour, referee).
# Set to 0 to disable.  Tune by inspecting the cluster centre printout —
# a good value is ~75–80 % of half the inter-cluster distance.
OUTLIER_THRESH = 25.0


def _jersey_lab(frame_bgr: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """Return median (L*, a*, b*) of the jersey zone, or None if crop is too small.

    All three channels are returned so that outlier detection can use lightness
    (L*) in addition to hue (a*b*). k-means is still fitted on a*b* only to
    stay robust against lighting variation, but the goalkeeper's bright green
    and the referee's distinct shade are far enough in 3-D L*a*b* space to be
    reliably excluded.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h = y2 - y1
    jy1 = y1 + int(h * JERSEY_TOP)
    jy2 = y1 + int(h * JERSEY_BOTTOM)
    crop = frame_bgr[jy1:jy2, x1:x2]
    if crop.size < MIN_CROP_PX * MIN_CROP_PX * 3:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return np.array([
        float(np.median(lab[:, :, 0])),   # L*
        float(np.median(lab[:, :, 1])),   # a*
        float(np.median(lab[:, :, 2])),   # b*
    ])


def _kmeans2(X: np.ndarray, n_init: int = 10, max_iter: int = 100,
             seed: int = 42) -> np.ndarray:
    """Lloyd's algorithm k=2, pure numpy — avoids sklearn/OpenBLAS DLL crash."""
    rng = np.random.default_rng(seed)
    best_centres: np.ndarray | None = None
    best_inertia = float("inf")

    for _ in range(n_init):
        idx = rng.choice(len(X), size=2, replace=False)
        centres = X[idx].astype(float)

        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centres = np.array([
                X[labels == k].mean(axis=0) if (labels == k).any() else centres[k]
                for k in range(2)
            ])
            if np.allclose(new_centres, centres):
                break
            centres = new_centres

        inertia = sum(
            np.sum((X[labels == k] - centres[k]) ** 2)
            for k in range(2) if (labels == k).any()
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_centres = centres.copy()

    return best_centres  # shape (2, 2)


def assign_teams(tracking_path: Path, video_path: Path) -> None:
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    frames   = tracking["frames"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # ── pass 1: collect jersey colours for k-means fitting ──────────────────
    print("Pass 1 — extracting jersey colours …")
    lab_samples: list[np.ndarray] = []        # full L*a*b* for 3-D outlier check
    frame_cache: dict[int, np.ndarray] = {}   # frame_idx → BGR image

    for frame_data in frames:
        fidx = frame_data["frame"]
        if fidx % SAMPLE_EVERY != 0:
            continue
        if not frame_data["players"]:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, img = cap.read()
        if not ok:
            continue
        frame_cache[fidx] = img

        for p in frame_data["players"]:
            lab = _jersey_lab(img, p["bbox"])
            if lab is not None:
                lab_samples.append(lab)

    cap.release()

    if len(lab_samples) < 2:
        raise RuntimeError("Too few jersey samples — check that bboxes are valid.")

    lab_arr = np.array(lab_samples)   # (N, 3)  L*, a*, b*
    ab_arr  = lab_arr[:, 1:]          # (N, 2)  a*, b* — used for k-means

    # ── k-means with k=2 on a*b* (pure numpy, no sklearn DLL issues) ────────
    print(f"  {len(lab_samples)} jersey samples collected, fitting k-means …")
    centres = _kmeans2(ab_arr, n_init=10, max_iter=100, seed=42)

    # Deterministic label: cluster with lower a* → "A", other → "B"
    # (arbitrary but consistent within a clip)
    if centres[0, 0] <= centres[1, 0]:
        team_map = {0: "A", 1: "B"}
    else:
        team_map = {0: "B", 1: "A"}

    # Build 3-D cluster centres (L*a*b*) by averaging L* of samples in each cluster
    labels_2d = np.argmin(
        np.linalg.norm(ab_arr[:, None, :] - centres[None, :, :], axis=2), axis=1
    )
    centres_3d = np.array([
        lab_arr[labels_2d == k].mean(axis=0) if (labels_2d == k).any()
        else np.array([128.0, centres[k, 0], centres[k, 1]])
        for k in range(2)
    ])   # shape (2, 3)

    inter_dist = float(np.linalg.norm(centres[0] - centres[1]))
    print(f"  Cluster 0 -> Team {team_map[0]}  |  centre L*a*b* = {centres_3d[0].round(1)}")
    print(f"  Cluster 1 -> Team {team_map[1]}  |  centre L*a*b* = {centres_3d[1].round(1)}")
    print(f"  Inter-cluster distance (a*b*) = {inter_dist:.1f}  |  3-D outlier threshold = {OUTLIER_THRESH}")

    # ── pass 2: assign every player in every frame ───────────────────────────
    print("Pass 2 — assigning team labels to all frames …")
    cap2 = cv2.VideoCapture(str(video_path))
    prev_img: np.ndarray | None = None
    prev_fidx: int = -1

    assigned = total = 0
    for frame_data in frames:
        fidx = frame_data["frame"]
        if not frame_data["players"]:
            continue

        # Reuse cached frame or read a new one
        if fidx in frame_cache:
            img = frame_cache[fidx]
        else:
            if fidx != prev_fidx + 1:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, img = cap2.read()
            if not ok:
                img = prev_img   # fall back to previous frame
            prev_img = img
            prev_fidx = fidx

        for p in frame_data["players"]:
            total += 1
            if img is None:
                continue
            lab = _jersey_lab(img, p["bbox"])
            if lab is None:
                continue
            # Outlier check in 3-D L*a*b* space — catches GK (bright green)
            # and referee (distinct shade) that look similar in a*b* alone
            if OUTLIER_THRESH > 0:
                dists_3d = np.linalg.norm(centres_3d - lab, axis=1)
                if float(np.min(dists_3d)) > OUTLIER_THRESH:
                    p["team"] = None   # goalkeeper / referee — excluded
                    continue
            # Team assignment uses a*b* only (lighting-robust)
            dists_2d = np.linalg.norm(centres - lab[1:], axis=1)
            cluster = int(np.argmin(dists_2d))
            p["team"] = team_map[cluster]
            assigned += 1

    cap2.release()
    print(f"  Assigned {assigned}/{total} players ({100*assigned//total}%)")

    tracking_path.write_text(
        json.dumps(tracking, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved -> {tracking_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--video",    required=True)
    args = parser.parse_args()

    t_path = Path(args.tracking)
    v_path = Path(args.video)
    if not t_path.exists():
        t_path = TRACKING_DIR / t_path.name
    if not v_path.exists():
        v_path = CLIPS_DIR / v_path.name

    assign_teams(t_path, v_path)


if __name__ == "__main__":
    main()
