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
JERSEY_TOP    = 0.15      # skip top 15 % of bbox (head)
JERSEY_BOTTOM = 0.50      # stop at 50 % of bbox — shirt zone, above shorts
MIN_CROP_PX   = 4         # ignore tiny boxes

# Players whose jersey a*b* is farther than this from the nearest cluster
# centre are treated as non-players (goalkeeper different colour, referee).
# Set to 0 to disable.
OUTLIER_THRESH = 40.0

# If the smallest k=3 cluster contains at least this fraction of players,
# there is no obvious outlier group → fall back to k=2 assignment.
REFEREE_CLUSTER_MAX_FRAC = 0.30


def _jersey_lab(frame_bgr: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """Return median (L*, a*, b*) of non-grass jersey pixels, or None.

    Crops the shirt zone (JERSEY_TOP–JERSEY_BOTTOM of bbox), masks out
    pitch-green pixels in HSV space, then computes per-channel median in
    L*a*b*. Grass masking prevents pitch bleed from biasing the colour
    estimate, especially when players are partially obscured or on the edge.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h = y2 - y1
    jy1 = y1 + int(h * JERSEY_TOP)
    jy2 = y1 + int(h * JERSEY_BOTTOM)
    crop = frame_bgr[jy1:jy2, x1:x2]
    if crop.size < MIN_CROP_PX * MIN_CROP_PX * 3:
        return None

    # Mask out grass-coloured pixels before computing jersey colour
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass_mask = (
        (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) &  # green hue
        (hsv[:, :, 1] > 60) &                             # saturated
        (hsv[:, :, 2] > 40)                               # not too dark
    )
    pixels_bgr = crop.reshape(-1, 3)[~grass_mask.reshape(-1)]
    if len(pixels_bgr) < MIN_CROP_PX:
        # Almost entirely grass — fall back to full crop
        pixels_bgr = crop.reshape(-1, 3)

    # Compute median in L*a*b* on the surviving pixels
    lab_pixels = cv2.cvtColor(
        pixels_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3)
    return np.array([
        float(np.median(lab_pixels[:, 0])),   # L*
        float(np.median(lab_pixels[:, 1])),   # a*
        float(np.median(lab_pixels[:, 2])),   # b*
    ])


def _kmeans_k(X: np.ndarray, k: int = 2, n_init: int = 10, max_iter: int = 100,
              seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with k clusters, pure numpy — no sklearn/OpenBLAS DLL.

    Returns (centres, labels) where centres has shape (k, d) and labels has
    shape (N,) with values in range(k).
    """
    rng = np.random.default_rng(seed)
    best_centres: np.ndarray | None = None
    best_labels:  np.ndarray | None = None
    best_inertia = float("inf")

    for _ in range(n_init):
        idx = rng.choice(len(X), size=k, replace=False)
        centres = X[idx].astype(float)

        for _ in range(max_iter):
            dists  = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centres = np.array([
                X[labels == c].mean(axis=0) if (labels == c).any() else centres[c]
                for c in range(k)
            ])
            if np.allclose(new_centres, centres):
                break
            centres = new_centres

        inertia = sum(
            float(np.sum((X[labels == c] - centres[c]) ** 2))
            for c in range(k) if (labels == c).any()
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_centres = centres.copy()
            best_labels  = labels.copy()

    return best_centres, best_labels  # type: ignore[return-value]


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

    # ── DBSCAN on a*b* — no need to guess k, outliers become noise (-1) ────────
    # Falls back to k-means k=2 when sklearn is absent or DBSCAN yields < 2 clusters.
    print(f"  {len(lab_samples)} jersey samples collected, fitting DBSCAN …")
    try:
        from sklearn.cluster import DBSCAN as _DBSCAN
        db = _DBSCAN(eps=22, min_samples=3, metric="euclidean").fit(ab_arr)
        raw_labels  = db.labels_          # -1 = noise/outlier
        cluster_ids = [l for l in np.unique(raw_labels) if l != -1]
    except ImportError:
        cluster_ids = []                  # force k-means fallback

    if len(cluster_ids) >= 2:
        sizes = {l: int((raw_labels == l).sum()) for l in cluster_ids}
        top2  = sorted(sizes, key=sizes.get, reverse=True)[:2]  # two largest
        centres3  = np.array([ab_arr[raw_labels == l].mean(axis=0) for l in top2])
        labels3   = np.full(len(ab_arr), -1, dtype=int)
        labels3[raw_labels == top2[0]] = 0
        labels3[raw_labels == top2[1]] = 1
        team_clusters = [0, 1]
        print(f"  DBSCAN: {len(cluster_ids)} clusters found; using top-2 as teams "
              f"(sizes {sizes[top2[0]]}, {sizes[top2[1]]}; "
              f"{int((raw_labels == -1).sum())} noise points)")
    else:
        print(f"  DBSCAN found < 2 clusters — falling back to k-means k=2 …")
        centres3, labels3 = _kmeans_k(ab_arr, k=2, n_init=10, max_iter=100, seed=42)
        team_clusters = [0, 1]

    # Deterministic labelling: cluster with lower a* → "A", other → "B"
    tc0, tc1 = team_clusters[0], team_clusters[1]
    if centres3[tc0, 0] <= centres3[tc1, 0]:
        team_map = {tc0: "A", tc1: "B"}
    else:
        team_map = {tc0: "B", tc1: "A"}

    # Build 3-D cluster centres (L*a*b*) for the team clusters
    centres_3d = np.array([
        lab_arr[labels3 == c].mean(axis=0) if (labels3 == c).any()
        else np.array([128.0, centres3[c, 0], centres3[c, 1]])
        for c in team_clusters
    ])   # shape (2, 3)

    inter_dist = float(np.linalg.norm(centres3[tc0] - centres3[tc1]))
    print(f"  Team A centre L*a*b* = {centres_3d[0].round(1)}")
    print(f"  Team B centre L*a*b* = {centres_3d[1].round(1)}")
    print(f"  Inter-cluster distance (a*b*) = {inter_dist:.1f}  |  outlier threshold = {OUTLIER_THRESH}")

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
            # Outlier check — log distance but always fall through to assignment.
            # Leaving team=None would exclude the player from the hull entirely;
            # the GK exclusion in metrics/visualizer drops the furthest player
            # per team, which is the correct place to handle the goalkeeper.
            if OUTLIER_THRESH > 0:
                dists_3d = np.linalg.norm(centres_3d - lab, axis=1)
                if float(np.min(dists_3d)) > OUTLIER_THRESH:
                    pass  # assign to nearest team anyway — don't leave as null
            # Team assignment: find nearest team cluster in a*b* space
            team_ab = centres3[team_clusters]   # shape (2, 2)
            dists_2d = np.linalg.norm(team_ab - lab[1:], axis=1)
            best_tc   = team_clusters[int(np.argmin(dists_2d))]
            p["team"] = team_map[best_tc]
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
