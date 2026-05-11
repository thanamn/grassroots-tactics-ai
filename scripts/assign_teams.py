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
# Jersey crop: chest band, also tightened horizontally to the centre half of
# the bbox. YOLO bboxes on broadcast tactical shots are loose — typically
# 1.5-2x wider than the player's silhouette and extending above the head.
# A 15-50% top-only crop (the previous default) sampled mostly grass on small
# detections, which collapsed the median jersey colour toward grass-green and
# made the team k-means unable to separate the two kits.
JERSEY_TOP    = 0.30      # skip head + shoulders
JERSEY_BOTTOM = 0.65      # stop above shorts
JERSEY_LEFT   = 0.25      # centre half horizontally — drops grass/edge
JERSEY_RIGHT  = 0.75
MIN_CROP_PX   = 4         # ignore tiny boxes
# Minimum bbox height (px) for samples used to fit k-means cluster centres.
# Small detections on zoomed-out tactical shots (~20-30 px tall) yield only
# a handful of jersey pixels, which after grass-masking and median-collapse
# look ~grass-green and pull the cluster centres into the grass cluster.
# Pass-2 assignment still uses every detection regardless of size.
MIN_TRAIN_BBOX_H_PX = 32

# Players whose jersey a*b* is farther than this from the nearest cluster
# centre are treated as non-players (goalkeeper different colour, referee).
# Set to 0 to disable.
OUTLIER_THRESH = 40.0

# Broadcast/tactical shots always place the technical area and coaching staff
# at the bottom of the frame. Any detection whose bounding-box bottom edge
# (y2) exceeds this fraction of frame height is almost certainly a coach,
# physio, or ball-boy standing behind the touchline — not a field player.
# GKs (cls=1) are exempt so a GK guarding the near goal is never silenced.
# Set to 1.0 to disable (e.g. ground-level grassroots clips where the
# technical area is off-camera or at the frame edge horizontally).
MAX_PLAYER_Y_FRAC = 0.83

# Camera operators and close-touchline staff have disproportionately large
# bounding boxes because they stand near the camera lens. Real players at
# broadcast distance have bbox heights of 22–45px. Anything taller than
# MAX_BBOX_H_PX whose bottom edge is below MAX_BBOX_Y_FRAC is almost
# certainly not a pitch player. GKs (cls=1) are exempt.
MAX_BBOX_H_PX  = 60
MAX_BBOX_Y_FRAC = 0.60

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
    w = x2 - x1
    jy1 = y1 + int(h * JERSEY_TOP)
    jy2 = y1 + int(h * JERSEY_BOTTOM)
    jx1 = x1 + int(w * JERSEY_LEFT)
    jx2 = x1 + int(w * JERSEY_RIGHT)
    crop = frame_bgr[jy1:jy2, jx1:jx2]
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
            # Only use sufficiently large bboxes for cluster training. On
            # zoomed-out shots small bboxes have just a few jersey pixels
            # that get drowned out by motion blur and grass bleed; their
            # noisy colour pulls the k-means centres toward grass-green.
            bbox = p["bbox"]
            if bbox[3] - bbox[1] < MIN_TRAIN_BBOX_H_PX:
                continue
            lab = _jersey_lab(img, bbox)
            if lab is not None:
                lab_samples.append(lab)

    cap.release()

    if len(lab_samples) < 2:
        raise RuntimeError("Too few jersey samples — check that bboxes are valid.")

    lab_arr = np.array(lab_samples)   # (N, 3)  L*, a*, b* — clustering input

    # ── Cluster in full 3-D L*a*b* ──────────────────────────────────────────
    # Earlier versions clustered only in a*b* (hue plane), retrying in 3-D
    # only when a*b* separation was tiny. This silently failed on PSG-vs-
    # Arsenal: a*b* distance was just above the retry threshold (~19) so 3-D
    # was skipped, even though L* (luminance) was the only dimension that
    # actually distinguished the two kits. Always clustering in 3-D removes
    # the brittle threshold and works equally well for hue-distinct kits
    # (hue dominates the 3-D distance) and luminance-distinct kits.
    print(f"  {len(lab_samples)} jersey samples collected, fitting DBSCAN (3-D) …")
    try:
        from sklearn.cluster import DBSCAN as _DBSCAN
        db = _DBSCAN(eps=25, min_samples=3, metric="euclidean").fit(lab_arr)
        raw_labels  = db.labels_          # -1 = noise/outlier
        cluster_ids = [l for l in np.unique(raw_labels) if l != -1]
    except ImportError:
        cluster_ids = []                  # force k-means fallback

    if len(cluster_ids) >= 2:
        sizes = {l: int((raw_labels == l).sum()) for l in cluster_ids}
        top2  = sorted(sizes, key=sizes.get, reverse=True)[:2]  # two largest
        centres_3d = np.array([lab_arr[raw_labels == l].mean(axis=0) for l in top2])
        labels3    = np.full(len(lab_arr), -1, dtype=int)
        labels3[raw_labels == top2[0]] = 0
        labels3[raw_labels == top2[1]] = 1
        team_clusters = [0, 1]
        print(f"  DBSCAN: {len(cluster_ids)} clusters found; using top-2 as teams "
              f"(sizes {sizes[top2[0]]}, {sizes[top2[1]]}; "
              f"{int((raw_labels == -1).sum())} noise points)")
    else:
        print(f"  DBSCAN found < 2 clusters — falling back to k-means k=2 (3-D) …")
        centres_3d, labels3 = _kmeans_k(lab_arr, k=2, n_init=10, max_iter=100, seed=42)
        team_clusters = [0, 1]

    inter_dist = float(np.linalg.norm(centres_3d[0] - centres_3d[1]))
    print(f"  3-D L*a*b* inter-cluster distance = {inter_dist:.1f}")

    # Deterministic labelling: cluster with lower a* (less red) → "A", other → "B"
    tc0, tc1 = team_clusters[0], team_clusters[1]
    if centres_3d[tc0, 1] <= centres_3d[tc1, 1]:
        team_map = {tc0: "A", tc1: "B"}
    else:
        team_map = {tc0: "B", tc1: "A"}

    # Refine cluster centres from the actually-assigned points (DBSCAN noise
    # excluded). Falls back to the cluster's k-means/DBSCAN centre when empty.
    centres_3d = np.array([
        lab_arr[labels3 == c].mean(axis=0) if (labels3 == c).any() else centres_3d[c]
        for c in team_clusters
    ])

    print(f"  Team A centre L*a*b* = {centres_3d[0].round(1)}")
    print(f"  Team B centre L*a*b* = {centres_3d[1].round(1)}")
    print(f"  Outlier threshold = {OUTLIER_THRESH}")

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

        frame_h = img.shape[0] if img is not None else None
        for p in frame_data["players"]:
            total += 1
            if img is None:
                continue
            # Y-boundary guard: reject detections whose bottom edge is below
            # MAX_PLAYER_Y_FRAC of frame height. Technical-area staff (coaches,
            # physios, ball-boys) always appear in this bottom strip on broadcast
            # tactical shots. GKs are exempt so a near-post GK isn't silenced.
            if MAX_PLAYER_Y_FRAC < 1.0 and p.get("cls") != 1:
                y2 = p["bbox"][3]
                if frame_h and y2 > frame_h * MAX_PLAYER_Y_FRAC:
                    continue  # leave team=None
            # Large-bbox guard: camera operators close to the lens have
            # disproportionately large boxes. Real pitch players at broadcast
            # distance have bbox heights of 22–45 px. GKs are exempt.
            if p.get("cls") != 1:
                bbox_h_px = p["bbox"][3] - p["bbox"][1]
                if (bbox_h_px > MAX_BBOX_H_PX
                        and frame_h
                        and p["bbox"][3] > frame_h * MAX_BBOX_Y_FRAC):
                    continue  # leave team=None
            lab = _jersey_lab(img, p["bbox"])
            if lab is None:
                continue
            # Outlier check: jersey colour is far from both team centres →
            # almost certainly a referee, linesman, or sideline staff.
            # GKs (cls=1) are exempt — their unusual kits legitimately differ
            # from outfield players but they ARE part of one team.
            if OUTLIER_THRESH > 0 and p.get("cls") != 1:
                dists_3d = np.linalg.norm(centres_3d - lab, axis=1)
                if float(np.min(dists_3d)) > OUTLIER_THRESH:
                    continue  # leave team=None → excluded from hull/lines
            # Team assignment in full L*a*b* space — more robust than a*b*-only
            # when luminance is the discriminative dimension (dark vs bright kits).
            dists_assign = np.linalg.norm(centres_3d - lab, axis=1)
            best_tc   = team_clusters[int(np.argmin(dists_assign))]
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
