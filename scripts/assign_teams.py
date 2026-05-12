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

# Broadcast/tactical shots always place the technical area and coaching staff
# at the bottom of the frame. Any detection whose bounding-box bottom edge
# (y2) exceeds this fraction of frame height is almost certainly a coach,
# physio, or ball-boy standing behind the touchline — not a field player.
# GKs (cls=1) are exempt so a GK guarding the near goal is never silenced.
# Set to 1.0 to disable (e.g. ground-level grassroots clips where the
# technical area is off-camera or at the frame edge horizontally).
MAX_PLAYER_Y_FRAC = 0.90

# If the smallest k=3 cluster contains at least this fraction of players,
# there is no obvious outlier group → fall back to k=2 assignment.
REFEREE_CLUSTER_MAX_FRAC = 0.30

# When k-means a*b* separation between the two team centres is below this
# threshold, retry clustering in full L*a*b* space. Night games (e.g. PSG
# dark navy vs Arsenal white/red under floodlights) can look nearly identical
# in hue (a*b*) but are clearly separated by luminance (L*).
MIN_INTER_DIST = 12.0

# In-shape filtering ----------------------------------------------------
# Keep raw detections, but only let on-field outfield players contribute to
# convex hulls/spacing. Sideline staff/subs often get detected as "player" and
# can wear similar colours, so jersey clustering alone is not enough.
# 0.18 (was 0.12) — coaches/staff standing near the touchline can have small
# patches of grass at their feet on broadcast cameras, but a real on-pitch
# player almost always has a clear majority of green pixels around their feet.
FOOT_GRASS_MIN_RATIO = 0.18
FOOT_PATCH_W_FRAC = 0.55
FOOT_PATCH_H_FRAC = 0.18

# Track-level non-player rules. Per-frame filters miss coaches whose foot
# patch happens to clip a green technical-area mat, or wing players who run
# very close to the touchline. Looking at the *whole track's* behaviour is
# much more robust than judging one detection at a time.
#
# NON_PLAYER_STICKY_THRESH — if at least this fraction of a track's frames
# were flagged sideline OR outlier, force the whole track non-player. This
# overrides the standard role-majority vote, which used to let a 30/70
# split sit as "outfield" and pull the hull.
NON_PLAYER_STICKY_THRESH = 0.30
# TRACK_AVG_Y_FRAC_THRESH — any track whose mean bbox-bottom / frame-height
# stays above this for the clip is almost certainly a person standing in the
# technical area. A genuine wing player dips this low occasionally but does
# not average there.
TRACK_AVG_Y_FRAC_THRESH = 0.88
# Lowered from 0.60: with the sticky-sideline override above doing the heavy
# lifting, the residual cases benefit from a more permissive role majority
# so brief role mis-flags on legit players don't drop them from the hull.
ROLE_MAJORITY_THRESH = 0.50

# Continuity pass -------------------------------------------------------
# The tactical overlay should read like a team shape, not a detector debug
# stream. These values intentionally prefer short, plausible continuity over
# frame-by-frame disappearance when the detector drops a distant player.
CONTINUITY_MAX_GAP_S = 6.0
CONTINUITY_MAX_DIST_SCALE = 5.0
CONTINUITY_MAX_PER_FRAME_PX = 60.0
CONTINUITY_DUPLICATE_PX = 30.0
TRANSIENT_MIN_TRACK_S = 0.28
STABLE_SHAPE_TRACKS_PER_TEAM = 12


def _grass_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return (
        (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90) &
        (hsv[:, :, 1] > 45) &
        (hsv[:, :, 2] > 35)
    )


def _foot_grass_ratio(frame_bgr: np.ndarray, bbox: list[float]) -> float:
    """Grass ratio around the player's feet/bottom-centre point."""
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw = max(2.0, x2 - x1)
    bh = max(4.0, y2 - y1)
    cx = (x1 + x2) / 2.0

    # Sample a small band around and just above the foot point. Players on the
    # pitch usually have grass in this neighbourhood; staff/subs beyond the
    # touchline often sit on track/concrete/bench areas.
    px1 = int(max(0, cx - bw * FOOT_PATCH_W_FRAC))
    px2 = int(min(w_img, cx + bw * FOOT_PATCH_W_FRAC))
    py1 = int(max(0, y2 - bh * FOOT_PATCH_H_FRAC))
    py2 = int(min(h_img, y2 + bh * 0.06))
    if px2 <= px1 or py2 <= py1:
        return 0.0
    patch = _grass_mask(frame_bgr[py1:py2, px1:px2])
    return float(np.mean(patch)) if patch.size else 0.0


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


def _shape_player(p: dict) -> bool:
    return p.get("team") in ("A", "B") and p.get("include_in_shape") is not False


def _point(p: dict) -> np.ndarray:
    return np.array([float(p["x"]), float(p["y"])], dtype=float)


def _bbox_diag(p: dict) -> float:
    x1, y1, x2, y2 = (float(v) for v in p["bbox"])
    return max(8.0, float(np.hypot(x2 - x1, y2 - y1)))


def _track_observations(frames: list[dict]) -> dict[int, list[tuple[int, dict]]]:
    obs: dict[int, list[tuple[int, dict]]] = {}
    for frame in frames:
        frame_no = int(frame["frame"])
        for player in frame.get("players", []):
            if _shape_player(player):
                obs.setdefault(int(player["track_id"]), []).append((frame_no, player))
    for track_obs in obs.values():
        track_obs.sort(key=lambda item: item[0])
    return obs


def _tail_velocity(track_obs: list[tuple[int, dict]]) -> np.ndarray:
    if len(track_obs) < 2:
        return np.zeros(2, dtype=float)
    tail = track_obs[-min(5, len(track_obs)):]
    f0, p0 = tail[0]
    f1, p1 = tail[-1]
    return (_point(p1) - _point(p0)) / max(1, f1 - f0)


def _resolve_parent(parent: dict[int, int], tid: int) -> int:
    root = tid
    while root in parent and parent[root] != root:
        root = parent[root]
    while tid in parent and parent[tid] != tid:
        nxt = parent[tid]
        parent[tid] = root
        tid = nxt
    return root


def _merge_same_team_fragments(frames: list[dict], fps: float) -> int:
    obs = _track_observations(frames)
    track_ids = list(obs)
    if len(track_ids) < 2:
        return 0

    max_gap_frames = min(90, int(round(fps * CONTINUITY_MAX_GAP_S)))
    team_by_track = {tid: obs[tid][0][1].get("team") for tid in track_ids}
    start = {tid: obs[tid][0][0] for tid in track_ids}
    end = {tid: obs[tid][-1][0] for tid in track_ids}
    parent: dict[int, int] = {}
    used_predecessors: set[int] = set()
    candidates: dict[int, list[tuple[int, float]]] = {}

    for a_tid in track_ids:
        a_obs = obs[a_tid]
        a_last = a_obs[-1][1]
        a_velocity = _tail_velocity(a_obs)
        a_diag = _bbox_diag(a_last)
        a_team = team_by_track.get(a_tid)
        if a_team not in ("A", "B"):
            continue
        for b_tid in track_ids:
            if a_tid == b_tid or team_by_track.get(b_tid) != a_team:
                continue
            gap = start[b_tid] - end[a_tid]
            if gap <= 0 or gap > max_gap_frames:
                continue
            b_first = obs[b_tid][0][1]
            predicted = _point(a_last) + a_velocity * gap
            dist = float(np.linalg.norm(_point(b_first) - predicted))
            avg_diag = (a_diag + _bbox_diag(b_first)) / 2.0
            if dist > max(45.0, avg_diag * CONTINUITY_MAX_DIST_SCALE):
                continue
            if dist / max(1, gap) > CONTINUITY_MAX_PER_FRAME_PX:
                continue
            candidates.setdefault(b_tid, []).append((a_tid, dist + 2.0 * gap))

    for b_tid in sorted(candidates, key=lambda tid: start[tid]):
        ranked = sorted(candidates[b_tid], key=lambda item: item[1])
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
        if runner_score is not None and runner_score < best[1] * 1.25:
            continue
        parent[b_tid] = _resolve_parent(parent, best[0])
        used_predecessors.add(best[0])

    if not parent:
        return 0

    for frame in frames:
        for player in frame.get("players", []):
            tid = int(player["track_id"])
            merged_tid = _resolve_parent(parent, tid)
            if merged_tid != tid:
                player["track_id"] = merged_tid
    return len(parent)


def _interpolate_shape_player(prev: dict, nxt: dict, alpha: float) -> dict:
    prev_box = np.asarray(prev["bbox"], dtype=float)
    next_box = np.asarray(nxt["bbox"], dtype=float)
    box = prev_box + (next_box - prev_box) * alpha
    x1, y1, x2, y2 = box
    player = dict(prev)
    player.update({
        "cls": int(prev.get("cls", nxt.get("cls", 2))),
        "team": prev.get("team"),
        "interpolated": True,
        "interpolation": "team_continuity",
        "x": float((x1 + x2) / 2.0),
        "y": float(y2),
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "on_pitch": True,
        "include_in_shape": True,
        "shape_role": "outfield",
    })
    return player


def _has_duplicate_shape_player(players: list[dict], candidate: dict) -> bool:
    cand_xy = _point(candidate)
    cand_team = candidate.get("team")
    cand_tid = int(candidate["track_id"])
    for player in players:
        if not _shape_player(player):
            continue
        if int(player["track_id"]) == cand_tid:
            return True
        if player.get("team") == cand_team and float(np.linalg.norm(_point(player) - cand_xy)) <= CONTINUITY_DUPLICATE_PX:
            return True
    return False


def _fill_same_track_gaps(frames: list[dict], fps: float, vid_stride: int) -> int:
    obs = _track_observations(frames)
    if not obs:
        return 0
    frames_by_no = {int(frame["frame"]): frame for frame in frames}
    step = max(1, int(vid_stride or 1))
    max_gap_frames = min(90, int(round(fps * CONTINUITY_MAX_GAP_S)))
    inserted = 0

    for track_obs in obs.values():
        for (f0, p0), (f1, p1) in zip(track_obs, track_obs[1:]):
            gap = f1 - f0
            if gap <= step or gap > max_gap_frames:
                continue
            if p0.get("team") != p1.get("team"):
                continue
            if float(np.linalg.norm(_point(p1) - _point(p0))) / max(1, gap) > CONTINUITY_MAX_PER_FRAME_PX:
                continue
            frame_no = f0 + step
            while frame_no < f1:
                frame = frames_by_no.get(frame_no)
                if frame is not None:
                    alpha = (frame_no - f0) / gap
                    candidate = _interpolate_shape_player(p0, p1, alpha)
                    players = frame.setdefault("players", [])
                    if not _has_duplicate_shape_player(players, candidate):
                        players.append(candidate)
                        inserted += 1
                frame_no += step

    if inserted:
        for frame in frames:
            frame["players"].sort(key=lambda player: int(player["track_id"]))
    return inserted


def _suppress_transient_shape_tracks(frames: list[dict], fps: float, vid_stride: int) -> int:
    obs = _track_observations(frames)
    min_obs = max(4, int(round((fps / max(1, vid_stride)) * TRANSIENT_MIN_TRACK_S)))
    transient_ids = {
        tid for tid, track_obs in obs.items()
        if len(track_obs) < min_obs and (track_obs[-1][0] - track_obs[0][0]) <= int(round(fps * TRANSIENT_MIN_TRACK_S))
    }
    if not transient_ids:
        return 0
    suppressed = 0
    for frame in frames:
        for player in frame.get("players", []):
            if int(player.get("track_id", -1)) in transient_ids and _shape_player(player):
                player["team"] = None
                player["include_in_shape"] = False
                player["on_pitch"] = False
                player["shape_role"] = "transient"
                suppressed += 1
    return suppressed


def _limit_shape_roster(frames: list[dict], max_tracks_per_team: int = STABLE_SHAPE_TRACKS_PER_TEAM) -> dict[str, int]:
    if max_tracks_per_team <= 0:
        return {
            "kept_shape_tracks": 0,
            "suppressed_roster_overflow_tracks": 0,
            "suppressed_roster_overflow_players": 0,
        }

    obs = _track_observations(frames)
    by_team: dict[str, list[tuple[int, tuple[int, int, int]]]] = {"A": [], "B": []}
    for tid, track_obs in obs.items():
        if not track_obs:
            continue
        team = track_obs[0][1].get("team")
        if team not in by_team:
            continue
        span = track_obs[-1][0] - track_obs[0][0] + 1
        measured = sum(1 for _, player in track_obs if not player.get("interpolated"))
        by_team[team].append((tid, (span, measured, len(track_obs))))

    keep_ids: set[int] = set()
    overflow_ids: set[int] = set()
    for team_tracks in by_team.values():
        ranked = sorted(team_tracks, key=lambda item: item[1], reverse=True)
        keep_ids.update(tid for tid, _ in ranked[:max_tracks_per_team])
        overflow_ids.update(tid for tid, _ in ranked[max_tracks_per_team:])

    if not overflow_ids:
        return {
            "kept_shape_tracks": len(keep_ids),
            "suppressed_roster_overflow_tracks": 0,
            "suppressed_roster_overflow_players": 0,
        }

    suppressed_players = 0
    for frame in frames:
        for player in frame.get("players", []):
            tid = int(player.get("track_id", -1))
            if tid in overflow_ids and _shape_player(player):
                player["include_in_shape"] = False
                player["on_pitch"] = True
                player["shape_role"] = "roster_overflow"
                suppressed_players += 1

    return {
        "kept_shape_tracks": len(keep_ids),
        "suppressed_roster_overflow_tracks": len(overflow_ids),
        "suppressed_roster_overflow_players": suppressed_players,
    }


def smooth_team_continuity(frames: list[dict], fps: float, vid_stride: int = 1) -> dict[str, int]:
    merged = _merge_same_team_fragments(frames, fps)
    inserted = _fill_same_track_gaps(frames, fps, vid_stride)
    suppressed = _suppress_transient_shape_tracks(frames, fps, vid_stride)
    roster_stats = _limit_shape_roster(frames)
    return {
        "merged_track_fragments": merged,
        "inserted_gap_players": inserted,
        "suppressed_transient_players": suppressed,
        **roster_stats,
    }


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
            if p.get("interpolated"):
                continue
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
        inter_dist_ab = float(np.linalg.norm(centres3[0] - centres3[1]))
        if inter_dist_ab < MIN_INTER_DIST:
            # a*b* (hue) failed to separate the two kits — retry in full L*a*b*.
            # Night games with dark-navy vs white kits (e.g. PSG vs Arsenal) are
            # nearly identical in hue but clearly different in luminance (L*).
            print(f"  a*b* separation {inter_dist_ab:.1f} < {MIN_INTER_DIST} — "
                  f"retrying with L*a*b* (3-D) …")
            centres_lab, labels_lab = _kmeans_k(lab_arr, k=2, n_init=10,
                                                max_iter=100, seed=42)
            inter_dist_lab = float(np.linalg.norm(centres_lab[0] - centres_lab[1]))
            if inter_dist_lab > inter_dist_ab:
                centres3 = centres_lab[:, 1:]   # a*b* slice — keeps label logic intact
                labels3  = labels_lab
                print(f"  L*a*b* separation {inter_dist_lab:.1f} — using 3-D result")

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
    rejected_sideline = 0
    rejected_goalkeeper = 0
    # Per-track bottom-edge fraction history — used by the track-level
    # sticky-sideline rule below. Collected here so we only walk the
    # detections once instead of doing a separate pass.
    y_frac_obs: dict[int, list[float]] = {}
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
            p["on_pitch"] = None
            p["include_in_shape"] = False
            p["shape_role"] = "unknown"
            if frame_h:
                y_frac_obs.setdefault(int(p["track_id"]), []).append(
                    float(p["bbox"][3]) / float(frame_h)
                )
            if img is None:
                continue
            if p.get("interpolated"):
                continue
            if p.get("cls") == 1:
                p["on_pitch"] = True
                p["include_in_shape"] = False
                p["shape_role"] = "goalkeeper"
                rejected_goalkeeper += 1
                continue
            # Y-boundary guard: reject detections whose bottom edge is below
            # MAX_PLAYER_Y_FRAC of frame height. Technical-area staff (coaches,
            # physios, ball-boys) always appear in this bottom strip on broadcast
            # tactical shots.
            if MAX_PLAYER_Y_FRAC < 1.0 and p.get("cls") != 1:
                y2 = p["bbox"][3]
                if frame_h and y2 > frame_h * MAX_PLAYER_Y_FRAC:
                    p["on_pitch"] = False
                    p["shape_role"] = "sideline"
                    rejected_sideline += 1
                    continue  # leave team=None
            foot_grass_ratio = _foot_grass_ratio(img, p["bbox"])
            p["foot_grass_ratio"] = round(foot_grass_ratio, 3)
            if foot_grass_ratio < FOOT_GRASS_MIN_RATIO:
                p["on_pitch"] = False
                p["shape_role"] = "sideline"
                rejected_sideline += 1
                continue
            p["on_pitch"] = True
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
                    p["include_in_shape"] = False
                    p["shape_role"] = "outlier"
                    continue  # leave team=None → excluded from hull/lines
            # Team assignment in full L*a*b* space — more robust than a*b*-only
            # when luminance is the discriminative dimension (dark vs bright kits).
            dists_assign = np.linalg.norm(centres_3d - lab, axis=1)
            best_tc   = team_clusters[int(np.argmin(dists_assign))]
            p["team"] = team_map[best_tc]
            p["include_in_shape"] = True
            p["shape_role"] = "outfield"
            assigned += 1

    cap2.release()

    # Short tracker bridges do not have a real detection crop, and real
    # detections can occasionally fail the jersey/outlier checks. Fill those
    # holes from the track's majority team so one bad crop doesn't make the
    # overlay blink a player out for a frame.
    votes: dict[int, dict[str, int]] = {}
    shape_votes: dict[int, dict[str, int]] = {}
    for frame_data in frames:
        for p in frame_data["players"]:
            tid = int(p["track_id"])
            team = p.get("team")
            if team in ("A", "B"):
                votes.setdefault(tid, {"A": 0, "B": 0})[team] += 1
            role = p.get("shape_role")
            if role in ("outfield", "goalkeeper", "sideline", "outlier"):
                shape_votes.setdefault(tid, {
                    "outfield": 0, "goalkeeper": 0, "sideline": 0, "outlier": 0,
                })[role] += 1

    track_team: dict[int, str] = {}
    for tid, counts in votes.items():
        total_votes = counts["A"] + counts["B"]
        if total_votes < 2:
            continue
        team, n_votes = max(counts.items(), key=lambda item: item[1])
        if n_votes / total_votes >= 0.65:
            track_team[tid] = team

    # Per-track mean bottom-edge fraction. Used by the y-frac sticky rule
    # below — a track that consistently sits near the bottom of the frame is
    # almost always technical-area staff, even if its individual detections
    # passed the per-frame y/grass gates.
    track_avg_y_frac: dict[int, float] = {
        tid: sum(fracs) / len(fracs)
        for tid, fracs in y_frac_obs.items()
        if len(fracs) >= 3
    }

    track_role: dict[int, str] = {}
    sticky_sideline_tracks = 0
    sticky_y_frac_tracks = 0
    for tid, counts in shape_votes.items():
        total_votes = sum(counts.values())
        if total_votes < 2:
            continue
        non_player = counts.get("sideline", 0) + counts.get("outlier", 0)
        # Sticky-sideline override: if a non-trivial fraction of the track's
        # frames were flagged sideline OR outlier, force the whole track
        # non-player regardless of the outfield/GK majority. This stops the
        # 30 % sideline / 70 % outfield split from leaking a coach into
        # the convex hull.
        if non_player / total_votes >= NON_PLAYER_STICKY_THRESH:
            track_role[tid] = (
                "sideline" if counts.get("sideline", 0) >= counts.get("outlier", 0)
                else "outlier"
            )
            sticky_sideline_tracks += 1
            continue
        # Track-y sticky rule: average bottom-edge fraction over the clip is
        # a strong signal — wing players dip low occasionally but do not
        # average there, technical-area staff do.
        avg_y = track_avg_y_frac.get(tid)
        if avg_y is not None and avg_y >= TRACK_AVG_Y_FRAC_THRESH and counts.get("goalkeeper", 0) == 0:
            track_role[tid] = "sideline"
            sticky_y_frac_tracks += 1
            continue
        role, n_votes = max(counts.items(), key=lambda item: item[1])
        if n_votes / total_votes >= ROLE_MAJORITY_THRESH:
            track_role[tid] = role

    inherited = 0
    corrected = 0
    role_inherited = 0
    for frame_data in frames:
        for p in frame_data["players"]:
            tid = int(p["track_id"])
            role = track_role.get(tid)
            if role:
                p["shape_role"] = role
                p["include_in_shape"] = role == "outfield"
                p["on_pitch"] = role in ("outfield", "goalkeeper")
                if p.get("interpolated") or p.get("team") not in ("A", "B"):
                    role_inherited += 1
            team = track_team.get(tid)
            if not team:
                continue
            if not p.get("include_in_shape"):
                p["team"] = None
                continue
            if p.get("team") not in ("A", "B"):
                p["team"] = team
                inherited += 1
            elif p["team"] != team:
                p["team"] = team
                corrected += 1

    if sticky_sideline_tracks:
        print(f"  Sticky-sideline override on {sticky_sideline_tracks} tracks "
              f"(>={int(NON_PLAYER_STICKY_THRESH*100)}% non-player frames)")
    if sticky_y_frac_tracks:
        print(f"  Y-frac override on {sticky_y_frac_tracks} tracks "
              f"(avg bottom-edge >= {TRACK_AVG_Y_FRAC_THRESH:.2f} of frame height)")
    print(f"  Assigned {assigned}/{total} players ({100*assigned//total}%)")
    if inherited:
        print(f"  Filled {inherited} short team-label gaps from track majority")
    if corrected:
        print(f"  Stabilized {corrected} noisy team labels from track majority")
    if role_inherited:
        print(f"  Inherited {role_inherited} in-shape/sideline roles from track history")
    if rejected_goalkeeper:
        print(f"  Excluded {rejected_goalkeeper} goalkeeper detections from hulls")
    if rejected_sideline:
        print(f"  Excluded {rejected_sideline} likely sideline/off-pitch detections")

    continuity_stats = smooth_team_continuity(
        frames,
        fps=float(tracking.get("fps", 25.0)),
        vid_stride=int(tracking.get("vid_stride", 1) or 1),
    )
    tracking["team_continuity"] = continuity_stats
    if any(continuity_stats.values()):
        print(
            "  Team continuity:"
            f" merged={continuity_stats['merged_track_fragments']}"
            f" inserted={continuity_stats['inserted_gap_players']}"
            f" suppressed={continuity_stats['suppressed_transient_players']}"
            f" kept_shape_tracks={continuity_stats['kept_shape_tracks']}"
            f" roster_overflow={continuity_stats['suppressed_roster_overflow_players']}"
        )

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
