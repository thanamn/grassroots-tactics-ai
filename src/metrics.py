"""Compute tactical spacing/compactness metrics from tracking JSON.

Usage:
    python -m src.metrics --input data/tracking/sample.json

Output: data/cache/<clip_stem>_metrics.json

Metrics computed per frame, per team:
    - hull_area      : area of the convex hull of outfield players (px²)
    - centroid       : (x, y) mean position
    - spread_std     : std-dev of player distances from team centroid
And cross-team:
    - centroid_distance : Euclidean distance between the two team centroids

Then we summarise across all frames and detect "events" where compactness
changes sharply (the moments worth highlighting to a coach).

Why these metrics
-----------------
They are simple, transparent, and well-grounded in the football analytics
literature (Memmert et al. 2017, Low et al. 2020): hull area = how much
pitch a team occupies, centroid distance = how stretched the game is,
spread_std = how compact the team is around its core.

Goalkeepers
-----------
Excluding the goalkeeper from the hull dramatically improves the metric
because GK position is structurally far from the outfield block. We pick
the player furthest from their team centroid as a proxy GK if exclude_gk
is on. For broadcast clips that often crop out the GK, this is harmless.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from src.config import CACHE_DIR, TRACKING_DIR


# --- helpers -----------------------------------------------------------

def _team_metrics(points: Sequence[tuple[float, float]], exclude_gk: bool = True) -> dict | None:
    """Return hull_area, centroid, spread_std for one team in one frame."""
    if len(points) < 3:
        return None
    pts = np.asarray(points, dtype=float)

    if exclude_gk and len(pts) >= 4:
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        keep_idx = np.argsort(dists)[:-1]    # drop furthest player
        pts = pts[keep_idx]

    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    spread_std = float(np.std(dists))

    try:
        hull = ConvexHull(pts)
        hull_area = float(hull.volume)   # in 2D, .volume is the area
    except QhullError:
        hull_area = 0.0

    return {
        "hull_area": hull_area,
        "centroid": [float(centroid[0]), float(centroid[1])],
        "spread_std": spread_std,
        "n_players": int(len(pts)),
    }


def _summarise(values: list[float]) -> dict:
    """Compact summary of a numeric series."""
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _detect_events(per_frame: list[dict], spike_threshold_pct: float = 25.0,
                   window_s: float = 1.5) -> list[dict]:
    """Flag moments where hull_area changes sharply within a short window.

    A 'compactness_spike' is a drop ≥ spike_threshold_pct% in hull_area
    over `window_s` seconds. A 'stretch' is the symmetric increase.
    """
    events = []
    for team in ("team_A", "team_B"):
        for i, frame in enumerate(per_frame):
            cur = frame.get(team)
            if not cur:
                continue
            t_now = frame["t"]
            # find frame ~window_s earlier
            j = i
            while j > 0 and t_now - per_frame[j]["t"] < window_s:
                j -= 1
            past = per_frame[j].get(team)
            if not past or past["hull_area"] == 0:
                continue
            delta_pct = (cur["hull_area"] - past["hull_area"]) / past["hull_area"] * 100
            if delta_pct <= -spike_threshold_pct:
                events.append({
                    "t": t_now, "team": team, "type": "compactness_spike",
                    "delta_pct": round(delta_pct, 1),
                    "hull_before": round(past["hull_area"], 1),
                    "hull_after": round(cur["hull_area"], 1),
                })
            elif delta_pct >= spike_threshold_pct:
                events.append({
                    "t": t_now, "team": team, "type": "stretch",
                    "delta_pct": round(delta_pct, 1),
                    "hull_before": round(past["hull_area"], 1),
                    "hull_after": round(cur["hull_area"], 1),
                })

    # de-duplicate: keep the strongest event in any 1-second window
    events.sort(key=lambda e: (e["team"], e["t"]))
    deduped: list[dict] = []
    for ev in events:
        if deduped and deduped[-1]["team"] == ev["team"] \
                and ev["t"] - deduped[-1]["t"] < 1.0:
            if abs(ev["delta_pct"]) > abs(deduped[-1]["delta_pct"]):
                deduped[-1] = ev
        else:
            deduped.append(ev)
    return deduped


# --- main pipeline -----------------------------------------------------

def compute_metrics(tracking: dict) -> dict:
    per_frame = []
    for frame in tracking["frames"]:
        teams: dict[str, list] = defaultdict(list)
        for p in frame["players"]:
            if p.get("team") in ("A", "B"):
                teams[p["team"]].append((p["x"], p["y"]))

        entry = {"t": frame["t"], "frame": frame["frame"]}
        ma = _team_metrics(teams.get("A", []))
        mb = _team_metrics(teams.get("B", []))
        if ma:
            entry["team_A"] = ma
        if mb:
            entry["team_B"] = mb
        if ma and mb:
            ax, ay = ma["centroid"]
            bx, by = mb["centroid"]
            entry["centroid_distance"] = float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
        per_frame.append(entry)

    summary: dict = {}
    for team in ("team_A", "team_B"):
        hulls = [f[team]["hull_area"] for f in per_frame if team in f]
        spreads = [f[team]["spread_std"] for f in per_frame if team in f]
        summary[team] = {
            "hull_area": _summarise(hulls),
            "spread_std": _summarise(spreads),
        }
    centroid_dists = [f["centroid_distance"] for f in per_frame if "centroid_distance" in f]
    summary["centroid_distance"] = _summarise(centroid_dists)

    events = _detect_events(per_frame)

    from src.ball_metrics import compute_ball_metrics
    ball_metrics = compute_ball_metrics(tracking)

    return {
        "clip_id": tracking["clip_id"],
        "fps": tracking["fps"],
        "duration_s": per_frame[-1]["t"] if per_frame else 0.0,
        "per_frame": per_frame,
        "summary": summary,
        "events": events,
        "ball_metrics": ball_metrics,   # None when ball data is absent or insufficient
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to tracking JSON")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        alt = TRACKING_DIR / in_path.name
        if alt.exists():
            in_path = alt
        else:
            raise FileNotFoundError(in_path)

    tracking = json.loads(in_path.read_text())
    metrics = compute_metrics(tracking)

    out_path = Path(args.output) if args.output else CACHE_DIR / f"{in_path.stem}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")
    print(f"  {len(metrics['per_frame'])} frames, {len(metrics['events'])} events")


if __name__ == "__main__":
    main()
