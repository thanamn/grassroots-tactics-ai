"""Deduplicate per-frame YOLO detections in tracking JSONs.

YOLO's default NMS (IoU=0.7) is too lenient for tightly-cropped broadcast
football, where the model often emits 2-3 overlapping boxes for the same
player (one for upper body, one for full body, etc.). The centroid tracker
then assigns each box a different track_id, producing visible duplicate
dots in the overlay and skewing possession/spacing metrics.

This pass merges any same-team detections in a single frame whose feet
positions are within DUPE_THRESH_PX of each other, keeping the detection
with the longest accumulated track history (most stable ID across frames).

Run after tracking, before metrics + visualizer:

    python scripts/dedupe_tracking.py [--job <id>]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRACKING_DIR

DUPE_THRESH_PX = 35.0   # foot positions closer than this = same physical player


def _track_lifetimes(tracking: dict) -> Counter:
    """Count how many frames each track_id appears in."""
    c: Counter = Counter()
    for f in tracking["frames"]:
        for p in f["players"]:
            c[p["track_id"]] += 1
    return c


def dedupe(tracking: dict) -> tuple[int, int]:
    """Modify tracking dict in place; return (frames_changed, dets_dropped)."""
    lifetimes = _track_lifetimes(tracking)
    frames_changed = 0
    dets_dropped = 0

    for f in tracking["frames"]:
        players = f["players"]
        if len(players) < 2:
            continue

        keep = [True] * len(players)
        for i in range(len(players)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(players)):
                if not keep[j]:
                    continue
                pi, pj = players[i], players[j]
                # Only merge same-team duplicates. Different teams = legit
                # close-marking situation (defender/attacker).
                if pi.get("team") != pj.get("team"):
                    continue
                dx = pi["x"] - pj["x"]
                dy = pi["y"] - pj["y"]
                if dx * dx + dy * dy >= DUPE_THRESH_PX * DUPE_THRESH_PX:
                    continue
                # Same-position duplicate. Drop the one with shorter lifetime;
                # tie-breaks go to the lower (older) track_id.
                li = lifetimes[pi["track_id"]]
                lj = lifetimes[pj["track_id"]]
                if (li, -pi["track_id"]) >= (lj, -pj["track_id"]):
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i is dead, stop comparing it

        if not all(keep):
            frames_changed += 1
            dropped = sum(1 for k in keep if not k)
            dets_dropped += dropped
            f["players"] = [p for p, k in zip(players, keep) if k]

    return frames_changed, dets_dropped


def process(tracking_path: Path) -> None:
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    before = sum(len(f["players"]) for f in tracking["frames"])
    fc, dd = dedupe(tracking)
    after = sum(len(f["players"]) for f in tracking["frames"])
    tracking_path.write_text(
        json.dumps(tracking, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {tracking_path.name}: {fc} frames cleaned, {dd} dets dropped "
          f"({before} -> {after}, {100*dd/before:.1f}% reduction)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default=None, help="Single job ID (omit for all)")
    args = parser.parse_args()

    if args.job:
        paths = [TRACKING_DIR / f"{args.job}.json"]
    else:
        paths = sorted(TRACKING_DIR.glob("*.json"))

    print(f"Deduplicating {len(paths)} tracking JSON(s) "
          f"(merge same-team dets within {DUPE_THRESH_PX:.0f}px)")
    for p in paths:
        if p.exists():
            process(p)
        else:
            print(f"  {p.name}: missing — skip")


if __name__ == "__main__":
    main()
