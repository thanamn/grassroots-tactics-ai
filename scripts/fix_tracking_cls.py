"""Retroactively re-apply the updated GK threshold to existing tracking JSONs.

tracking.py v2 requires ≥5 frames AND >50 % cls=1 votes to label a track
as goalkeeper.  Old JSONs used 1 frame / 40 % which occasionally produced
false GK markers.  This script patches the cls field in-place without
re-running YOLO inference.

Usage:
    .venv\Scripts\python.exe scripts/fix_tracking_cls.py
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TRACKING_DIR

MIN_FRAMES = 5
GK_FRAC    = 0.50


def fix_json(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))

    # Pass 1: aggregate per-track votes across ALL frames
    votes: dict[int, dict] = {}  # track_id → {gk: int, total: int}
    for frame in data["frames"]:
        for p in frame["players"]:
            tid = p["track_id"]
            d = votes.setdefault(tid, {"gk": 0, "total": 0})
            d["total"] += 1
            if p.get("cls") == 1:
                d["gk"] += 1

    # Determine dominant cls per track with new thresholds
    dominant: dict[int, int] = {}
    for tid, d in votes.items():
        if d["total"] >= MIN_FRAMES and d["gk"] / d["total"] > GK_FRAC:
            dominant[tid] = 1
        else:
            dominant[tid] = 2

    # Pass 2: update every player entry
    changed = 0
    for frame in data["frames"]:
        for p in frame["players"]:
            new_cls = dominant.get(p["track_id"], 2)
            if p.get("cls") != new_cls:
                p["cls"] = new_cls
                changed += 1

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    n_gk = sum(1 for v in dominant.values() if v == 1)
    print(f"  {path.name}: {changed} cls entries updated, {n_gk} GK tracks identified")


def main() -> None:
    jsons = list(TRACKING_DIR.glob("*.json"))
    if not jsons:
        print("No tracking JSONs found.")
        return
    print(f"Fixing {len(jsons)} tracking JSON(s) in {TRACKING_DIR} …")
    for p in sorted(jsons):
        fix_json(p)
    print("Done.")


if __name__ == "__main__":
    main()
