"""Compute possession and pass metrics from ball + player tracking data.

Requires that the tracking JSON already has a "ball" key per frame (added by
src/tracking.py when the football-specific YOLO model is used). If ball data
is absent or too sparse, all functions return None so the UI can render "–"
without crashing.

Algorithm outline
-----------------
Possession: per frame, the team whose player is nearest the ball (within a
pixel threshold) has possession. No player within threshold → "contested".

Pass detection: runs on possession spells (consecutive frames owned by the
same track_id). A spell boundary within the same team = completed pass.
A spell boundary crossing to the opposing team = turnover for the old team.
Gaps ≤ MAX_PASS_GAP_S are bridged; larger gaps are treated as contested/out.
"""
from __future__ import annotations

import math
from typing import Any


POSSESSION_THRESHOLD_PX = 150.0  # player feet must be within this distance of ball
MIN_BALL_FRAMES         = 30     # below this, return None (not enough data)
MAX_PASS_GAP_S          = 3.0    # max seconds a ball can be loose and still count as a pass
MIN_SPELL_DURATION_S    = 0.2    # spells shorter than this are tracker jitter, not real possession


def compute_ball_metrics(
    tracking: dict[str, Any],
    possession_threshold_px: float = POSSESSION_THRESHOLD_PX,
) -> dict[str, Any] | None:
    """Return possession % and pass stats, or None if ball data is insufficient."""
    frames = tracking.get("frames", [])
    if not frames:
        return None

    frames_with_ball = sum(1 for f in frames if f.get("ball"))
    frames_no_ball   = len(frames) - frames_with_ball

    if frames_with_ball < MIN_BALL_FRAMES:
        return None

    # ── possession per frame ────────────────────────────────────────────────
    # Each entry: ("A" | "B" | "contested" | None)
    # None means ball not detected in this frame (excluded from denominator).
    poss_seq: list[tuple[str | None, int | None, float]] = []
    # (state, track_id_of_possessing_player, timestamp_t)

    for frame in frames:
        ball = frame.get("ball")
        players = frame.get("players", [])
        t = frame.get("t", 0.0)

        if not ball:
            poss_seq.append((None, None, t))
            continue

        bx, by = ball["x"], ball["y"]
        best_dist  = float("inf")
        best_team  = None
        best_tid   = None

        for p in players:
            team = p.get("team")
            if team not in ("A", "B"):
                continue
            dx = p["x"] - bx
            dy = p["y"] - by
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_team = team
                best_tid  = p["track_id"]

        if best_dist <= possession_threshold_px:
            poss_seq.append((best_team, best_tid, t))
        else:
            poss_seq.append(("contested", None, t))

    # ── aggregate possession counts ─────────────────────────────────────────
    frames_A          = sum(1 for s, _, _ in poss_seq if s == "A")
    frames_B          = sum(1 for s, _, _ in poss_seq if s == "B")
    frames_contested  = sum(1 for s, _, _ in poss_seq if s == "contested")
    total_tracked     = frames_A + frames_B + frames_contested

    if total_tracked == 0:
        return None

    # Round individually then correct the last value so the three always
    # sum to exactly 100.0 — independent rounding can leave a 0.1 gap.
    pct_a = round(frames_A         / total_tracked * 100, 1)
    pct_b = round(frames_B         / total_tracked * 100, 1)
    pct_c = round(100.0 - pct_a - pct_b, 1)
    possession_pct = {"A": pct_a, "B": pct_b, "contested": pct_c}

    # ── possession spells ───────────────────────────────────────────────────
    # Collapse consecutive frames with the same (team, track_id) into spells.
    spells: list[dict] = []
    for state, tid, t in poss_seq:
        if state is None or state == "contested":
            continue
        if (spells
                and spells[-1]["team"] == state
                and spells[-1]["track_id"] == tid):
            spells[-1]["t_end"] = t
        else:
            spells.append({"team": state, "track_id": tid, "t_start": t, "t_end": t})

    # ── pass detection ──────────────────────────────────────────────────────
    # Filter out tracker-jitter spells before counting passes.  The centroid
    # tracker re-assigns IDs on re-detection, so a player who briefly leaves
    # the frame gets a new ID — each switch looks like a "pass" in the spell
    # sequence.  Spells shorter than MIN_SPELL_DURATION_S are almost always
    # this tracker noise rather than real ball control.
    meaningful_spells = [s for s in spells
                         if s["t_end"] - s["t_start"] >= MIN_SPELL_DURATION_S]

    pass_count  = {"A": 0, "B": 0}
    turnovers   = {"A": 0, "B": 0}   # failed pass attempts

    for i in range(len(meaningful_spells) - 1):
        curr = meaningful_spells[i]
        nxt  = meaningful_spells[i + 1]
        gap  = nxt["t_start"] - curr["t_end"]

        if gap > MAX_PASS_GAP_S:
            continue   # ball was lost too long; not a pass attempt

        if nxt["team"] == curr["team"]:
            # Same team, different player → completed pass
            if nxt["track_id"] != curr["track_id"]:
                pass_count[curr["team"]] += 1
        else:
            # Ball changed team → turnover (counts as failed pass for curr team)
            turnovers[curr["team"]] += 1

    # ── pass accuracy ───────────────────────────────────────────────────────
    pass_accuracy: dict[str, float | None] = {}
    for team in ("A", "B"):
        total_attempts = pass_count[team] + turnovers[team]
        if total_attempts == 0:
            pass_accuracy[team] = None
        else:
            pass_accuracy[team] = round(pass_count[team] / total_attempts, 2)

    return {
        "possession_pct":   possession_pct,
        "pass_count":       pass_count,
        "pass_accuracy":    pass_accuracy,
        "frames_with_ball": frames_with_ball,
        "frames_no_ball":   frames_no_ball,
    }
