"""Score ball detection and possession outputs against manual annotations.

The current annotation format stores player points in ``points`` and the ball
in a separate ``ball`` object so player-tracking benchmarks remain unchanged.
True pass-count validation still needs event labels; this script reports the
software pass metrics but does not treat them as validated ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ball_metrics import POSSESSION_THRESHOLD_PX, compute_ball_metrics
from src.config import EVAL_ANNOTATIONS_DIR, EVAL_MANIFESTS_DIR, EVAL_REPORTS_DIR, EVAL_RUNS_DIR


MANUAL_POSSESSION_LABELS = {"A", "B", "contested", "absent"}


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round(value: float | None, digits: int = 4) -> float | str:
    return "" if value is None else round(float(value), digits)


def _safe_ratio(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def _frame_predictions(tracking: dict) -> dict[int, dict]:
    return {int(frame["frame"]): frame for frame in tracking.get("frames", [])}


def _manual_ball(frame_ann: dict) -> tuple[str, dict | None]:
    ball = frame_ann.get("ball")
    if not ball:
        return "missing", None
    if ball.get("status") == "absent":
        return "absent", None
    if "x" in ball and "y" in ball:
        return "visible", {"x": float(ball["x"]), "y": float(ball["y"])}
    return "missing", None


def _match_points(gt_points: list[dict], pred_points: list[dict], threshold: float) -> list[tuple[int, int]]:
    if not gt_points or not pred_points:
        return []
    cost = np.zeros((len(gt_points), len(pred_points)), dtype=float)
    for gi, gt in enumerate(gt_points):
        for pi, pred in enumerate(pred_points):
            cost[gi, pi] = math.hypot(gt["x"] - pred["x"], gt["y"] - pred["y"])
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for gi, pi in zip(rows, cols):
        if float(cost[gi, pi]) <= threshold:
            matches.append((gi, pi))
    return matches


def _swapped_team(team: str | None) -> str | None:
    if team == "A":
        return "B"
    if team == "B":
        return "A"
    return team


def _team_mapping_by_clip(manifest: dict, annotations: dict, clip_cache: dict[str, dict[int, dict]], threshold: float) -> dict[str, dict[str, str]]:
    votes = defaultdict(lambda: {"direct": 0, "swapped": 0})
    for frame in manifest.get("frames", []):
        frame_ann = annotations.get("frames", {}).get(frame["frame_id"], {})
        gt_points = [pt for pt in frame_ann.get("points", []) if pt.get("team") in ("A", "B")]
        pred_frame = clip_cache.get(frame["clip_id"], {}).get(int(frame["frame_index"]), {})
        pred_points = [
            {"x": p["x"], "y": p["y"], "team": p.get("team")}
            for p in pred_frame.get("players", [])
        ]
        for gi, pi in _match_points(gt_points, pred_points, threshold):
            gt_team = gt_points[gi].get("team")
            pred_team = pred_points[pi].get("team")
            if pred_team not in ("A", "B"):
                continue
            if pred_team == gt_team:
                votes[frame["clip_id"]]["direct"] += 1
            if _swapped_team(pred_team) == gt_team:
                votes[frame["clip_id"]]["swapped"] += 1

    mapping = {}
    for clip_id, clip_votes in votes.items():
        mapping[clip_id] = {"A": "B", "B": "A"} if clip_votes["swapped"] > clip_votes["direct"] else {"A": "A", "B": "B"}
    return mapping


def _nearest_team_from_points(ball: dict, points: list[dict], threshold: float) -> str:
    best_dist = float("inf")
    best_team = None
    for point in points:
        team = point.get("team")
        if team not in ("A", "B"):
            continue
        dist = math.hypot(point["x"] - ball["x"], point["y"] - ball["y"])
        if dist < best_dist:
            best_dist = dist
            best_team = team
    return best_team if best_dist <= threshold else "contested"


def _predicted_possession_state(
    pred_ball: dict | None,
    players: list[dict],
    team_mapping: dict[str, str],
    threshold: float,
) -> str:
    if not pred_ball:
        return "absent"
    best_dist = float("inf")
    best_team = None
    for player in players:
        raw_team = player.get("team")
        team = team_mapping.get(raw_team, raw_team)
        if team not in ("A", "B"):
            continue
        dist = math.hypot(player["x"] - pred_ball["x"], player["y"] - pred_ball["y"])
        if dist < best_dist:
            best_dist = dist
            best_team = team
    return best_team if best_dist <= threshold else "contested"


def _pct(counter: Counter, keys: tuple[str, ...]) -> dict[str, float | None]:
    total = sum(counter[k] for k in keys)
    if total == 0:
        return {k: None for k in keys}
    return {k: counter[k] / total * 100 for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", dest="models", action="append", default=[])
    parser.add_argument("--player-match-threshold", type=float, default=35.0)
    parser.add_argument("--ball-match-threshold", type=float, default=50.0)
    parser.add_argument("--possession-threshold", type=float, default=POSSESSION_THRESHOLD_PX)
    args = parser.parse_args()

    manifest = _load_json(EVAL_MANIFESTS_DIR / f"{args.manifest_id}.json")
    annotations = _load_json(EVAL_ANNOTATIONS_DIR / f"{args.manifest_id}.json")
    run_dir = EVAL_RUNS_DIR / args.run_id
    run_summary = _load_json(run_dir / "summary.json")

    selected_models = set(args.models)
    models = [
        model for model in run_summary.get("models", [])
        if not selected_models or model["preset"] in selected_models
    ]

    summary_rows: list[dict] = []
    frame_rows: list[dict] = []
    clip_rows: list[dict] = []

    for model_info in models:
        preset = model_info["preset"]
        team_dir = run_dir / preset / "tracking_with_teams"
        tracking_by_clip = {
            clip["clip_id"]: _load_json(team_dir / f"{clip['clip_id']}.json")
            for clip in model_info["clips"]
        }
        clip_cache = {
            clip_id: _frame_predictions(tracking)
            for clip_id, tracking in tracking_by_clip.items()
        }
        team_mapping = _team_mapping_by_clip(
            manifest,
            annotations,
            clip_cache,
            args.player_match_threshold,
        )

        totals = Counter()
        distances: list[float] = []
        manual_possession_scored = 0
        manual_possession_correct = 0
        proxy_possession_scored = 0
        proxy_possession_correct = 0
        clip_state_counts = defaultdict(lambda: {
            "manual_proxy": Counter(),
            "pred_sample": Counter(),
            "manual_labeled": Counter(),
        })

        for frame in manifest.get("frames", []):
            clip_id = frame["clip_id"]
            frame_id = frame["frame_id"]
            frame_ann = annotations.get("frames", {}).get(frame_id, {})
            gt_status, gt_ball = _manual_ball(frame_ann)
            pred_frame = clip_cache.get(clip_id, {}).get(int(frame["frame_index"]), {})
            pred_ball = pred_frame.get("ball")
            pred_ball_visible = bool(pred_ball)

            if gt_status == "visible":
                totals["gt_visible"] += 1
                if pred_ball_visible:
                    totals["pred_visible_on_gt_visible"] += 1
                    dist = math.hypot(gt_ball["x"] - pred_ball["x"], gt_ball["y"] - pred_ball["y"])
                    distances.append(dist)
                    if dist <= args.ball_match_threshold:
                        totals["pred_within_threshold"] += 1
                else:
                    totals["missed_visible"] += 1
            elif gt_status == "absent":
                totals["gt_absent"] += 1
                if pred_ball_visible:
                    totals["false_visible_on_absent"] += 1
                else:
                    totals["true_absent"] += 1
            else:
                totals["gt_missing"] += 1

            manual_possession = frame_ann.get("possession", "unknown")
            proxy_possession = ""
            if gt_status == "visible":
                proxy_possession = _nearest_team_from_points(
                    gt_ball,
                    frame_ann.get("points", []),
                    args.possession_threshold,
                )
                clip_state_counts[clip_id]["manual_proxy"][proxy_possession] += 1
            elif gt_status == "absent":
                proxy_possession = "absent"

            pred_possession = _predicted_possession_state(
                pred_ball,
                pred_frame.get("players", []),
                team_mapping.get(clip_id, {"A": "A", "B": "B"}),
                args.possession_threshold,
            )
            if gt_status in ("visible", "absent"):
                clip_state_counts[clip_id]["pred_sample"][pred_possession] += 1

            if manual_possession in MANUAL_POSSESSION_LABELS:
                manual_possession_scored += 1
                clip_state_counts[clip_id]["manual_labeled"][manual_possession] += 1
                manual_possession_correct += int(pred_possession == manual_possession)

            if proxy_possession in MANUAL_POSSESSION_LABELS:
                proxy_possession_scored += 1
                proxy_possession_correct += int(pred_possession == proxy_possession)

            frame_rows.append({
                "run_id": args.run_id,
                "manifest_id": args.manifest_id,
                "model_preset": preset,
                "frame_id": frame_id,
                "clip_id": clip_id,
                "frame_index": frame["frame_index"],
                "timestamp_s": frame["timestamp_s"],
                "gt_ball_status": gt_status,
                "pred_ball_visible": pred_ball_visible,
                "ball_error_px": _round(distances[-1], 3) if gt_status == "visible" and pred_ball_visible else "",
                "within_ball_threshold": bool(gt_status == "visible" and pred_ball_visible and distances[-1] <= args.ball_match_threshold) if gt_status == "visible" and pred_ball_visible else "",
                "manual_possession": manual_possession,
                "proxy_possession": proxy_possession,
                "pred_possession": pred_possession,
                "manual_possession_correct": (pred_possession == manual_possession) if manual_possession in MANUAL_POSSESSION_LABELS else "",
                "proxy_possession_correct": (pred_possession == proxy_possession) if proxy_possession in MANUAL_POSSESSION_LABELS else "",
            })

        for clip_id, states in sorted(clip_state_counts.items()):
            proxy_pct = _pct(states["manual_proxy"], ("A", "B", "contested"))
            pred_pct = _pct(states["pred_sample"], ("A", "B", "contested"))
            full_metrics = compute_ball_metrics(tracking_by_clip[clip_id])
            full_possession = (full_metrics or {}).get("possession_pct") or {}
            full_pass_count = (full_metrics or {}).get("pass_count") or {}
            full_pass_accuracy = (full_metrics or {}).get("pass_accuracy") or {}
            clip_rows.append({
                "run_id": args.run_id,
                "manifest_id": args.manifest_id,
                "model_preset": preset,
                "clip_id": clip_id,
                "proxy_A_pct": _round(proxy_pct["A"], 2),
                "proxy_B_pct": _round(proxy_pct["B"], 2),
                "proxy_contested_pct": _round(proxy_pct["contested"], 2),
                "pred_sample_A_pct": _round(pred_pct["A"], 2),
                "pred_sample_B_pct": _round(pred_pct["B"], 2),
                "pred_sample_contested_pct": _round(pred_pct["contested"], 2),
                "sample_A_abs_error": _round(abs(pred_pct["A"] - proxy_pct["A"]), 2) if pred_pct["A"] is not None and proxy_pct["A"] is not None else "",
                "sample_B_abs_error": _round(abs(pred_pct["B"] - proxy_pct["B"]), 2) if pred_pct["B"] is not None and proxy_pct["B"] is not None else "",
                "sample_contested_abs_error": _round(abs(pred_pct["contested"] - proxy_pct["contested"]), 2) if pred_pct["contested"] is not None and proxy_pct["contested"] is not None else "",
                "full_metrics_available": bool(full_metrics),
                "full_possession_A_pct": full_possession.get("A", ""),
                "full_possession_B_pct": full_possession.get("B", ""),
                "full_possession_contested_pct": full_possession.get("contested", ""),
                "full_pass_count_A": full_pass_count.get("A", ""),
                "full_pass_count_B": full_pass_count.get("B", ""),
                "full_pass_accuracy_A": full_pass_accuracy.get("A", ""),
                "full_pass_accuracy_B": full_pass_accuracy.get("B", ""),
                "pass_validation_status": "not_scored_no_manual_pass_events",
            })

        summary_rows.append({
            "run_id": args.run_id,
            "manifest_id": args.manifest_id,
            "model_preset": preset,
            "gt_visible_frames": totals["gt_visible"],
            "gt_absent_frames": totals["gt_absent"],
            "gt_missing_frames": totals["gt_missing"],
            "ball_recall": _round(_safe_ratio(totals["pred_visible_on_gt_visible"], totals["gt_visible"])),
            "ball_within_50px": _round(_safe_ratio(totals["pred_within_threshold"], totals["gt_visible"])),
            "ball_false_visible_on_absent": totals["false_visible_on_absent"],
            "mean_ball_error_px": _round(_mean(distances), 3),
            "median_ball_error_px": _round(_median(distances), 3),
            "manual_possession_frames": manual_possession_scored,
            "manual_possession_accuracy": _round(_safe_ratio(manual_possession_correct, manual_possession_scored)),
            "proxy_possession_frames": proxy_possession_scored,
            "proxy_possession_agreement": _round(_safe_ratio(proxy_possession_correct, proxy_possession_scored)),
            "full_metric_clips_available": sum(1 for tracking in tracking_by_clip.values() if compute_ball_metrics(tracking)),
            "full_metric_clip_count": len(tracking_by_clip),
        })

    reports_base = EVAL_REPORTS_DIR / f"{args.manifest_id}_{args.run_id}"
    _write_csv(
        reports_base.with_name(reports_base.name + "_ball_summary.csv"),
        summary_rows,
        [
            "run_id", "manifest_id", "model_preset", "gt_visible_frames",
            "gt_absent_frames", "gt_missing_frames", "ball_recall",
            "ball_within_50px", "ball_false_visible_on_absent",
            "mean_ball_error_px", "median_ball_error_px",
            "manual_possession_frames", "manual_possession_accuracy",
            "proxy_possession_frames", "proxy_possession_agreement",
            "full_metric_clips_available", "full_metric_clip_count",
        ],
    )
    _write_csv(
        reports_base.with_name(reports_base.name + "_ball_frame_metrics.csv"),
        frame_rows,
        [
            "run_id", "manifest_id", "model_preset", "frame_id", "clip_id",
            "frame_index", "timestamp_s", "gt_ball_status", "pred_ball_visible",
            "ball_error_px", "within_ball_threshold", "manual_possession",
            "proxy_possession", "pred_possession", "manual_possession_correct",
            "proxy_possession_correct",
        ],
    )
    _write_csv(
        reports_base.with_name(reports_base.name + "_ball_clip_metrics.csv"),
        clip_rows,
        [
            "run_id", "manifest_id", "model_preset", "clip_id",
            "proxy_A_pct", "proxy_B_pct", "proxy_contested_pct",
            "pred_sample_A_pct", "pred_sample_B_pct", "pred_sample_contested_pct",
            "sample_A_abs_error", "sample_B_abs_error", "sample_contested_abs_error",
            "full_metrics_available", "full_possession_A_pct", "full_possession_B_pct",
            "full_possession_contested_pct", "full_pass_count_A", "full_pass_count_B",
            "full_pass_accuracy_A", "full_pass_accuracy_B", "pass_validation_status",
        ],
    )
    reports_base.with_name(reports_base.name + "_ball_summary.json").write_text(
        json.dumps({
            "run_id": args.run_id,
            "manifest_id": args.manifest_id,
            "ball_match_threshold_px": args.ball_match_threshold,
            "possession_threshold_px": args.possession_threshold,
            "summary": summary_rows,
            "notes": {
                "ball_recall": "Predicted ball exists on frames manually marked with a visible ball.",
                "ball_within_50px": "Predicted ball exists and is within the configured pixel threshold.",
                "manual_possession_accuracy": "Only scored when annotator explicitly labels possession as A/B/contested/absent.",
                "proxy_possession_agreement": "Uses nearest manually annotated player to the manual ball point; useful as a heuristic check, not true tactical possession ground truth.",
                "pass_validation_status": "Pass count and pass accuracy require separate manual pass/turnover event labels.",
            },
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ball-score] wrote reports under {EVAL_REPORTS_DIR}")


if __name__ == "__main__":
    main()
