"""Score a tracking benchmark against manual point annotations.

Ground truth is point-based: each labelled player has a bottom-centre point,
team label, and stable ID. That is lighter than full bounding-box annotation
but directly matches what the spacing metrics consume downstream.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import EVAL_ANNOTATIONS_DIR, EVAL_MANIFESTS_DIR, EVAL_REPORTS_DIR, EVAL_RUNS_DIR


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


def _frame_predictions(tracking: dict) -> dict[int, list[dict]]:
    return {frame["frame"]: frame.get("players", []) for frame in tracking.get("frames", [])}


def _relevant_gt_points(frame_ann: dict) -> list[dict]:
    return [pt for pt in frame_ann.get("points", []) if pt.get("team") in ("A", "B")]


def _match_points(
    gt_points: list[dict],
    pred_points: list[dict],
    threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Hungarian point matching, then reject matches beyond threshold."""
    if not gt_points or not pred_points:
        return [], list(range(len(gt_points))), list(range(len(pred_points)))

    cost = np.zeros((len(gt_points), len(pred_points)), dtype=float)
    for gi, gt in enumerate(gt_points):
        for pi, pred in enumerate(pred_points):
            cost[gi, pi] = np.hypot(gt["x"] - pred["x"], gt["y"] - pred["y"])

    rows, cols = linear_sum_assignment(cost)
    matches = []
    used_gt = set()
    used_pred = set()
    for gi, pi in zip(rows, cols):
        dist = float(cost[gi, pi])
        if dist <= threshold:
            matches.append((gi, pi, dist))
            used_gt.add(gi)
            used_pred.add(pi)

    unmatched_gt = [i for i in range(len(gt_points)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_points)) if i not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def _swapped_team(team: str | None) -> str | None:
    if team == "A":
        return "B"
    if team == "B":
        return "A"
    return team


def _align_team(team: str | None, mapping: dict[str, str]) -> str | None:
    return mapping.get(team, team)


def _safe_ratio(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def _round(value: float | None, digits: int = 4) -> float | str:
    return "" if value is None else round(float(value), digits)


def _bootstrap_ci(values: list[float], iterations: int, rng: np.random.Generator) -> tuple[float | None, float | None]:
    clean = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(clean) < 2:
        return None, None
    idx = rng.integers(0, len(clean), size=(iterations, len(clean)))
    means = clean[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_bootstrap_delta(
    baseline_values: dict[str, float | None],
    model_values: dict[str, float | None],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None, float | None, int]:
    keys = [
        k for k in baseline_values.keys() & model_values.keys()
        if baseline_values[k] is not None and model_values[k] is not None
    ]
    if len(keys) < 2:
        return None, None, None, len(keys)

    base = np.array([baseline_values[k] for k in keys], dtype=float)
    model = np.array([model_values[k] for k in keys], dtype=float)
    deltas = model - base
    idx = rng.integers(0, len(deltas), size=(iterations, len(deltas)))
    sample_means = deltas[idx].mean(axis=1)
    return (
        float(deltas.mean()),
        float(np.percentile(sample_means, 2.5)),
        float(np.percentile(sample_means, 97.5)),
        len(keys),
    )


def _metric_from_frame(row: dict, metric: str) -> float | None:
    if metric == "coverage":
        return _safe_ratio(row["matched"], row["gt"])
    if metric == "precision":
        return _safe_ratio(row["matched"], row["matched"] + row["false_positives"])
    if metric == "team_accuracy":
        return _safe_ratio(row["team_correct"], row["team_scored"])
    if metric == "mean_error_px":
        return row["mean_error_px"]
    raise ValueError(metric)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline", default="football_players")
    parser.add_argument("--distance-threshold", type=float, default=35.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", dest="models", action="append", default=[])
    args = parser.parse_args()

    manifest = _load_json(EVAL_MANIFESTS_DIR / f"{args.manifest_id}.json")
    annotations = _load_json(EVAL_ANNOTATIONS_DIR / f"{args.manifest_id}.json")
    run_dir = EVAL_RUNS_DIR / args.run_id
    run_summary = _load_json(run_dir / "summary.json")
    rng = np.random.default_rng(args.seed)

    detail_rows: list[dict] = []
    frame_rows: list[dict] = []
    summary_rows: list[dict] = []
    per_model_frame_metric_values: dict[str, dict[str, dict[str, float | None]]] = defaultdict(dict)

    selected_models = set(args.models)
    models = [
        model for model in run_summary["models"]
        if not selected_models or model["preset"] in selected_models
    ]

    for model_info in models:
        preset = model_info["preset"]
        team_dir = run_dir / preset / "tracking_with_teams"
        clip_cache = {
            clip["clip_id"]: _frame_predictions(_load_json(team_dir / f"{clip['clip_id']}.json"))
            for clip in model_info["clips"]
        }

        totals = {
            "gt": 0,
            "matched": 0,
            "false_positives": 0,
            "team_correct": 0,
            "team_scored": 0,
            "continuity_correct": 0,
            "continuity_scored": 0,
        }
        all_distances: list[float] = []
        prev_matches_by_seq = defaultdict(dict)
        team_votes_by_clip = defaultdict(lambda: {"direct": 0, "swapped": 0})

        for frame in manifest["frames"]:
            frame_ann = annotations.get("frames", {}).get(frame["frame_id"], {"points": []})
            gt_points = _relevant_gt_points(frame_ann)
            pred_all = clip_cache[frame["clip_id"]].get(frame["frame_index"], [])
            pred_points = [
                {
                    "track_id": p["track_id"],
                    "team": p.get("team"),
                    "x": p["x"],
                    "y": p["y"],
                }
                for p in pred_all
            ]

            matches, _, _ = _match_points(gt_points, pred_points, args.distance_threshold)
            for gi, pi, _ in matches:
                gt_team = gt_points[gi].get("team")
                pred_team = pred_points[pi].get("team")
                if pred_team not in ("A", "B"):
                    continue
                if pred_team == gt_team:
                    team_votes_by_clip[frame["clip_id"]]["direct"] += 1
                if _swapped_team(pred_team) == gt_team:
                    team_votes_by_clip[frame["clip_id"]]["swapped"] += 1

        team_mapping_by_clip = {}
        for clip_id, votes in team_votes_by_clip.items():
            if votes["swapped"] > votes["direct"]:
                team_mapping_by_clip[clip_id] = {"A": "B", "B": "A"}
            else:
                team_mapping_by_clip[clip_id] = {"A": "A", "B": "B"}

        for frame in manifest["frames"]:
            frame_ann = annotations.get("frames", {}).get(frame["frame_id"], {"points": []})
            gt_points = _relevant_gt_points(frame_ann)
            pred_all = clip_cache[frame["clip_id"]].get(frame["frame_index"], [])
            pred_points = [
                {
                    "track_id": p["track_id"],
                    "team": p.get("team"),
                    "x": p["x"],
                    "y": p["y"],
                }
                for p in pred_all
            ]

            matches, _, unmatched_pred = _match_points(gt_points, pred_points, args.distance_threshold)
            frame_team_correct = 0
            frame_team_scored = 0
            frame_distances: list[float] = []
            current_seq_map = {}

            for gi, pi, dist in matches:
                gt = gt_points[gi]
                pred = pred_points[pi]
                pred_team_raw = pred.get("team")
                pred_team_aligned = _align_team(
                    pred_team_raw,
                    team_mapping_by_clip.get(frame["clip_id"], {"A": "A", "B": "B"}),
                )
                team_correct = pred_team_aligned == gt.get("team")
                frame_team_scored += 1
                frame_team_correct += int(team_correct)
                frame_distances.append(dist)
                all_distances.append(dist)
                detail_rows.append({
                    "run_id": args.run_id,
                    "manifest_id": args.manifest_id,
                    "model_preset": preset,
                    "frame_id": frame["frame_id"],
                    "clip_id": frame["clip_id"],
                    "frame_index": frame["frame_index"],
                    "timestamp_s": frame["timestamp_s"],
                    "kind": frame["kind"],
                    "sequence_id": frame.get("sequence_id") or "",
                    "gt_id": gt["id"],
                    "gt_team": gt["team"],
                    "pred_track_id": pred["track_id"],
                    "pred_team_raw": pred_team_raw or "",
                    "pred_team_aligned": pred_team_aligned or "",
                    "distance_px": round(dist, 3),
                    "team_correct": team_correct,
                })
                if frame.get("sequence_id"):
                    current_seq_map[gt["id"]] = pred["track_id"]

            continuity_correct = 0
            continuity_scored = 0
            seq_id = frame.get("sequence_id")
            if seq_id:
                previous = prev_matches_by_seq[seq_id]
                for gt_id, track_id in current_seq_map.items():
                    if gt_id in previous:
                        continuity_scored += 1
                        continuity_correct += int(previous[gt_id] == track_id)
                prev_matches_by_seq[seq_id] = current_seq_map

            row = {
                "run_id": args.run_id,
                "manifest_id": args.manifest_id,
                "model_preset": preset,
                "frame_id": frame["frame_id"],
                "clip_id": frame["clip_id"],
                "frame_index": frame["frame_index"],
                "timestamp_s": frame["timestamp_s"],
                "kind": frame["kind"],
                "sequence_id": frame.get("sequence_id") or "",
                "gt": len(gt_points),
                "matched": len(matches),
                "false_positives": len(unmatched_pred),
                "coverage": _round(_safe_ratio(len(matches), len(gt_points))),
                "precision": _round(_safe_ratio(len(matches), len(matches) + len(unmatched_pred))),
                "team_accuracy": _round(_safe_ratio(frame_team_correct, frame_team_scored)),
                "mean_error_px": _round(_mean(frame_distances), 3),
                "continuity_scored": continuity_scored,
                "continuity_correct": continuity_correct,
            }
            frame_rows.append(row)

            totals["gt"] += len(gt_points)
            totals["matched"] += len(matches)
            totals["false_positives"] += len(unmatched_pred)
            totals["team_correct"] += frame_team_correct
            totals["team_scored"] += frame_team_scored
            totals["continuity_correct"] += continuity_correct
            totals["continuity_scored"] += continuity_scored

            for metric in ("coverage", "precision", "team_accuracy", "mean_error_px"):
                per_model_frame_metric_values[preset].setdefault(metric, {})[frame["frame_id"]] = _metric_from_frame(
                    {
                        "gt": len(gt_points),
                        "matched": len(matches),
                        "false_positives": len(unmatched_pred),
                        "team_correct": frame_team_correct,
                        "team_scored": frame_team_scored,
                        "mean_error_px": _mean(frame_distances),
                    },
                    metric,
                )

        coverage = _safe_ratio(totals["matched"], totals["gt"])
        precision = _safe_ratio(totals["matched"], totals["matched"] + totals["false_positives"])
        f1 = None
        if coverage is not None and precision is not None and coverage + precision > 0:
            f1 = 2 * coverage * precision / (coverage + precision)
        team_accuracy = _safe_ratio(totals["team_correct"], totals["team_scored"])
        continuity = _safe_ratio(totals["continuity_correct"], totals["continuity_scored"])
        runtimes = [clip["runtime_s"] for clip in model_info["clips"] if clip.get("runtime_s") is not None]
        processed_fps = [clip["processed_fps"] for clip in model_info["clips"] if clip.get("processed_fps") is not None]

        coverage_ci = _bootstrap_ci(
            [v for v in per_model_frame_metric_values[preset]["coverage"].values()],
            args.bootstrap_iterations,
            rng,
        )
        precision_ci = _bootstrap_ci(
            [v for v in per_model_frame_metric_values[preset]["precision"].values()],
            args.bootstrap_iterations,
            rng,
        )

        summary_rows.append({
            "run_id": args.run_id,
            "manifest_id": args.manifest_id,
            "model_preset": preset,
            "role": model_info.get("role", ""),
            "gt_points": totals["gt"],
            "matched_points": totals["matched"],
            "coverage": _round(coverage),
            "coverage_ci_low": _round(coverage_ci[0]),
            "coverage_ci_high": _round(coverage_ci[1]),
            "false_positives": totals["false_positives"],
            "false_positive_per_gt": _round(_safe_ratio(totals["false_positives"], totals["gt"])),
            "precision": _round(precision),
            "precision_ci_low": _round(precision_ci[0]),
            "precision_ci_high": _round(precision_ci[1]),
            "f1": _round(f1),
            "team_accuracy": _round(team_accuracy),
            "continuity": _round(continuity),
            "mean_error_px": _round(_mean(all_distances), 3),
            "median_error_px": _round(_median(all_distances), 3),
            "avg_runtime_s": _round(_mean(runtimes), 3),
            "avg_processed_fps": _round(_mean(processed_fps), 3),
        })

    comparison_rows = []
    practical_thresholds = {
        "coverage": 0.05,
        "precision": 0.05,
        "team_accuracy": 0.05,
        "mean_error_px": 5.0,
    }
    if args.baseline in per_model_frame_metric_values:
        for preset in sorted(per_model_frame_metric_values):
            if preset == args.baseline:
                continue
            for metric, practical_delta in practical_thresholds.items():
                delta, low, high, n = _paired_bootstrap_delta(
                    per_model_frame_metric_values[args.baseline][metric],
                    per_model_frame_metric_values[preset][metric],
                    args.bootstrap_iterations,
                    rng,
                )
                separated = bool(low is not None and high is not None and (low > 0 or high < 0))
                meaningful = bool(delta is not None and abs(delta) >= practical_delta)
                comparison_rows.append({
                    "run_id": args.run_id,
                    "manifest_id": args.manifest_id,
                    "baseline": args.baseline,
                    "model_preset": preset,
                    "metric": metric,
                    "paired_frames": n,
                    "delta_model_minus_baseline": _round(delta, 4 if metric != "mean_error_px" else 3),
                    "delta_ci_low": _round(low, 4 if metric != "mean_error_px" else 3),
                    "delta_ci_high": _round(high, 4 if metric != "mean_error_px" else 3),
                    "bootstrap_separated_from_zero": separated,
                    "practically_meaningful": meaningful,
                    "comparison_flag": separated and meaningful,
                    "practical_threshold": practical_delta,
                })

    reports_base = EVAL_REPORTS_DIR / f"{args.manifest_id}_{args.run_id}"
    _write_csv(
        reports_base.with_name(reports_base.name + "_tracking_summary.csv"),
        summary_rows,
        [
            "run_id", "manifest_id", "model_preset", "role", "gt_points",
            "matched_points", "coverage", "coverage_ci_low", "coverage_ci_high",
            "false_positives", "false_positive_per_gt", "precision",
            "precision_ci_low", "precision_ci_high", "f1", "team_accuracy",
            "continuity", "mean_error_px", "median_error_px", "avg_runtime_s",
            "avg_processed_fps",
        ],
    )
    _write_csv(
        reports_base.with_name(reports_base.name + "_tracking_frame_metrics.csv"),
        frame_rows,
        [
            "run_id", "manifest_id", "model_preset", "frame_id", "clip_id",
            "frame_index", "timestamp_s", "kind", "sequence_id", "gt", "matched",
            "false_positives", "coverage", "precision", "team_accuracy",
            "mean_error_px", "continuity_scored", "continuity_correct",
        ],
    )
    _write_csv(
        reports_base.with_name(reports_base.name + "_tracking_details.csv"),
        detail_rows,
        [
            "run_id", "manifest_id", "model_preset", "frame_id", "clip_id",
            "frame_index", "timestamp_s", "kind", "sequence_id", "gt_id",
            "gt_team", "pred_track_id", "pred_team_raw", "pred_team_aligned",
            "distance_px", "team_correct",
        ],
    )
    _write_csv(
        reports_base.with_name(reports_base.name + "_tracking_comparisons.csv"),
        comparison_rows,
        [
            "run_id", "manifest_id", "baseline", "model_preset", "metric",
            "paired_frames", "delta_model_minus_baseline", "delta_ci_low",
            "delta_ci_high", "bootstrap_separated_from_zero",
            "practically_meaningful", "comparison_flag", "practical_threshold",
        ],
    )
    json_report = {
        "run_id": args.run_id,
        "manifest_id": args.manifest_id,
        "distance_threshold_px": args.distance_threshold,
        "baseline": args.baseline,
        "bootstrap_iterations": args.bootstrap_iterations,
        "summary": summary_rows,
        "comparisons": comparison_rows,
        "notes": {
            "coverage": "matched GT points / GT points, using Hungarian point matching within threshold.",
            "precision": "matched predictions / all scored predictions.",
            "f1": "harmonic mean of coverage and precision.",
            "team_accuracy": "team-correct matched predictions / matched predictions after per-clip A/B alignment.",
            "continuity": "same GT ID matched to same predicted track ID across consecutive annotated sequence frames.",
            "team_alignment": "Predicted A/B names are arbitrary, so the scorer chooses direct or swapped mapping per clip.",
            "significance_rule": "comparison_flag requires bootstrap CI excluding 0 and a practical delta at least as large as the metric threshold.",
            "caveat": "The current PSG set has 40 frames and short sequences; comparisons are useful engineering evidence, not final population statistics.",
        },
    }
    reports_base.with_name(reports_base.name + "_tracking_summary.json").write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[score] wrote reports under {EVAL_REPORTS_DIR}")


if __name__ == "__main__":
    main()
