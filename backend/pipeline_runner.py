"""Run the full tactical-spacing pipeline for one uploaded job.

Invoked as a subprocess from backend.main:

    python -m backend.pipeline_runner --job <job_id>

It is its own process so YOLO inference (the slowest stage) can't lock
up the FastAPI worker. The runner updates the job JSON between stages
so the frontend can poll /api/jobs/<id> and show progress.

Pipeline stages:
    0. tracking      — YOLO + centroid tracker → data/tracking/<job_id>.json
    1. assign_teams  — k-means jersey colours  → tracking JSON in-place
    2. metrics       — hull / centroid / events → data/cache/<job_id>_metrics.json
    3. visualizer    — overlay MP4 → data/cache/<job_id>_overlay.mp4
    4. explainer     — DeepSeek → data/cache/<job_id>_explanation_<lang>.json
                       (runs once for en, once for th — both languages
                       are pre-rendered so the language toggle on the
                       analysis screen is instant)

The explainer step is wrapped so a missing DEEPSEEK_API_KEY (or transient
network failure) doesn't fail the whole job — the metrics + overlay are
still useful without the AI text.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.jobs import list_jobs, load_job, update_job
from src.config import CACHE_DIR, CLIPS_DIR, TRACKING_DIR


STAGES = [
    ("tracking",     "Spotting the players…"),
    ("assign_teams", "Mapping team colours…"),
    ("metrics",      "Computing spacing metrics…"),
    ("visualizer",   "Rendering tactical overlay…"),
    ("explainer",    "Generating coaching insights…"),
]

STAGE_LABELS = ("tracking", "assign_teams", "metrics", "visualizer", "explainer")


# ── Auto-tune for long videos ──────────────────────────────────────────────
# YOLO inference scales with frame count. A 1.5h match at 25 fps is 135k
# frames, so long clips still need frame skipping. Keep the football-specific
# model, though: the generic nano detector is fast but misses far too many
# broadcast players for stable team shapes.
#
# (duration_s_threshold, vid_stride, model_name) — first row whose
# threshold exceeds the actual duration wins. ``None`` model means "use
# whatever is in src.config.YOLO_MODEL".
AUTO_TUNE_TABLE = (
    (60,    1, None),           # < 1 min: full quality
    (180,   2, None),           # 1–3 min: light skip, same model
    (600,   3, None),           # 3–10 min: stride 3, same football model
    (1800,  5, None),           # 10–30 min: aggressive skip, same model
    (float("inf"), 8, None),    # 30+ min: maximum throughput, same model
)


def _auto_tune(duration_s: float) -> tuple[int, str | None]:
    for limit, stride, model in AUTO_TUNE_TABLE:
        if duration_s < limit:
            return stride, model
    # Unreachable because the last row uses ``inf`` — but keeps mypy happy.
    return 1, None


def _probe_duration(video_path: Path) -> float:
    """Return clip duration in seconds without running tracking yet."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return nframes / fps if fps else 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _seconds_since(value: str | None) -> float | None:
    started = _parse_iso(value)
    if not started:
        return None
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _infer_total_seconds(job: dict) -> float | None:
    actual = job.get("actual_total_s")
    if isinstance(actual, (int, float)) and actual > 0:
        return float(actual)
    if job.get("status") != "done":
        return None
    start = _parse_iso(job.get("pipeline_started_at") or job.get("created_at"))
    end = _parse_iso(job.get("updated_at"))
    if not start or not end:
        return None
    seconds = (end - start).total_seconds()
    return seconds if seconds > 0 else None


def _baseline_stage_seconds(duration_s: float, vid_stride: int) -> dict[str, float]:
    tracked_seconds = duration_s / max(1, vid_stride)
    estimates = {
        "tracking": max(40.0, tracked_seconds * 6.5),
        "assign_teams": max(8.0, tracked_seconds * 0.65),
        "metrics": max(3.0, tracked_seconds * 0.08),
        "visualizer": max(20.0, duration_s * 1.55),
        "explainer": 18.0,
    }
    estimates["total"] = round(sum(estimates.values()), 1)
    return {k: round(v, 1) for k, v in estimates.items()}


def _history_runtime_scale() -> tuple[float, int]:
    ratios: list[float] = []
    for job in list_jobs():
        duration_s = job.get("probed_duration_s") or job.get("duration_s")
        if not isinstance(duration_s, (int, float)) or duration_s <= 0:
            continue
        actual_s = _infer_total_seconds(job)
        if not actual_s:
            continue
        baseline = _baseline_stage_seconds(float(duration_s), int(job.get("vid_stride") or 1))["total"]
        if baseline <= 0:
            continue
        ratios.append(max(0.4, min(3.0, actual_s / baseline)))

    if len(ratios) < 3:
        return 1.0, len(ratios)

    # Slightly conservative: local runs vary with thermals / model cache /
    # explainer latency, and under-estimating feels worse than being early.
    scale = statistics.median(ratios) * 1.10
    return max(0.75, min(2.50, scale)), len(ratios)


def _estimate_stage_seconds(duration_s: float, vid_stride: int) -> tuple[dict[str, float], dict[str, float | int | str]]:
    estimates = _baseline_stage_seconds(duration_s, vid_stride)
    scale, sample_count = _history_runtime_scale()
    source = "local_history" if sample_count >= 3 else "duration_model"
    if source == "local_history":
        estimates = {
            key: round(value * scale, 1)
            for key, value in estimates.items()
            if key != "total"
        }
        estimates["total"] = round(sum(estimates.values()), 1)

    return estimates, {
        "estimate_source": source,
        "estimate_sample_count": sample_count,
        "estimate_scale": round(scale, 3),
    }


def _stage_timings_with_current(job_id: str) -> dict[str, float]:
    job = load_job(job_id) or {}
    timings = dict(job.get("stage_timings") or {})
    current = job.get("status")
    elapsed = _seconds_since(job.get("stage_started_at"))
    if current in STAGE_LABELS and elapsed is not None:
        timings[current] = round(elapsed, 1)
    return timings


def _set_stage(job_id: str, idx: int) -> None:
    timings = _stage_timings_with_current(job_id)
    update_job(job_id,
               status=STAGES[idx][0],
               stage_index=idx,
               stage_message=STAGES[idx][1],
               stage_started_at=_now_iso(),
               stage_progress=None,
               stage_timings=timings)


def _update_stage_progress(
    job_id: str,
    stage: str,
    *,
    progress: float,
    elapsed_s: float,
    estimated_remaining_s: float,
    processed: int | None = None,
    total: int | None = None,
) -> None:
    job = load_job(job_id) or {}
    stage_estimates = dict(job.get("stage_estimates") or {})
    stage_estimates[stage] = round(max(elapsed_s + estimated_remaining_s, elapsed_s), 1)

    try:
        stage_pos = STAGE_LABELS.index(stage)
    except ValueError:
        stage_pos = 0
    future_s = sum(float(stage_estimates.get(name, 0) or 0) for name in STAGE_LABELS[stage_pos + 1:])
    elapsed_total_s = _seconds_since(job.get("pipeline_started_at")) or 0.0
    estimated_total_s = max(
        elapsed_total_s + estimated_remaining_s + future_s,
        elapsed_total_s + 1.0,
    )

    update_job(
        job_id,
        stage_progress={
            "stage": stage,
            "progress": round(max(0.0, min(1.0, progress)), 4),
            "elapsed_s": round(elapsed_s, 1),
            "estimated_remaining_s": round(max(0.0, estimated_remaining_s), 1),
            "processed": processed,
            "total": total,
        },
        stage_estimates=stage_estimates,
        estimated_total_s=round(estimated_total_s, 1),
    )


def _stage_tracking(job_id: str, video_path: Path,
                    vid_stride: int = 1, model_name: str | None = None) -> Path:
    from src.tracking import run_tracking

    def progress_callback(info: dict) -> None:
        _update_stage_progress(
            job_id,
            "tracking",
            progress=float(info.get("progress", 0.0)),
            elapsed_s=float(info.get("elapsed_s", 0.0)),
            estimated_remaining_s=float(info.get("estimated_remaining_s", 0.0)),
            processed=info.get("processed"),
            total=info.get("total"),
        )

    data = run_tracking(
        video_path,
        model_name=model_name,
        vid_stride=vid_stride,
        progress_callback=progress_callback,
    )
    out = TRACKING_DIR / f"{job_id}.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _stage_assign_teams(tracking_path: Path, video_path: Path) -> None:
    from scripts.assign_teams import assign_teams
    assign_teams(tracking_path, video_path)


def _stage_metrics(job_id: str, tracking_path: Path) -> Path:
    from src.metrics import compute_metrics
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    metrics = compute_metrics(tracking)
    out = CACHE_DIR / f"{job_id}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out


def _stage_visualizer(job_id: str, video_path: Path, tracking_path: Path) -> Path:
    from src.visualizer import render_overlay
    out = CACHE_DIR / f"{job_id}_overlay.mp4"

    def progress_callback(info: dict) -> None:
        _update_stage_progress(
            job_id,
            "visualizer",
            progress=float(info.get("progress", 0.0)),
            elapsed_s=float(info.get("elapsed_s", 0.0)),
            estimated_remaining_s=float(info.get("estimated_remaining_s", 0.0)),
            processed=info.get("processed"),
            total=info.get("total"),
        )

    render_overlay(video_path, tracking_path, out, progress_callback=progress_callback)
    return out


def _stage_explainer(job_id: str, metrics_path: Path) -> None:
    """Run the Gemini explainer for both languages.

    Both are pre-rendered so the analysis-screen language toggle is
    free at view time. If one language fails (rate limit, partial outage)
    we still keep the other.
    """
    from src.explainer import explain
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    last_error: Exception | None = None
    for lang in ("en", "th"):
        try:
            result = explain(metrics, lang=lang)
            p = CACHE_DIR / f"{job_id}_explanation_{lang}.json"
            p.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001 — surface the real reason
            last_error = e
            traceback.print_exc()
    if last_error is not None:
        # Both could have failed (no API key) or just one. Either way,
        # surface the most recent error onto the job record but DON'T
        # mark the job itself as failed — metrics + overlay are still
        # serviceable without the AI text.
        update_job(job_id, explainer_error=f"{type(last_error).__name__}: {last_error}")


def run(job_id: str) -> None:
    job = load_job(job_id)
    if not job:
        print(f"[runner] Job {job_id} not found", file=sys.stderr)
        sys.exit(1)

    video_path = CLIPS_DIR / f"{job_id}.mp4"
    if not video_path.exists():
        update_job(job_id,
                   status="error",
                   error=f"Uploaded video missing: {video_path}",
                   stage_message="Pipeline aborted — video file missing.")
        sys.exit(1)

    # Probe duration once up front so the dashboard can show "fast mode"
    # before the heavy tracking call starts. We can't fully trust this
    # number (some webm-in-mp4 containers report wrong frame counts) but
    # for the auto-tune decision it's good enough; the real duration
    # gets corrected later from the metrics step.
    probed_dur = _probe_duration(video_path)
    vid_stride, model_name = _auto_tune(probed_dur)
    stage_estimates, estimate_meta = _estimate_stage_seconds(probed_dur, vid_stride)
    update_job(
        job_id,
        probed_duration_s=round(probed_dur, 1),
        vid_stride=vid_stride,
        tracking_model=model_name or "default",
        pipeline_started_at=_now_iso(),
        estimated_total_s=stage_estimates["total"],
        stage_estimates={k: stage_estimates[k] for k in STAGE_LABELS},
        **estimate_meta,
    )
    print(
        f"[runner] auto-tune: duration~{probed_dur:.1f}s "
        f"-> vid_stride={vid_stride} model={model_name or 'default'}"
    )

    try:
        _set_stage(job_id, 0)
        tracking_path = _stage_tracking(
            job_id, video_path, vid_stride=vid_stride, model_name=model_name,
        )

        _set_stage(job_id, 1)
        _stage_assign_teams(tracking_path, video_path)

        _set_stage(job_id, 2)
        metrics_path = _stage_metrics(job_id, tracking_path)

        # Surface a tiny summary into the job record so the dashboard
        # can show stats (events count, duration) without re-reading
        # the whole metrics JSON.
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        update_job(job_id,
                   duration_s=metrics.get("duration_s"),
                   fps=metrics.get("fps"),
                   events_count=len(metrics.get("events", [])),
                   summary=metrics.get("summary"))

        _set_stage(job_id, 3)
        _stage_visualizer(job_id, video_path, tracking_path)

        _set_stage(job_id, 4)
        _stage_explainer(job_id, metrics_path)

        actual_total_s = _seconds_since((load_job(job_id) or {}).get("pipeline_started_at"))
        update_job(job_id,
                   status="done",
                   stage_index=len(STAGES),
                   stage_message="Analysis complete.",
                   stage_progress=None,
                   stage_timings=_stage_timings_with_current(job_id),
                   actual_total_s=round(actual_total_s, 1) if actual_total_s is not None else None)
        print(f"[runner] Job {job_id} done.")

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        update_job(job_id,
                   status="error",
                   error=f"{type(e).__name__}: {e}",
                   stage_message="Pipeline failed — see error.",
                   stage_progress=None,
                   stage_timings=_stage_timings_with_current(job_id))
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Job ID to process")
    args = parser.parse_args()
    run(args.job)


if __name__ == "__main__":
    main()
