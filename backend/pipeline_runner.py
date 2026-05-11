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
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.jobs import load_job, update_job
from src.config import CACHE_DIR, CLIPS_DIR, TRACKING_DIR


STAGES = [
    ("tracking",     "Spotting the players…"),
    ("assign_teams", "Mapping team colours…"),
    ("metrics",      "Computing spacing metrics…"),
    ("visualizer",   "Rendering tactical overlay…"),
    ("explainer",    "Generating coaching insights…"),
]


# ── Auto-tune for long videos ──────────────────────────────────────────────
# YOLO inference scales with frame count. A 1.5h match at 25 fps is 135k
# frames — even on the dev RTX 4070 that's ~20 min just for tracking with
# the x-large model. We auto-pick a stride and a smaller model so long
# clips actually finish in a session. Short clips keep the high-quality
# defaults.
#
# (duration_s_threshold, vid_stride, model_name) — first row whose
# threshold exceeds the actual duration wins. ``None`` model means "use
# whatever is in src.config.YOLO_MODEL".
AUTO_TUNE_TABLE = (
    (60,    1, None),           # < 1 min: full quality
    (180,   2, None),           # 1–3 min: light skip, same model
    (600,   3, "yolo11n.pt"),   # 3–10 min: nano model + stride 3
    (1800,  5, "yolo11n.pt"),   # 10–30 min: aggressive skip
    (float("inf"), 8, "yolo11n.pt"),  # 30+ min: maximum throughput
)


def _auto_tune(duration_s: float) -> tuple[int, str | None]:
    for limit, stride, model in AUTO_TUNE_TABLE:
        if duration_s < limit:
            return stride, model
    # Unreachable because the last row uses ``inf`` — but keeps mypy happy.
    return 1, None


def _probe_codec(video_path: Path) -> str:
    """Return the video codec name (e.g. 'av1', 'h264', 'vp9')."""
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    return result.stdout.strip().lower()


def _transcode_to_h264(src: Path) -> Path:
    """Re-encode to H.264 in-place so OpenCV can decode it.

    AV1 and some VP9 containers are not decodable by OpenCV's default
    ffmpeg build on Linux. We transcode once before tracking; the output
    overwrites the source so the rest of the pipeline is unaware.
    """
    tmp = src.with_suffix(".tmp.mp4")
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "copy", str(tmp)],
        check=True, capture_output=True,
    )
    tmp.replace(src)
    print(f"[runner] Transcoded {src.name} to H.264")


def _probe_duration(video_path: Path) -> float:
    """Return clip duration in seconds without running tracking yet."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return nframes / fps if fps else 0.0


def _set_stage(job_id: str, idx: int) -> None:
    update_job(job_id,
               status=STAGES[idx][0],
               stage_index=idx,
               stage_message=STAGES[idx][1])


def _stage_tracking(job_id: str, video_path: Path,
                    vid_stride: int = 1, model_name: str | None = None) -> Path:
    from src.tracking import run_tracking
    data = run_tracking(video_path, model_name=model_name, vid_stride=vid_stride)
    out = TRACKING_DIR / f"{job_id}.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _stage_assign_teams(tracking_path: Path, video_path: Path) -> None:
    # Drop YOLO duplicate-box artefacts before clustering jersey colours so
    # the k-means fit isn't biased by counting one player twice.
    from scripts.dedupe_tracking import process as _dedupe
    _dedupe(tracking_path)
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
    render_overlay(video_path, tracking_path, out)
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
    update_job(
        job_id,
        probed_duration_s=round(probed_dur, 1),
        vid_stride=vid_stride,
        tracking_model=model_name or "default",
    )
    print(
        f"[runner] auto-tune: duration~{probed_dur:.1f}s "
        f"-> vid_stride={vid_stride} model={model_name or 'default'}"
    )

    try:
        # AV1-encoded uploads can't be decoded by OpenCV — transcode first.
        codec = _probe_codec(video_path)
        if codec in ("av1", "vp9"):
            update_job(job_id, stage_message=f"Transcoding {codec.upper()} video…")
            _transcode_to_h264(video_path)

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

        update_job(job_id,
                   status="done",
                   stage_index=len(STAGES),
                   stage_message="Analysis complete.")
        print(f"[runner] Job {job_id} done.")

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        update_job(job_id,
                   status="error",
                   error=f"{type(e).__name__}: {e}",
                   stage_message="Pipeline failed — see error.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Job ID to process")
    args = parser.parse_args()
    run(args.job)


if __name__ == "__main__":
    main()
