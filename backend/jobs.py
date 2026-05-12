"""Job persistence — one JSON blob per upload under data/jobs/.

Why JSON-on-disk and not a real DB
----------------------------------
Single-user prototype, week-4 timeline. A SQLite/Postgres setup would
add migrations, ORM glue, and a deployment story we don't need yet.
File-per-job means the pipeline subprocess and the FastAPI server can
read and write status without coordinating through a DB connection;
each side just opens the same file. Trade-off: no transactional
guarantees, so concurrent writes from the API and the runner could
clobber each other. The runner is the only writer for status fields
during a job's lifetime, and the API only writes at create-time, so
in practice the windows don't overlap.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "data" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def create_job(*, filename: str, video_size: int = 0,
               session_type: str = "match",
               opponent: str | None = None,
               notes: str | None = None,
               language: str = "en") -> dict[str, Any]:
    """Allocate a new job_id and persist an initial 'queued' record."""
    job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "job_id": job_id,
        "filename": filename,
        "session_type": session_type,
        "opponent": opponent,
        "notes": notes,
        "language": language,
        "video_size": video_size,
        "duration_s": None,
        "fps": None,
        "status": "queued",
        "stage_message": "Queued — starting up…",
        "stage_index": 0,
        "stage_total": 5,
        "pipeline_started_at": None,
        "stage_started_at": None,
        "probed_duration_s": None,
        "estimated_total_s": None,
        "stage_estimates": None,
        "estimate_source": None,
        "estimate_sample_count": 0,
        "estimate_scale": None,
        "stage_progress": None,
        "stage_timings": {},
        "actual_total_s": None,
        "error": None,
        "explainer_error": None,
        "events_count": None,
        "summary": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    save_job(job)
    return job


def save_job(job: dict[str, Any]) -> None:
    job["updated_at"] = _now_iso()
    _path(job["job_id"]).write_text(
        json.dumps(job, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_job(job_id: str) -> dict[str, Any] | None:
    p = _path(job_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def update_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
    job = load_job(job_id)
    if not job:
        return None
    job.update(fields)
    save_job(job)
    return job


def list_jobs() -> list[dict[str, Any]]:
    """Return every job, newest first."""
    out: list[dict[str, Any]] = []
    for p in JOBS_DIR.glob("*.json"):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            # A half-written file from a crashed runner — skip rather than
            # 500 the dashboard. The runner will rewrite it on next update.
            continue
    out.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return out


def delete_job(job_id: str) -> bool:
    p = _path(job_id)
    if not p.exists():
        return False
    p.unlink()
    return True
