"""FastAPI server for the Grassroots Tactics AI live-pipeline web product.

Run:
    .venv\\Scripts\\uvicorn backend.main:app --reload --port 8000

Endpoints
---------
    GET  /                       → web/index.html (the React SPA)
    GET  /web/<asset>            → static frontend assets
    POST /api/jobs               → upload video (multipart) + start pipeline
    GET  /api/jobs               → list every job, newest first
    GET  /api/jobs/{id}          → status of one job
    GET  /api/jobs/{id}/result   → full metrics + explanation (only when done)
    GET  /api/jobs/{id}/overlay  → the rendered overlay MP4
    POST /api/jobs/{id}/explain  → re-run Gemini explainer for one language
    POST /api/jobs/{id}/chat     → free-form Q&A over the spacing metrics
    DELETE /api/jobs/{id}        → remove job + all artefacts

Upload size cap is enforced server-side: anything beyond MAX_UPLOAD_MB is
413'd mid-stream so we don't fill the disk on a runaway upload. The 10-min
clip limit talked about in the UI is enforced client-side (we'd need to
probe the file with ffprobe to enforce it server-side, and that adds a
dependency we'd rather not need yet).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.jobs import (
    create_job, delete_job, list_jobs, load_job, save_job,
)
from src.config import CACHE_DIR, CLIPS_DIR

WEB_DIR = ROOT / "web"
MAX_UPLOAD_MB = 500   # honest cap given CPU pipeline; ~10–15 min HD


app = FastAPI(title="Grassroots Tactics AI", version="0.1.0")

# CORS only matters in dev when the frontend is served from a different
# origin than the API (e.g. Vite on :5173). In production we serve both
# from this same uvicorn, so this is mostly belt-and-braces.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API ─────────────────────────────────────────────────────────────────────

@app.post("/api/jobs")
async def upload_job(
    video: UploadFile = File(...),
    session_type: str = Form("match"),
    opponent: str | None = Form(None),
    notes: str | None = Form(None),
    language: str = Form("en"),
):
    """Save the uploaded video, create a job record, kick off the pipeline."""
    if not video.filename:
        raise HTTPException(400, "No filename provided")

    job = create_job(
        filename=video.filename,
        session_type=session_type,
        opponent=opponent,
        notes=notes,
        language=language,
    )
    job_id = job["job_id"]
    dest = CLIPS_DIR / f"{job_id}.mp4"

    # Stream-save to disk so we never load the whole video into memory.
    # Cap at MAX_UPLOAD_MB; if exceeded we abort mid-write and clean up.
    cap = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with open(dest, "wb") as fh:
        while True:
            chunk = await video.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            size += len(chunk)
            if size > cap:
                fh.close()
                dest.unlink(missing_ok=True)
                delete_job(job_id)
                raise HTTPException(
                    413,
                    f"Upload exceeds {MAX_UPLOAD_MB} MB cap — "
                    f"long videos won't finish on CPU during a session.",
                )
            fh.write(chunk)

    job["video_size"] = size
    save_job(job)

    # Spawn the pipeline runner. Don't wait — return immediately so the
    # frontend can switch into its progress screen and start polling.
    subprocess.Popen(
        [sys.executable, "-m", "backend.pipeline_runner", "--job", job_id],
        cwd=str(ROOT),
        # New process group on Windows so Ctrl-C in the uvicorn shell
        # doesn't also kill the running pipeline (and vice versa).
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    return job


@app.get("/api/jobs")
def all_jobs():
    return list_jobs()


@app.get("/api/jobs/{job_id}")
def one_job(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str, lang: str = "en"):
    """Return the full per-frame metrics + the explanation for one language.

    409 if the job hasn't finished yet — frontend should keep polling
    /api/jobs/{job_id} for status until status == 'done', THEN call this.
    """
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"Job not ready (status: {job['status']})")

    metrics_path = CACHE_DIR / f"{job_id}_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(500, "Metrics file missing despite job=done")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    explanation_path = CACHE_DIR / f"{job_id}_explanation_{lang}.json"
    explanation = (
        json.loads(explanation_path.read_text(encoding="utf-8"))
        if explanation_path.exists() else None
    )
    return {"job": job, "metrics": metrics, "explanation": explanation}


@app.get("/api/jobs/{job_id}/overlay")
def job_overlay(job_id: str):
    p = CACHE_DIR / f"{job_id}_overlay.mp4"
    if not p.exists():
        raise HTTPException(404, "Overlay not ready")
    return FileResponse(p, media_type="video/mp4")


@app.post("/api/jobs/{job_id}/explain")
def regenerate_explanation(job_id: str, lang: str = "en"):
    """Re-run the Gemini explainer for a single language.

    Useful when the original pipeline call hit a transient 503 — the
    user can hit "Retry" in the UI rather than re-uploading the video.
    Synchronous because the call is cheap (one API request) and the
    user is actively waiting on it; no need for a job/queue layer.
    """
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if lang not in ("en", "th"):
        raise HTTPException(400, "lang must be 'en' or 'th'")
    metrics_path = CACHE_DIR / f"{job_id}_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(409, "Metrics not ready")

    from src.explainer import explain
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    try:
        result = explain(metrics, lang=lang)
    except Exception as e:  # noqa: BLE001
        # Surface the failure to the job record so the dashboard reflects it.
        job["explainer_error"] = f"{type(e).__name__}: {e}"
        save_job(job)
        raise HTTPException(502, f"Gemini call failed: {e}")

    out = CACHE_DIR / f"{job_id}_explanation_{lang}.json"
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Clear any prior error now that at least one language succeeded.
    job["explainer_error"] = None
    save_job(job)
    return result


@app.post("/api/jobs/{job_id}/chat")
def chat_endpoint(job_id: str, payload: dict = Body(...)):
    """Free-form Q&A grounded in this job's spacing metrics.

    Payload: {"question": str, "history": [{"role": "user"|"assistant", "text": str}, ...], "lang": "en"|"th"}
    Stateless on the server — the frontend holds the full conversation
    in component state and sends it back each turn. This keeps server
    state tiny (one job JSON) and lets the user "clear chat" purely
    client-side without a DELETE.
    """
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    metrics_path = CACHE_DIR / f"{job_id}_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(409, "Metrics not ready")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    if len(question) > 1000:
        raise HTTPException(413, "question too long")
    history = payload.get("history") or []
    lang = payload.get("lang") or "en"
    if lang not in ("en", "th"):
        raise HTTPException(400, "lang must be 'en' or 'th'")

    from src.coach_chat import chat
    try:
        answer = chat(metrics, question, history=history, lang=lang)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Gemini call failed: {e}")
    return {"answer": answer}


@app.delete("/api/jobs/{job_id}")
def remove_job(job_id: str):
    if not delete_job(job_id):
        raise HTTPException(404, "Job not found")
    # Best-effort artefact cleanup — never fails the request.
    for p in (
        CLIPS_DIR / f"{job_id}.mp4",
        CACHE_DIR / f"{job_id}_metrics.json",
        CACHE_DIR / f"{job_id}_overlay.mp4",
        CACHE_DIR / f"{job_id}_explanation_en.json",
        CACHE_DIR / f"{job_id}_explanation_th.json",
        ROOT / "data" / "tracking" / f"{job_id}.json",
    ):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    return {"ok": True}


# ── Frontend ───────────────────────────────────────────────────────────────

# Mount /web/ as static files. index.html is served from / for cleanliness.
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.get("/")
def root():
    idx = WEB_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(503, "Frontend not built. Expected web/index.html")
    return FileResponse(idx)
