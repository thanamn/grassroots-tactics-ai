"""Download 5 UCL tactical-cam clips, auto-trim to the best 20-second active-play window.

Run:
    .venv\\Scripts\\python.exe scripts\\download_active_clips.py
"""
from __future__ import annotations

import subprocess, sys, shutil, uuid
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

# ── config ──────────────────────────────────────────────────────────────────

OUT_DIR = Path(r"C:\Users\TSURUGI\Desktop\football_clips")
YTDLP   = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "yt-dlp.exe"
FFMPEG  = imageio_ffmpeg.get_ffmpeg_exe()
PYTHON  = sys.executable

WINDOW_SEC = 20.0
STRIDE     = 4       # analyse every 4th frame

MATCHES = [
    {"name": "liverpool_vs_psg",       "url": "https://www.youtube.com/watch?v=Rfylh5wwReI", "t": "25:00-26:30"},
    {"name": "benfica_vs_realmadrid",  "url": "https://www.youtube.com/watch?v=hOcWLSkTYrk", "t": "22:00-23:30"},
    {"name": "psg_vs_newcastle",       "url": "https://www.youtube.com/watch?v=VsgzWWSr1QE", "t": "30:00-31:30"},
    {"name": "psg_vs_arsenal",         "url": "https://www.youtube.com/watch?v=S-MGti2APuE", "t": "28:00-29:30"},
    {"name": "psg_vs_intermilan",      "url": "https://www.youtube.com/watch?v=LxfN4yBJ4K4", "t": "35:00-36:30"},
]

# ── motion analysis ─────────────────────────────────────────────────────────

def analyse_motion(video_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    cap  = cv2.VideoCapture(str(video_path))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    times, scores, prev = [], [], None

    for idx in range(0, total, STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (320, 180))
        gray  = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        scores.append(float(cv2.absdiff(gray, prev).mean()) if prev is not None else 0.0)
        times.append(idx / fps)
        prev = gray

    cap.release()
    return np.array(times), np.array(scores), fps


def best_window_start(times: np.ndarray, scores: np.ndarray, fps: float) -> float:
    """Return start-time (s) of the most consistently active 20-second window."""
    n = max(1, int(WINDOW_SEC * fps / STRIDE))
    if len(scores) <= n:
        return 0.0

    best_s, best_t = -1.0, 0.0
    chunk = max(1, n // 4)

    for i in range(len(scores) - n):
        w = scores[i: i + n]
        # Penalise chunks with near-zero motion (dead ball hiding inside the window)
        chunk_mins = [w[j: j + chunk].mean() for j in range(0, n - chunk, chunk)]
        weakest    = min(chunk_mins) if chunk_mins else w.mean()
        score      = float(w.mean() - 0.4 * w.std() + 0.3 * weakest)
        if score > best_s:
            best_s, best_t = score, float(times[i])

    return best_t


# ── helpers ─────────────────────────────────────────────────────────────────

def download_tmp(url: str, time_range: str, tmp: Path) -> bool:
    # Use a template so yt-dlp can handle video+audio merging without
    # complaining about a fixed output name for multiple streams.
    tmp.unlink(missing_ok=True)
    tmp_template = tmp.parent / (tmp.stem + ".%(ext)s")
    result = subprocess.run([
        str(YTDLP),
        "--ffmpeg-location", FFMPEG,
        "--download-sections", f"*{time_range}",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", str(tmp_template),
        url,
    ], capture_output=True, text=True)
    # yt-dlp exits non-zero on warnings (e.g. JS runtime) even after a
    # successful download — check file existence instead of return code.
    merged = tmp.parent / (tmp.stem + ".mp4")
    if merged.exists() and merged != tmp:
        merged.rename(tmp)
    if not tmp.exists():
        print("  ERROR:", result.stderr[-500:])
        return False
    return True


def trim(src: Path, start: float, dst: Path) -> None:
    subprocess.run([
        FFMPEG, "-y",
        "-ss", f"{start:.3f}",
        "-i",  str(src),
        "-t",  f"{WINDOW_SEC:.3f}",
        "-c",  "copy",
        str(dst),
    ], check=True, capture_output=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for m in MATCHES:
        name = m["name"]
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")

        # Use a unique temp name so a locked leftover from a previous run
        # never blocks the current one (Windows Defender sometimes holds files).
        tmp = OUT_DIR / f"_tmp_{name}_{uuid.uuid4().hex[:8]}.mp4"
        out = OUT_DIR / f"{name}_active.mp4"

        if out.exists():
            print("  Already done — skipping")
            continue

        # 1. download raw window
        print(f"  Downloading {m['t']} ...")
        if not download_tmp(m["url"], m["t"], tmp):
            print("  SKIP — download failed")
            continue

        # 2. motion analysis
        print("  Analysing motion ...")
        times, scores, fps = analyse_motion(tmp)
        start = best_window_start(times, scores, fps)
        print(f"  Best window: {start:.1f}s – {start + WINDOW_SEC:.1f}s  "
              f"(mean motion {scores.mean():.2f})")

        # 3. trim
        trim(tmp, start, out)
        print(f"  Saved: {out.name}  ({out.stat().st_size // 1024} KB)")

        # 4. delete temp
        tmp.unlink(missing_ok=True)

    print("\nDone. Files in:", OUT_DIR)


if __name__ == "__main__":
    main()
