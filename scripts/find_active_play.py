"""Find the best 20-second active-play window in a football clip.

Uses frame-difference motion analysis — no YOLO needed.
Active play = consistently high, smooth motion across the pitch.
Dead balls (corners, throw-ins, fouls, injuries) show as low motion
or sudden spikes followed by a pause.

Usage:
    python scripts/find_active_play.py <input.mp4> <output.mp4> <ffmpeg_path>
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


WINDOW_SEC = 20.0
ANALYSIS_STRIDE = 4   # analyse every 4th frame (fast, still accurate at 25fps)


def motion_scores(video_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (times_sec, scores, fps) for each sampled frame."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    times, scores = [], []
    prev = None

    for idx in range(0, total, ANALYSIS_STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        # Downscale + blur for speed and noise reduction
        small = cv2.resize(frame, (320, 180))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev is not None:
            diff = float(cv2.absdiff(gray, prev).mean())
        else:
            diff = 0.0

        times.append(idx / fps)
        scores.append(diff)
        prev = gray

    cap.release()
    return np.array(times), np.array(scores), fps


def find_best_window(times: np.ndarray, scores: np.ndarray, fps: float) -> float:
    """Return the start time (seconds) of the best 20-second active-play window.

    Scoring: reward high mean motion, penalise variance.
    High variance = sudden stop/restart = dead ball or set piece.
    Low mean motion = dead ball or slow build-up that stalls.
    """
    pts_per_window = max(1, int(WINDOW_SEC * fps / ANALYSIS_STRIDE))

    best_score = -1.0
    best_start = 0.0

    for i in range(len(scores) - pts_per_window):
        w = scores[i: i + pts_per_window]
        # Penalise any sub-windows with very low motion (ball dead)
        min_chunk = max(1, pts_per_window // 4)
        chunk_mins = [w[j: j + min_chunk].mean()
                      for j in range(0, pts_per_window - min_chunk, min_chunk)]
        weakest_chunk = min(chunk_mins) if chunk_mins else w.mean()

        # Score = mean motion  -  0.4 * std  +  0.3 * weakest_chunk
        # The weakest_chunk term rewards clips where EVERY part is active,
        # not just a burst followed by a pause.
        s = float(w.mean() - 0.4 * w.std() + 0.3 * weakest_chunk)
        if s > best_score:
            best_score = s
            best_start = float(times[i])

    return best_start


def trim(src: str, start: float, duration: float, dst: str, ffmpeg: str) -> None:
    subprocess.run(
        [ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", src,
         "-t", f"{duration:.3f}", "-c", "copy", dst],
        check=True, capture_output=True,
    )


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: find_active_play.py <input.mp4> <output.mp4> <ffmpeg>")
        sys.exit(1)

    src, dst, ffmpeg = sys.argv[1], sys.argv[2], sys.argv[3]

    print(f"Analysing {Path(src).name} …", flush=True)
    times, scores, fps = motion_scores(src)

    if len(times) < 10:
        print("  Clip too short — using start")
        start = 0.0
    else:
        start = find_best_window(times, scores, fps)

    end = start + WINDOW_SEC
    print(f"  Best active-play window: {start:.1f}s – {end:.1f}s")
    trim(src, start, WINDOW_SEC, dst, ffmpeg)
    print(f"  Saved → {dst}")


if __name__ == "__main__":
    main()
