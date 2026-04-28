"""Project-wide configuration. Import from here, never hard-code paths."""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLIPS_DIR = DATA_DIR / "clips"
TRACKING_DIR = DATA_DIR / "tracking"
CACHE_DIR = DATA_DIR / "cache"
PROMPTS_DIR = ROOT / "prompts"

for d in (CLIPS_DIR, TRACKING_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Video / pitch -----------------------------------------------------
DEFAULT_FPS = 25  # most YouTube broadcast clips
# Standard pitch dimensions in metres. Used to calibrate pixel→metre
# conversions later. For now we work in pixel space and label as "px".
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# --- Tracking ----------------------------------------------------------
YOLO_MODEL = "yolo26x.pt"   # YOLO26 extra-large — NMS-free, 43% faster CPU than yolo11x
TRACKER_CONFIG = "bytetrack.yaml"
PERSON_CLASS_ID = 0   # COCO class id for "person"

# --- Border filter (remove ball boys / coaches at frame edges) ----------
# Fractions of frame width/height. Detections whose centroid falls inside
# the exclusion zone are dropped before tracking.
BORDER_X    = 0.06   # 6% from left and right edges
BORDER_TOP  = 0.06   # 6% from top edge
BORDER_BOT  = 0.13   # 13% from bottom edge (ball boys sit here)

# --- LLM ---------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
