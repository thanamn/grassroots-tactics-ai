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
# Football-specific YOLOv8 weights (huggingface: uisikdag/yolo-v8-football-players-detection).
# Classes: 0=ball, 1=goalkeeper, 2=player, 3=referee. We track only outfield
# players + GK so the convex hull is built from the team, not officials.
YOLO_MODEL = str(ROOT / "models" / "football_players.pt")
TRACKER_CONFIG = "bytetrack.yaml"
TRACK_CLASSES = [1, 2]   # goalkeeper + player; deliberately excludes ball and referee

# --- LLM ---------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
