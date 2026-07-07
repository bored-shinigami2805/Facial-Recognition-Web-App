"""App configuration.

I kept this deliberately small - just a few constants and a tiny helper to
read/write the runtime match threshold. A student project doesn't need a full
settings framework, so this is plain module-level values plus one JSON file for
the one setting the user is allowed to change at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
THUMB_DIR = DATA_DIR / "thumbnails"
DB_PATH = DATA_DIR / "faces.db"
DB_URL = f"sqlite:///{DB_PATH}"
CONFIG_FILE = DATA_DIR / "runtime_config.json"

# Make sure the folders exist on import so nothing else has to worry about it.
DATA_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# --- Face pipeline ---------------------------------------------------------
# We use InsightFace (ArcFace) which produces 512-d embeddings and we compare
# them with COSINE distance. The default threshold below is a reasonable
# starting point for the buffalo_l model; lower = stricter.
MODEL_NAME = "buffalo_l"          # InsightFace model pack (RetinaFace + ArcFace)
EMBEDDING_DIM = 512
DISTANCE_METRIC = "cosine"
DEFAULT_THRESHOLD = 0.45          # cosine distance; matches below this are accepted

THUMB_SIZE = 150                  # thumbnail width/height in px


def get_threshold() -> float:
    """Return the current match threshold, falling back to the default."""
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            return float(data.get("threshold", DEFAULT_THRESHOLD))
        except (json.JSONDecodeError, ValueError, OSError):
            # Corrupt config file - just ignore it and use the default.
            return DEFAULT_THRESHOLD
    return DEFAULT_THRESHOLD


def set_threshold(value: float) -> float:
    """Persist a new match threshold and return it."""
    value = float(value)
    CONFIG_FILE.write_text(json.dumps({"threshold": value}))
    return value
