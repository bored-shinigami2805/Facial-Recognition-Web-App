"""Paths, model settings and upload limits."""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
THUMB_DIR = DATA_DIR / "thumbnails"
DB_PATH = DATA_DIR / "faces.db"
DB_URL = f"sqlite:///{DB_PATH}"

DATA_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# --- Face pipeline ---------------------------------------------------------
# We use InsightFace (ArcFace) which produces 512-d embeddings and we compare
# them with COSINE distance. The default threshold below is a reasonable
# starting point for the buffalo_l model; lower = stricter.
MODEL_NAME = "buffalo_l"          # InsightFace model pack (RetinaFace + ArcFace)
EMBEDDING_DIM = 512
DISTANCE_METRIC = "cosine"

# Default match threshold (cosine distance; matches below this are accepted).
# 0.77 is the accuracy-optimal point from eval/threshold_sweep.py on LFW.
# Override per deployment with the FACEMATCH_THRESHOLD env var.
DEFAULT_THRESHOLD = float(os.getenv("FACEMATCH_THRESHOLD", "0.77"))

THUMB_SIZE = 150                  # thumbnail width/height in px

# --- Upload limits ---------------------------------------------------------
MAX_UPLOAD_BYTES = 8 * 1024 * 1024   # 8 MB per image
MAX_ENROLL_FILES = 10                # photos accepted in one enroll call
MAX_NAME_LEN = 80                    # longest accepted person name
