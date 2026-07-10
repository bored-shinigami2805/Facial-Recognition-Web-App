"""Face pipeline: detect, embed and compare.

Backed by InsightFace's buffalo_l pack (RetinaFace detection + ArcFace 512-d
embeddings, cosine distance). The ~280 MB pack downloads on first use and is
cached in ~/.insightface. This is the only module that imports insightface.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from . import config

log = logging.getLogger(__name__)

# Cap image size to guard against decompression-bomb uploads.
Image.MAX_IMAGE_PIXELS = 50_000_000

# insightface is imported lazily so importing this module stays cheap.
_app = None


def _get_app():
    """Load and cache the InsightFace app (CPU only)."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis

        log.info("loading InsightFace model %s (first call downloads the pack)", config.MODEL_NAME)
        app = FaceAnalysis(
            name=config.MODEL_NAME,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _app = app
    return _app


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    embedding: np.ndarray             # L2-normalised, shape (512,)
    det_score: float

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def load_image(data: bytes) -> np.ndarray:
    """Decode image bytes to an RGB numpy array, applying EXIF rotation."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise ValueError("Image is too large to process.") from exc
    except Exception as exc:
        raise ValueError("Could not read image file (is it a valid image?)") from exc
    return np.array(img)


def detect_faces(rgb_image: np.ndarray) -> list[DetectedFace]:
    """Detect faces in an RGB image, largest first, each with its embedding."""
    app = _get_app()
    bgr = rgb_image[:, :, ::-1]  # insightface expects BGR
    faces = app.get(bgr)

    results: list[DetectedFace] = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int).tolist()
        results.append(
            DetectedFace(
                bbox=(x1, y1, x2, y2),
                embedding=f.normed_embedding.astype(np.float32),
                det_score=float(f.det_score),
            )
        )
    results.sort(key=lambda d: d.area, reverse=True)
    return results


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalised embeddings (0 = identical)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(1.0 - np.dot(a, b))


class Gallery:
    """All enrolled embeddings held as one (N, 512) matrix for fast matching."""

    def __init__(self, person_ids: np.ndarray, matrix: np.ndarray):
        self._person_ids = person_ids
        self._matrix = matrix

    @classmethod
    def from_rows(cls, rows: list[tuple[int, np.ndarray]]) -> "Gallery":
        if rows:
            ids = np.array([pid for pid, _ in rows], dtype=np.int64)
            matrix = np.vstack([emb for _, emb in rows]).astype(np.float32)
        else:
            ids = np.empty(0, dtype=np.int64)
            matrix = np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)
        return cls(ids, matrix)

    def __len__(self) -> int:
        return int(self._person_ids.shape[0])

    def match(self, query: np.ndarray) -> tuple[int | None, float]:
        """Nearest enrolled (person_id, distance) using each person's closest embedding."""
        if len(self) == 0:
            return None, float("inf")
        q = np.asarray(query, dtype=np.float32)
        dists = 1.0 - self._matrix @ q
        best_id, best_dist = None, float("inf")
        for pid in np.unique(self._person_ids):
            d = float(dists[self._person_ids == pid].min())
            if d < best_dist:
                best_id, best_dist = int(pid), d
        return best_id, best_dist


def confidence_from_distance(d: float) -> float:
    """Map a distance to a 0..1 display confidence (not a real probability)."""
    return round(max(0.0, min(1.0, 1.0 - d / 2.0)), 3)


def embedding_to_bytes(emb: np.ndarray) -> bytes:
    """Serialise an embedding to raw float32 bytes for SQLite storage."""
    return np.asarray(emb, dtype=np.float32).tobytes()


def embedding_from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
