"""Face pipeline: detect -> embed -> compare.

This is the ONLY module that knows which face library we use underneath. The
rest of the app talks to the small interface defined here (`detect_faces`,
`get_embeddings`, `distance`, ...), so if I ever swapped InsightFace for
face_recognition/dlib or DeepFace I'd only rewrite this file.

Currently backed by InsightFace's `buffalo_l` pack:
  - detection:   RetinaFace
  - recognition: ArcFace -> 512-d embedding
  - matching:    cosine distance between L2-normalised embeddings

The model is loaded lazily the first time it's needed (the first call is slow
because InsightFace downloads the ~300MB model pack, after that it's cached in
~/.insightface).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from . import config

# The heavy import (insightface) is done lazily inside _get_app() so that just
# importing this module (e.g. in a quick unit test) stays cheap.
_app = None


def _get_app():
    """Load and cache the InsightFace model. CPU-only, which is fine for a demo."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=config.MODEL_NAME,
            providers=["CPUExecutionProvider"],
            # We only need detection + recognition, skip the extra models
            # (landmarks/age/gender) to keep it lighter and faster.
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 -> CPU
        _app = app
    return _app


@dataclass
class DetectedFace:
    """One detected face: pixel bounding box + its 512-d embedding."""

    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    embedding: np.ndarray             # L2-normalised, shape (512,)
    det_score: float                  # detector confidence 0..1

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def load_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB numpy array.

    Uses Pillow (not cv2.imdecode) so we can also apply EXIF orientation - phone
    photos are often rotated and this fixes that. Raises ValueError on garbage.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)   # respect phone rotation
        img = img.convert("RGB")
    except Exception as exc:  # Pillow raises a bunch of different errors
        raise ValueError("Could not read image file (is it a valid image?)") from exc
    return np.array(img)


def detect_faces(rgb_image: np.ndarray) -> list[DetectedFace]:
    """Detect all faces in an RGB image and compute an embedding for each.

    Returns a list sorted largest-face-first, which is handy for the enroll
    flow where we want "the main subject" of the photo.
    """
    app = _get_app()
    # InsightFace expects BGR (OpenCV convention); our array is RGB.
    bgr = rgb_image[:, :, ::-1]
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
    """Cosine distance between two embeddings (0 = identical, 2 = opposite).

    Embeddings from InsightFace are already L2-normalised, so cosine similarity
    is just their dot product and cosine distance is 1 - that.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(1.0 - np.dot(a, b))


def nearest(embedding: np.ndarray, gallery: list[tuple[int, np.ndarray]]):
    """Find the nearest gallery entry to `embedding`.

    `gallery` is a list of (person_id, embedding). Returns
    (person_id, distance) for the closest entry, or (None, inf) if the gallery
    is empty.
    """
    best_id, best_dist = None, float("inf")
    for pid, emb in gallery:
        d = distance(embedding, emb)
        if d < best_dist:
            best_id, best_dist = pid, d
    return best_id, best_dist


def confidence_from_distance(d: float) -> float:
    """Turn a cosine distance into a friendly 0..1 'confidence' for the UI.

    This is just a linear-ish mapping for display, not a real probability -
    distance is what actually drives the accept/reject decision.
    """
    return round(max(0.0, min(1.0, 1.0 - d / 2.0)), 3)


def embedding_to_bytes(emb: np.ndarray) -> bytes:
    """Serialise an embedding to raw float32 bytes for storage in SQLite."""
    return np.asarray(emb, dtype=np.float32).tobytes()


def embedding_from_bytes(blob: bytes) -> np.ndarray:
    """Inverse of embedding_to_bytes."""
    return np.frombuffer(blob, dtype=np.float32)
