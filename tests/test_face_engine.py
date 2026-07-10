"""Unit tests for the maths in face_engine.

These deliberately avoid loading the actual InsightFace model (that needs a
~300MB download and is slow), so they only cover the pure functions:
distance, nearest-neighbour search and embedding (de)serialisation.
"""

import numpy as np

from app import face_engine


def _unit(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_distance_identical_is_zero():
    v = _unit(1, 2, 3)
    assert abs(face_engine.distance(v, v)) < 1e-6


def test_distance_orthogonal_is_one():
    a = np.array([1, 0], dtype=np.float32)
    b = np.array([0, 1], dtype=np.float32)
    assert abs(face_engine.distance(a, b) - 1.0) < 1e-6


def test_embedding_roundtrip():
    emb = np.random.rand(512).astype(np.float32)
    blob = face_engine.embedding_to_bytes(emb)
    back = face_engine.embedding_from_bytes(blob)
    assert np.allclose(emb, back)


def test_gallery_match_picks_closest():
    query = _unit(1, 0, 0)
    gallery = face_engine.Gallery.from_rows(
        [
            (1, _unit(0, 1, 0)),
            (2, _unit(0.9, 0.1, 0)),
            (3, _unit(-1, 0, 0)),
        ]
    )
    pid, dist = gallery.match(query)
    assert pid == 2
    assert dist < 0.05


def test_gallery_match_empty():
    pid, dist = face_engine.Gallery.from_rows([]).match(_unit(1, 0))
    assert pid is None
    assert dist == float("inf")


def test_confidence_is_clamped():
    assert face_engine.confidence_from_distance(0.0) == 1.0
    assert 0.0 <= face_engine.confidence_from_distance(2.0) <= 1.0
    assert face_engine.confidence_from_distance(5.0) == 0.0
