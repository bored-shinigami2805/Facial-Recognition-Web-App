"""Unit tests for the pure functions in face_engine (no model loaded)."""

import numpy as np
import pytest

from app import config, face_engine


def _unit(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


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


def test_gallery_match_uses_best_embedding_per_person():
    # Person 1 has one bad and one good photo; person 2 one mediocre photo.
    # Taking each person's closest embedding lets person 1's good photo win,
    # even though their average distance is worse than person 2's.
    query = _unit(1, 0, 0)
    gallery = face_engine.Gallery.from_rows(
        [
            (1, _unit(0, 1, 0)),        # person 1, far
            (1, _unit(0.98, 0.2, 0)),   # person 1, close
            (2, _unit(0.7, 0.7, 0)),    # person 2, mid
        ]
    )
    pid, dist = gallery.match(query)
    assert pid == 1
    assert dist < 0.05


@pytest.mark.slow
def test_real_model_on_demo_images():
    demo = config.BASE_DIR / "app" / "static" / "demo"
    einstein = face_engine.detect_faces(face_engine.load_image((demo / "einstein.jpg").read_bytes()))
    lincoln = face_engine.detect_faces(face_engine.load_image((demo / "lincoln.jpg").read_bytes()))
    assert einstein and lincoln
    assert face_engine.distance(einstein[0].embedding, einstein[0].embedding) < 0.3
    assert face_engine.distance(einstein[0].embedding, lincoln[0].embedding) > 0.8
