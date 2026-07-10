"""API tests using FastAPI's TestClient.

The real face model is monkeypatched out so these run fast and deterministically
- we're testing the web/DB plumbing (enroll -> store -> recognize -> match),
not the neural network itself. The database is swapped for an in-memory SQLite
via a dependency override.
"""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import db, face_engine, main


def _png_bytes(color=(128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (120, 120), color).save(buf, "PNG")
    return buf.getvalue()


def _fake_face(seed=0.0):
    """A deterministic unit embedding standing in for a real detected face."""
    emb = np.zeros(512, dtype=np.float32)
    emb[int(seed) % 512] = 1.0
    return face_engine.DetectedFace(bbox=(10, 10, 60, 60), embedding=emb, det_score=0.99)


@pytest.fixture
def client(monkeypatch):
    # in-memory DB shared across threads/connections
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db.Base.metadata.create_all(engine)

    def override_session():
        s = TestingSession()
        try:
            yield s
        finally:
            s.close()

    main.app.dependency_overrides[db.get_session] = override_session
    # start each test with a fresh (empty) gallery cache
    monkeypatch.setattr(main, "_gallery", None)
    # default fake: every image has exactly one face with embedding "0"
    monkeypatch.setattr(face_engine, "detect_faces", lambda rgb: [_fake_face(0)])

    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


def test_config_get(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    assert set(r.json()) == {"threshold", "model", "metric"}


def test_enroll_then_recognize_match(client):
    r = client.post(
        "/api/enroll",
        data={"name": "Alice"},
        files={"files": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["faces_enrolled"] == 1

    # gallery now has Alice
    people = client.get("/api/people").json()
    assert len(people) == 1 and people[0]["name"] == "Alice"
    assert people[0]["image_count"] == 1

    # recognizing the same embedding should match Alice (distance ~0)
    r = client.post(
        "/api/recognize",
        data={"threshold": 0.45},
        files={"file": ("q.png", _png_bytes(), "image/png")},
    )
    body = r.json()
    assert r.status_code == 200
    assert body["faces_found"] == 1
    assert body["matches"][0]["name"] == "Alice"
    assert body["annotated_image"].startswith("data:image/png;base64,")


def test_recognize_unknown_when_gallery_empty(client):
    r = client.post(
        "/api/recognize",
        files={"file": ("q.png", _png_bytes(), "image/png")},
    )
    body = r.json()
    assert body["faces_found"] == 1
    assert body["matches"][0]["name"] == "Unknown"
    assert body["matches"][0]["distance"] is None


def test_enroll_rejects_photo_with_no_face(client, monkeypatch):
    monkeypatch.setattr(face_engine, "detect_faces", lambda rgb: [])  # no faces
    r = client.post(
        "/api/enroll",
        data={"name": "Nobody"},
        files={"files": ("blank.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 422


def test_auth_gates_everything_when_password_set(client, monkeypatch):
    # by default (no password) the app is open
    assert client.get("/api/people").status_code == 200

    # once a password is configured, every request needs Basic auth
    monkeypatch.setattr(main, "_ADMIN_PASSWORD", "s3cret")
    assert client.get("/api/people").status_code == 401
    assert client.get("/").status_code == 401  # the SPA is gated too

    good = base64.b64encode(b"admin:s3cret").decode()
    assert client.get("/api/people", headers={"Authorization": f"Basic {good}"}).status_code == 200

    bad = base64.b64encode(b"admin:wrong").decode()
    assert client.get("/api/people", headers={"Authorization": f"Basic {bad}"}).status_code == 401


def test_delete_person(client):
    client.post(
        "/api/enroll",
        data={"name": "Bob"},
        files={"files": ("b.png", _png_bytes(), "image/png")},
    )
    pid = client.get("/api/people").json()[0]["id"]
    assert client.delete(f"/api/people/{pid}").status_code == 200
    assert client.get("/api/people").json() == []
    assert client.delete(f"/api/people/{pid}").status_code == 404
