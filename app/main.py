"""FastAPI application: routes, static serving and the glue between the web
layer, the database and the face engine.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import base64
import logging
import os
import secrets
import threading
import uuid
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sqlalchemy import func
from sqlalchemy.orm import Session

from . import config, db, face_engine, schemas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("facematch")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # On the hosted demo we auto-enroll a sample face in the background so the
    # gallery isn't empty on first visit. Gated behind an env var so it never
    # runs locally or during tests.
    if os.getenv("FACEMATCH_SEED_DEMO") == "1":
        threading.Thread(target=_seed_demo_people, daemon=True).start()
    yield


app = FastAPI(title="Face Recognition Demo", version="1.0", lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# --- Access control ---------------------------------------------------------
# Face embeddings are biometric data, so on any public deployment the whole app
# should sit behind a login. If ADMIN_PASSWORD is set (e.g. as a Hugging Face
# Space secret) every request needs HTTP Basic auth; if it's unset (local dev
# and tests) the app stays open. Set it in production and you're protected even
# if the hosting is public.
_ADMIN_USER = os.getenv("ADMIN_USER", "admin")
_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")


def _check_basic_auth(header: str | None) -> bool:
    """Return True if the Authorization header holds the admin credentials."""
    if not header or not header.startswith("Basic "):
        return False
    try:
        user, _, pw = base64.b64decode(header[6:]).decode("utf-8").partition(":")
    except Exception:
        return False
    # constant-time comparison to avoid leaking the password via timing
    return secrets.compare_digest(user, _ADMIN_USER) and secrets.compare_digest(
        pw, _ADMIN_PASSWORD
    )


@app.middleware("http")
async def _require_login(request: Request, call_next):
    """Gate every route (API, SPA and thumbnails) behind Basic auth when a
    password is configured. No password configured -> open (local use)."""
    if _ADMIN_PASSWORD and not _check_basic_auth(request.headers.get("Authorization")):
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="FaceMatch"'},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Gallery cache
# ---------------------------------------------------------------------------
# The gallery matrix is expensive to rebuild from SQLite, so we keep one in
# memory and drop it whenever the enrolled set changes.
_gallery: face_engine.Gallery | None = None


def _rebuild_gallery(session: Session) -> face_engine.Gallery:
    global _gallery
    rows = [
        (e.person_id, face_engine.embedding_from_bytes(e.vector))
        for e in session.query(db.Embedding).all()
    ]
    _gallery = face_engine.Gallery.from_rows(rows)
    return _gallery


def _get_gallery(session: Session) -> face_engine.Gallery:
    if _gallery is None:
        return _rebuild_gallery(session)
    return _gallery


def _invalidate_gallery() -> None:
    global _gallery
    _gallery = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _save_thumbnail(rgb_image, box: tuple[int, int, int, int]) -> str:
    """Crop the face box, make a square thumbnail, save it and return its URL."""
    x1, y1, x2, y2 = box
    h, w = rgb_image.shape[:2]
    # clamp to image bounds (detector boxes can spill slightly off-image)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = Image.fromarray(rgb_image[y1:y2, x1:x2])
    crop.thumbnail((config.THUMB_SIZE, config.THUMB_SIZE))
    fname = f"{uuid.uuid4().hex}.png"
    crop.save(config.THUMB_DIR / fname)
    return f"/thumbnails/{fname}"


def _read_upload(upload: UploadFile) -> bytes:
    if not (upload.content_type or "").startswith("image/"):
        raise HTTPException(400, f"'{upload.filename}' is not an image.")
    limit = config.MAX_UPLOAD_BYTES
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = upload.file.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, f"'{upload.filename}' exceeds the {limit // (1024 * 1024)} MB limit.")
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(400, f"File '{upload.filename}' is empty.")
    return b"".join(chunks)


# Which bundled demo photo to auto-enroll on the hosted demo. We only seed
# Einstein so the "Lincoln" sample still demonstrates an "Unknown" result.
_DEMO_SEED = {"einstein.jpg": "Albert Einstein"}


def _seed_demo_people() -> None:
    """Enroll the bundled demo face(s) once, so the hosted demo works on first
    visit. Runs in a background thread (the first detect also downloads the
    model), no-ops if anyone is already enrolled, and never raises."""
    try:
        session = db.SessionLocal()
        if session.query(db.Person).count() > 0:
            session.close()
            return
        for fname, name in _DEMO_SEED.items():
            fpath = STATIC_DIR / "demo" / fname
            if not fpath.exists():
                continue
            rgb = face_engine.load_image(fpath.read_bytes())
            faces = face_engine.detect_faces(rgb)
            if not faces:
                continue
            person = db.Person(name=name)
            session.add(person)
            session.flush()
            session.add(
                db.Embedding(
                    person_id=person.id,
                    vector=face_engine.embedding_to_bytes(faces[0].embedding),
                    thumb_path=_save_thumbnail(rgb, faces[0].bbox),
                )
            )
        session.commit()
        session.close()
        _invalidate_gallery()
    except Exception:
        log.exception("demo seed failed")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.post("/api/enroll", response_model=schemas.EnrollResponse)
def enroll(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    session: Session = Depends(db.get_session),
):
    """Enroll a person: largest face per photo, photos with no face are skipped."""
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name must not be empty.")
    if len(files) > config.MAX_ENROLL_FILES:
        raise HTTPException(400, f"Too many files; {config.MAX_ENROLL_FILES} max per enroll.")

    # reuse an existing person with the same name, otherwise create one
    person = session.query(db.Person).filter(func.lower(db.Person.name) == name.lower()).first()
    created_new = person is None
    if created_new:
        person = db.Person(name=name)
        session.add(person)
        session.flush()  # get person.id without committing yet

    enrolled = 0
    notes: list[str] = []
    for upload in files:
        data = _read_upload(upload)  # size/content-type errors abort the request
        try:
            rgb = face_engine.load_image(data)
        except ValueError as exc:
            notes.append(f"{upload.filename}: {exc}")
            continue

        faces = face_engine.detect_faces(rgb)
        if not faces:
            notes.append(f"{upload.filename}: no face found, skipped")
            continue
        if len(faces) > 1:
            notes.append(f"{upload.filename}: {len(faces)} faces, used the largest")

        face = faces[0]  # detect_faces() returns largest-first
        thumb = _save_thumbnail(rgb, face.bbox)
        session.add(
            db.Embedding(
                person_id=person.id,
                vector=face_engine.embedding_to_bytes(face.embedding),
                thumb_path=thumb,
            )
        )
        enrolled += 1

    if enrolled == 0:
        # nothing usable, roll back so we don't leave an empty person behind
        session.rollback()
        detail = "No faces could be enrolled. " + " | ".join(notes)
        raise HTTPException(422, detail)

    session.commit()
    _invalidate_gallery()
    msg = f"Enrolled {enrolled} face(s) for '{name}'."
    if notes:
        msg += " Notes: " + " | ".join(notes)
    return schemas.EnrollResponse(
        person_id=person.id, name=name, faces_enrolled=enrolled, message=msg
    )


@app.post("/api/recognize", response_model=schemas.RecognizeResponse)
def recognize(
    file: UploadFile = File(...),
    threshold: float | None = Form(None),
    session: Session = Depends(db.get_session),
):
    """Detect every face in the uploaded photo and match against the gallery."""
    data = _read_upload(file)
    try:
        rgb = face_engine.load_image(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    thr = float(threshold) if threshold is not None else config.DEFAULT_THRESHOLD

    gallery = _get_gallery(session)
    id_to_name = {p.id: p.name for p in session.query(db.Person).all()}

    faces = face_engine.detect_faces(rgb)
    matches: list[dict] = []
    for f in faces:
        matched_id, dist = gallery.match(f.embedding)
        has_match = matched_id is not None and dist <= thr
        matches.append(
            {
                "box": list(f.bbox),
                "name": id_to_name.get(matched_id, "Unknown") if has_match else "Unknown",
                "person_id": matched_id if has_match else None,
                "distance": round(dist, 4) if len(gallery) else None,
                "confidence": face_engine.confidence_from_distance(dist) if len(gallery) else None,
                "det_score": round(f.det_score, 3),
            }
        )

    best = min((m["distance"] for m in matches if m["distance"] is not None), default=None)
    log.info(
        "recognize: faces=%d threshold=%.2f best_distance=%s decisions=%s",
        len(faces), thr, f"{best:.4f}" if best is not None else "n/a",
        [m["name"] for m in matches],
    )
    return schemas.RecognizeResponse(
        faces_found=len(faces),
        threshold=thr,
        matches=[schemas.Match(**m) for m in matches],
    )


@app.get("/api/people", response_model=list[schemas.PersonOut])
def list_people(session: Session = Depends(db.get_session)):
    people = session.query(db.Person).order_by(db.Person.name).all()
    out = []
    for p in people:
        thumb = p.embeddings[0].thumb_path if p.embeddings else None
        out.append(
            schemas.PersonOut(
                id=p.id, name=p.name, image_count=len(p.embeddings), thumbnail=thumb
            )
        )
    return out


@app.delete("/api/people/{person_id}")
def delete_person(person_id: int, session: Session = Depends(db.get_session)):
    """Delete a person, their embeddings and thumbnail files."""
    person = session.get(db.Person, person_id)
    if person is None:
        raise HTTPException(404, "Person not found.")

    # best-effort cleanup of thumbnail files on disk
    for emb in person.embeddings:
        if emb.thumb_path:
            fpath = config.DATA_DIR / emb.thumb_path.lstrip("/")
            try:
                fpath.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not delete thumbnail %s: %s", fpath, exc)

    session.delete(person)
    session.commit()
    _invalidate_gallery()
    return {"deleted": person_id}


@app.get("/api/config", response_model=schemas.ConfigOut)
def get_config():
    return schemas.ConfigOut(
        threshold=config.DEFAULT_THRESHOLD,
        model=config.MODEL_NAME,
        metric=config.DISTANCE_METRIC,
    )


# ---------------------------------------------------------------------------
# Static files / SPA
# ---------------------------------------------------------------------------
# Thumbnails are served straight from the data folder.
app.mount("/thumbnails", StaticFiles(directory=str(config.THUMB_DIR)), name="thumbnails")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# Mount the rest of the SPA assets (app.js, styles.css) at /static.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
