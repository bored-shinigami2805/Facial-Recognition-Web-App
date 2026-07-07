"""FastAPI application: routes, static serving and the glue between the web
layer, the database and the face engine.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import base64
import io
import os
import threading
import uuid
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import func
from sqlalchemy.orm import Session

from . import config, db, face_engine, schemas


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create the SQLite tables on startup
    db.init_db()
    # On the hosted demo we auto-enroll a sample face in the background so the
    # gallery isn't empty on first visit. Gated behind an env var so it never
    # runs locally or during tests.
    if os.getenv("FACEMATCH_SEED_DEMO") == "1":
        threading.Thread(target=_seed_demo_people, daemon=True).start()
    yield


app = FastAPI(title="Face Recognition Demo", version="1.0", lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent / "static"


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


def _annotate(rgb_image, matches: list[dict]) -> str:
    """Draw boxes + labels on the image and return a base64 PNG data URL."""
    img = Image.fromarray(rgb_image).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for m in matches:
        x1, y1, x2, y2 = m["box"]
        known = m["name"] != "Unknown"
        color = (34, 197, 94) if known else (239, 68, 68)  # green / red
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = m["name"]
        if m["distance"] is not None:
            label += f"  ({m['distance']:.2f})"
        # label background so text stays readable
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0, y1 - th - 6)
        draw.rectangle([x1, ly, x1 + tw + 8, ly + th + 6], fill=color)
        draw.text((x1 + 4, ly + 3), label, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _read_upload(upload: UploadFile) -> bytes:
    data = upload.file.read()
    if not data:
        raise HTTPException(400, f"File '{upload.filename}' is empty.")
    return data


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
    except Exception:
        pass  # seeding is best-effort; never crash startup over it


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.post("/api/enroll", response_model=schemas.EnrollResponse)
def enroll(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    session: Session = Depends(db.get_session),
):
    """Enroll a person from one or more photos.

    For each photo we detect faces. If a photo has several faces we take the
    largest one (and note it in the message) rather than failing the whole
    upload; photos with no detectable face are skipped and reported.
    """
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name must not be empty.")

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
        try:
            data = _read_upload(upload)
            rgb = face_engine.load_image(data)
        except (ValueError, HTTPException) as exc:
            notes.append(f"{upload.filename}: {getattr(exc, 'detail', str(exc))}")
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
        # nothing usable - roll back so we don't leave an empty person behind
        session.rollback()
        detail = "No faces could be enrolled. " + " | ".join(notes)
        raise HTTPException(422, detail)

    session.commit()
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

    thr = float(threshold) if threshold is not None else config.get_threshold()

    # load the whole gallery into memory - fine for a small demo
    rows = session.query(db.Embedding).all()
    gallery = [(e.person_id, face_engine.embedding_from_bytes(e.vector)) for e in rows]
    id_to_name = {p.id: p.name for p in session.query(db.Person).all()}

    faces = face_engine.detect_faces(rgb)
    matches: list[dict] = []
    for f in faces:
        matched_id, dist = face_engine.nearest(f.embedding, gallery)
        has_match = matched_id is not None and dist <= thr
        # distance/confidence are only meaningful when there's something to
        # compare against (an empty gallery gives no nearest neighbour).
        matches.append(
            {
                "box": list(f.bbox),
                "name": id_to_name.get(matched_id, "Unknown") if has_match else "Unknown",
                "person_id": matched_id if has_match else None,
                "distance": round(dist, 4) if gallery else None,
                "confidence": face_engine.confidence_from_distance(dist) if gallery else None,
                "det_score": round(f.det_score, 3),
            }
        )

    annotated = _annotate(rgb, matches)
    return schemas.RecognizeResponse(
        faces_found=len(faces),
        threshold=thr,
        matches=[schemas.Match(**m) for m in matches],
        annotated_image=annotated,
    )


@app.get("/api/people", response_model=list[schemas.PersonOut])
def list_people(session: Session = Depends(db.get_session)):
    """List enrolled people with an image count and one thumbnail each."""
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
            except OSError:
                pass  # TODO: log this instead of silently ignoring

    session.delete(person)
    session.commit()
    return {"deleted": person_id}


@app.get("/api/config", response_model=schemas.ConfigOut)
def get_config():
    return schemas.ConfigOut(
        threshold=config.get_threshold(),
        model=config.MODEL_NAME,
        metric=config.DISTANCE_METRIC,
    )


@app.post("/api/config", response_model=schemas.ConfigOut)
def update_config(body: schemas.ConfigIn):
    if not (0.0 < body.threshold < 2.0):
        raise HTTPException(400, "Threshold must be between 0 and 2 (cosine distance).")
    config.set_threshold(body.threshold)
    return schemas.ConfigOut(
        threshold=config.get_threshold(),
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
