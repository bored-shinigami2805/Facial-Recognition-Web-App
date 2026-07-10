---
title: FaceMatch
emoji: 🙂
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# FaceMatch — a face recognition web app

FaceMatch is a small web app that recognises faces. You **enroll** people by name
with a few photos, then **recognize** a new photo (or a webcam capture): it finds
every face, compares each one to the people you've enrolled, and labels the ones
it's confident about — everything else comes back as *Unknown*. I built it as a
portfolio project to learn the classic **detect → embed → match** face-recognition
pipeline end to end, with a clean single-page UI on top of a Python API.

> UI: a top nav (**Recognize / Enroll / Gallery**), a circular face-scan frame,
> a match/identity card with confidence, and a recent-scans list — light theme,
> card-based, keyboard/mouse friendly.

## What it does

- **Recognize** — upload a photo or capture from your webcam. Every face is
  detected and turned into an embedding, matched against the gallery by distance,
  and shown with an annotated image, a match/confidence card, and a running
  history. Faces further than the threshold are marked *Unknown*.
- **Enroll** — give a name and one or more clear photos. Each photo's face is
  detected, embedded and stored so the person can be recognised later.
- **Gallery** — see everyone enrolled (name, photo count, thumbnail) and delete
  people you no longer want.
- **Threshold** — a slider lets you tune how strict matching is, live.

## Tech stack & language

| Layer | Choice |
|-------|--------|
| **Language** | Python 3.10+ (backend) &middot; vanilla JavaScript (frontend) |
| **Web framework** | FastAPI + Uvicorn |
| **Face pipeline** | [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` on `onnxruntime` (CPU) — RetinaFace detection + ArcFace 512-d embeddings |
| **Matching** | cosine distance, nearest-neighbour over the stored gallery |
| **Database** | SQLite via SQLAlchemy |
| **Frontend** | one static HTML page + CSS + vanilla JS (`fetch`) — no build step, no framework |
| **Image handling** | Pillow (decode, EXIF-rotate, draw boxes, thumbnails) |
| **Tests** | pytest |

> **Why InsightFace?** The original plan was `face_recognition` (dlib) with a
> DeepFace fallback. On my machine (Python 3.13, Windows, no C++ compiler) dlib
> has no prebuilt wheel and won't compile, and DeepFace drags in a ~1GB
> TensorFlow install. InsightFace installs as a pure-Python wheel + onnxruntime
> and gives the same kind of ArcFace 512-d embeddings — and it all sits behind
> `app/face_engine.py`, so swapping the library later means editing just that one
> file.

## Setup & run

```bash
# 1. create + activate a virtual env
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the server
uvicorn app.main:app --reload

# 4. open the app in your browser
#    http://127.0.0.1:8000
```

The **first** recognise/enroll request downloads the InsightFace model pack
(~280MB) into `~/.insightface`; that's a one-time download and it's cached after.

If you're updating an existing checkout, delete `data/faces.db` once: the schema
gained a unique index on names and `create_all()` doesn't migrate in place.

There are two public-domain sample photos wired into the UI ("Try a sample:
Einstein / Lincoln") so you can test recognition immediately without a webcam.

## How it works

The pipeline has three stages. **Face detection** (RetinaFace) finds each face in
an image and returns a bounding box. For every detected face we compute a **face
embedding** — a 512-number feature vector from the ArcFace model that captures the
identity of the face, trained so that photos of the same person land close
together and different people land far apart. To recognise a face we do
**nearest-neighbour matching**: we measure the **cosine distance** from the new
embedding to every stored embedding and take the closest one. A **recognition
threshold** then decides accept vs reject — if the nearest distance is below the
threshold we label it with that person, otherwise it's *Unknown*. Lowering the
threshold is stricter (higher precision, more Unknowns); raising it is more
lenient (higher recall, more false matches) — the classic precision/recall
tradeoff, which the slider in the UI lets you feel.

## Project layout

```
app/
  main.py        # FastAPI app: routes, static serving, box drawing, thumbnails
  face_engine.py # detect + embed + compare (the ONLY file tied to InsightFace)
  db.py          # SQLAlchemy models (Person, Embedding) + session
  schemas.py     # pydantic response models
  config.py      # paths, model name, threshold get/set
  static/        # index.html, app.js, styles.css, demo/ sample images
data/            # sqlite db + thumbnails (gitignored, created on first run)
tests/           # test_face_engine.py, test_api.py
samples/         # public-domain faces for manual/curl testing (gitignored)
```

## API

| Method | Path                 | What it does                                |
|--------|----------------------|---------------------------------------------|
| POST   | `/api/enroll`        | form-data `name` + `files`; enroll a person |
| POST   | `/api/recognize`     | form-data `file` (+ optional `threshold`)   |
| GET    | `/api/people`        | list enrolled people                        |
| DELETE | `/api/people/{id}`   | delete a person + their embeddings          |
| GET/POST | `/api/config`      | read / update the match threshold           |
| GET    | `/`                  | the single-page app                         |

## Testing

Run the automated tests:

```bash
pytest
```

The tests mock out the neural network (so they're fast and need no model
download) and check the web/DB plumbing (enroll → store → recognize → match) plus
the matching maths.

Quick manual smoke test with `curl` (server running):

```bash
curl -X POST http://127.0.0.1:8000/api/enroll \
     -F "name=Albert Einstein" -F "files=@samples/personA.jpg"
curl -X POST http://127.0.0.1:8000/api/recognize -F "file=@samples/personA_variant.jpg"  # match
curl -X POST http://127.0.0.1:8000/api/recognize -F "file=@samples/personB.jpg"          # Unknown
```

When I ran this the same person came back at distance **~0.06** and a different
person at **~0.94** (threshold 0.5) — a nice clean separation.

## Deployment

The app is containerised for **Hugging Face Spaces** (Docker SDK). The included
`Dockerfile` runs it on port 7860 as a non-root user and sets
`FACEMATCH_SEED_DEMO=1`, which auto-enrols a sample face on first boot so the
hosted gallery isn't empty. To deploy:

1. Create a new Space at huggingface.co (SDK: **Docker**, blank).
2. Push this repo to the Space's git remote (or link the GitHub repo).
3. Wait for the build; the ArcFace model downloads on the first request.

Storage on Spaces is ephemeral, so the gallery resets when the Space restarts —
fine for a public demo.

## Limitations (being honest)

- The embeddings come from a **pretrained** ArcFace model — I did not train any
  network myself; I'm using the concepts and a good off-the-shelf model.
- It **struggles with extreme pose, poor lighting, occlusion** (masks, hands,
  sunglasses) and very low-resolution faces, like any single-embedding system.
- It's a **small-gallery demo**: recognition is a linear scan over all stored
  embeddings and the threshold isn't tuned for thousands of people.
- Webcam results depend heavily on camera quality and lighting.
- **This is a learning project — not for any security, surveillance or
  access-control use.** Please don't use it to make decisions about real people.

## Possible improvements

- Use a heavier/GPU detector for harder images.
- Index the gallery with **FAISS** so it scales past a linear scan.
- Add **liveness / anti-spoofing** so a photo of a photo doesn't pass.
- Average **multiple embeddings per person** for a more robust profile.

## License & credits

© 2026 **Ehina**. All rights reserved.

Sample faces are public-domain historical portraits from Wikimedia Commons. The
face detection/recognition models are pretrained weights from the InsightFace
project, used under their respective licenses.
