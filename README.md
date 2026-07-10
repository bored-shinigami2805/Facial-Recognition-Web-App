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

<!-- REWRITE -->
FaceMatch is a small web app that recognises faces. You **enroll** people by name
with a few photos, then **recognize** a new photo (or a webcam capture): it finds
every face, compares each one to the people you've enrolled, and labels the ones
it's confident about — everything else comes back as *Unknown*. I built it as a
portfolio project to learn the classic **detect → embed → match** face-recognition
pipeline end to end, with a clean single-page UI on top of a Python API.

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

- Language: Python 3.13 (backend), vanilla JavaScript (frontend)
- Web framework: FastAPI + Uvicorn
- Face pipeline: [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` on `onnxruntime` (CPU) — RetinaFace detection + ArcFace 512-d embeddings
- Matching: cosine distance, nearest-neighbour over the stored gallery
- Database: SQLite via SQLAlchemy
- Frontend: one static HTML page + CSS + vanilla JS (`fetch`), no build step
- Image handling: Pillow (decode, EXIF-rotate, thumbnails)
- Tests: pytest

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

<!-- REWRITE -->
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
  main.py        # FastAPI app: routes, static serving, thumbnails
  face_engine.py # detect + embed + compare (the ONLY file tied to InsightFace)
  db.py          # SQLAlchemy models (Person, Embedding) + session
  schemas.py     # pydantic response models
  config.py      # paths, model settings, upload limits
  static/        # index.html, app.js, styles.css, demo/ sample images
data/            # sqlite db + thumbnails (gitignored, created on first run)
eval/            # threshold_sweep.py + the ROC / histogram PNGs
tests/           # test_face_engine.py, test_api.py, conftest.py
```

## API

- `POST /api/enroll` — form-data `name` + `files`; enroll a person
- `POST /api/recognize` — form-data `file` (+ optional `threshold`)
- `GET /api/people` — list enrolled people
- `DELETE /api/people/{id}` — delete a person and their embeddings
- `GET /api/config` — read the default threshold and model info
- `GET /` — the single-page app

## Testing

Run the automated tests:

```bash
pytest
```

The fast tests mock out the neural network and check the web/DB plumbing and the
`Gallery` matching. One integration test runs the real model on the bundled demo
images; it's marked `slow` and skipped unless you pass `--run-slow`.

Quick manual smoke test with `curl` (server running):

```bash
curl -X POST http://127.0.0.1:8000/api/enroll \
     -F "name=Albert Einstein" -F "files=@app/static/demo/einstein.jpg"
curl -X POST http://127.0.0.1:8000/api/recognize -F "file=@app/static/demo/einstein.jpg"  # match
curl -X POST http://127.0.0.1:8000/api/recognize -F "file=@app/static/demo/lincoln.jpg"    # Unknown
```

The default threshold of **0.77** comes from `eval/threshold_sweep.py`, which
scores ROC AUC **0.98** on the LFW test pairs (see below).

## Threshold selection

The default match threshold comes from `eval/threshold_sweep.py`, which runs the
pipeline over the LFW test pairs, sweeps the cosine-distance cutoff, and picks the
accuracy-optimal value (0.77, ROC AUC 0.98; 11 of 1000 pairs skipped for no
detection). Same- and different-person distances separate cleanly, so one cutoff
works well:

![ROC curve](eval/roc_curve.png)
![Distance histogram](eval/distance_hist.png)

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

## Limitations

- The embeddings come from a pretrained ArcFace model; I did not train it.
- Struggles with extreme pose, poor lighting, occlusion and low-resolution faces.
- A small-gallery demo: recognition is a linear scan, not tuned for large galleries.
- A learning project, not for security, surveillance or access-control use.

## License & credits

Released under the MIT License (see [LICENSE](LICENSE)). © 2026 Ehina.

Sample faces are public-domain historical portraits from Wikimedia Commons. The
face detection/recognition models are pretrained weights from the InsightFace
project, used under their respective licenses.
