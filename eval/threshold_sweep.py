"""Pick the match threshold from data instead of guessing it.

Runs the face pipeline over the LFW "test" pairs, measures the cosine distance
for same-person and different-person pairs, and sweeps the decision threshold to
report TPR / FPR / precision / accuracy. Saves an ROC curve and a distance
histogram, and prints the accuracy-optimal threshold.

    python eval/threshold_sweep.py

Detected embeddings are cached to eval/lfw_embeddings.npz, so a second run is
fast and doesn't touch the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import face_engine  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CACHE = EVAL_DIR / "lfw_embeddings.npz"


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """LFW images come back as float; normalise to an RGB uint8 array."""
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _largest_embedding(img: np.ndarray) -> np.ndarray | None:
    faces = face_engine.detect_faces(_to_uint8(img))
    return faces[0].embedding if faces else None


def build_distances() -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (distances, labels, n_total, n_skipped), using the cache if present."""
    if CACHE.exists():
        data = np.load(CACHE)
        return data["distances"], data["labels"], int(data["n_total"]), int(data["n_skipped"])

    from sklearn.datasets import fetch_lfw_pairs

    print("loading LFW test pairs (first run downloads ~200 MB) ...")
    # slice_=None keeps the full 250x250 funneled image; the default tight crop
    # leaves the detector too little context and it finds no face.
    lfw = fetch_lfw_pairs(subset="test", color=True, funneled=True, resize=1.0, slice_=None)
    pairs, targets = lfw.pairs, lfw.target
    n_total = len(pairs)

    distances, labels, skipped = [], [], 0
    for i, ((img_a, img_b), label) in enumerate(zip(pairs, targets)):
        if i % 100 == 0:
            print(f"  pair {i}/{n_total} (kept {len(distances)}, skipped {skipped})")
        emb_a = _largest_embedding(img_a)
        emb_b = _largest_embedding(img_b)
        if emb_a is None or emb_b is None:
            skipped += 1
            continue
        distances.append(face_engine.distance(emb_a, emb_b))
        labels.append(int(label))

    distances = np.array(distances, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    np.savez(CACHE, distances=distances, labels=labels, n_total=n_total, n_skipped=skipped)
    return distances, labels, n_total, skipped


def sweep(distances: np.ndarray, labels: np.ndarray):
    same = labels == 1
    diff = labels == 0
    n_pos, n_neg = int(same.sum()), int(diff.sum())

    thresholds = np.round(np.arange(0.0, 1.5 + 1e-9, 0.01), 2)
    tpr = np.zeros_like(thresholds)
    fpr = np.zeros_like(thresholds)
    precision = np.zeros_like(thresholds)
    accuracy = np.zeros_like(thresholds)

    for k, t in enumerate(thresholds):
        pred = distances <= t
        tp = int((pred & same).sum())
        fp = int((pred & diff).sum())
        tn = n_neg - fp
        tpr[k] = tp / n_pos if n_pos else 0.0
        fpr[k] = fp / n_neg if n_neg else 0.0
        precision[k] = tp / (tp + fp) if (tp + fp) else 0.0
        accuracy[k] = (tp + tn) / (n_pos + n_neg)

    return thresholds, tpr, fpr, precision, accuracy, n_pos, n_neg


def main() -> None:
    distances, labels, n_total, n_skipped = build_distances()
    thresholds, tpr, fpr, precision, accuracy, n_pos, n_neg = sweep(distances, labels)

    print("\nthresh   TPR     FPR    prec    acc")
    for t, a, b, p, c in zip(thresholds, tpr, fpr, precision, accuracy):
        print(f"{t:5.2f}  {a:6.3f}  {b:6.3f}  {p:6.3f}  {c:6.3f}")

    best = int(np.argmax(accuracy))
    best_t = float(thresholds[best])

    # Strictest useful operating point: lowest FPR we can hit (<= 0.001 target),
    # then the most lenient threshold that still holds it.
    strict_mask = fpr <= 0.001
    if strict_mask.any():
        idxs = np.where(strict_mask)[0]
        strict = int(idxs[np.argmax(tpr[idxs])])
    else:
        strict = int(np.argmin(fpr))

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(labels, -distances)

    print("\n----- results -----")
    print(f"pairs total in LFW test: {n_total}")
    print(f"pairs used:              {len(distances)}  (same={n_pos}, diff={n_neg})")
    print(f"pairs skipped (no face detected on one/both images): {n_skipped}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"best accuracy: threshold={best_t:.2f}  acc={accuracy[best]:.3f}  "
          f"TPR={tpr[best]:.3f}  FPR={fpr[best]:.3f}")
    print(f"FPR<=0.001 point: threshold={thresholds[strict]:.2f}  "
          f"TPR={tpr[strict]:.3f}  FPR={fpr[strict]:.3f}")

    _plot(thresholds, tpr, fpr, distances, labels, best_t, auc)
    print(f"\nsaved {EVAL_DIR / 'roc_curve.png'} and {EVAL_DIR / 'distance_hist.png'}")
    print(f"\n>>> set config.DEFAULT_THRESHOLD to {best_t:.2f}")


def _plot(thresholds, tpr, fpr, distances, labels, best_t, auc) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#2f56e6", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#cbd5e1", lw=1, ls="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("LFW verification ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(distances[labels == 1], bins=40, alpha=0.6, color="#22c55e", label="same person")
    ax.hist(distances[labels == 0], bins=40, alpha=0.6, color="#ef4444", label="different person")
    ax.axvline(best_t, color="#1b2534", lw=2, ls="--", label=f"threshold={best_t:.2f}")
    ax.set_xlabel("cosine distance")
    ax.set_ylabel("pair count")
    ax.set_title("Same vs different person distances (LFW test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "distance_hist.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
