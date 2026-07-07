# Dockerfile for deploying FaceMatch to Hugging Face Spaces (Docker SDK).
# HF Spaces run the container as a non-root user (uid 1000) and expect the app
# to listen on port 7860.
# Python 3.13 to match the pinned dependency versions (numpy 2.5.x needs >=3.12).
FROM python:3.13-slim

# System libraries needed by opencv-python-headless / onnxruntime / scipy.
# (OpenCV 5's "headless" build still links a full set of X/GL libs at import.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 \
        libgl1 libxcb1 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (recommended by Hugging Face Spaces).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FACEMATCH_SEED_DEMO=1

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# InsightFace caches its model under $HOME/.insightface on first request.
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
