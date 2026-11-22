# Group Pulse API on a Hugging Face Space (free CPU Docker Space: 2 vCPU, 16 GB RAM).
# Build context = repository root:  docker build -f deploy/huggingface/Dockerfile -t grouppulse-api .
FROM python:3.12-slim

# OpenCV / MediaPipe runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 app
WORKDIR /app

COPY --chown=app pyproject.toml README.md ./
COPY --chown=app pipeline ./pipeline
COPY --chown=app configs ./configs
COPY --chown=app demo ./demo

# CPU-only PyTorch keeps the image small; then the app with API, perception and pose extras
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[api,perception,pose]" sentry-sdk

# Models are fetched at build time so a cold start does not download anything
RUN mkdir -p models data && \
    curl -fsSL -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx && \
    curl -fsSL -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task && \
    curl -fsSL -o models/yolo11n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt && \
    chown -R app /app

USER app
ENV GP_DATABASE_URL=sqlite:////app/data/grouppulse.db \
    GP_FILES_DIR=/app/data/uploads \
    GP_MAX_UPLOAD_MB=200 \
    GP_RETENTION_HOURS=24 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    PORT=7860
EXPOSE 7860

# The Space's disk is not persistent: re-import the AMI demo sessions on every start.
CMD python -m grouppulse.api.import_ami demo/sessions && python -m grouppulse.api --host 0.0.0.0 --port ${PORT}
