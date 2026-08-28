# ---------------------------------------------------------------------------
# Reproducible CPU execution environment for the retail video analytics
# pipeline.  CPU-only by design: the assessment forbids external inference
# services, and a CPU image runs anywhere a reviewer might try it.
#
#   docker build -t rva .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" rva
#
# For a CUDA machine, swap the base image for
# `pytorch/pytorch:2.13.0-cuda12.1-cudnn9-runtime`, drop the CPU torch index
# below, and run with `--gpus all --set model.device=cuda`.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    # keep Ultralytics from phoning home or writing outside the container
    YOLO_CONFIG_DIR=/app/.ultralytics \
    MPLCONFIGDIR=/tmp/mpl

# libgl / libglib are needed by OpenCV; ffmpeg for robust video decode+encode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU build of torch first so the CUDA wheels are never pulled.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 torchvision==0.28.0 \
 && pip install --no-cache-dir -r requirements.txt

# Model weights: fetched at build time so the container runs fully offline.
COPY models/download_models.sh models/download_models.sh
RUN bash models/download_models.sh

COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/
COPY pytest.ini Makefile ./

# Fail the build if the counting logic is broken.
RUN python -m pytest -q

# `data/` (inputs) and `outputs/` (results) are mounted at run time.
VOLUME ["/app/data", "/app/outputs"]

ENTRYPOINT ["python", "-m", "rva.cli"]
CMD ["all"]
