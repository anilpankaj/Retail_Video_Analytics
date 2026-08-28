#!/usr/bin/env bash
# Fetch the pretrained YOLO11 pose weights.
#
# The assessment allows publicly available pretrained weights to be downloaded
# during environment setup, so they are not committed to the repository. These
# are the stock Ultralytics COCO-pretrained pose checkpoints - no fine-tuning
# was performed, and all inference runs locally.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0"

# yolo11s-pose is the default in configs/common.yaml.
# yolo11n-pose is the faster fallback for slow CPUs (--set model.weights=...).
MODELS=("yolo11s-pose.pt" "yolo11n-pose.pt")

for model in "${MODELS[@]}"; do
    target="${DIR}/${model}"
    if [[ -f "${target}" ]]; then
        echo "already present: ${model}"
        continue
    fi
    echo "downloading ${model} ..."
    curl -fsSL --retry 3 -o "${target}" "${BASE_URL}/${model}"
done

echo "model weights ready in ${DIR}"
ls -lh "${DIR}"/*.pt
