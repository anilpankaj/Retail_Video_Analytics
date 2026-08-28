#!/usr/bin/env bash
# Re-encode the annotated videos to H.264 for distribution.
#
# OpenCV's VideoWriter is used in the pipeline with the `mp4v` FourCC because it
# is the one codec available in every OpenCV build on every platform - but it is
# an MPEG-4 Part 2 encoder with no rate control worth the name, and it produces
# roughly 8x the file size of H.264 at the same visual quality (415 MB vs 54 MB
# for the 7.5-minute entrance clip).
#
# The pipeline is deliberately left portable and this re-encode is a separate,
# optional step, so a reviewer without ffmpeg can still run everything.
#
#   bash tools/compress_outputs.sh [outputs_dir] [crf]
#
# CRF 24 is visually lossless for annotation overlays at this resolution; lower
# it to 20 if you want to be certain nothing in the text is softened.
set -euo pipefail

OUT_DIR="${1:-outputs}"
CRF="${2:-24}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found - skipping compression (the mp4v files are already valid)" >&2
    exit 0
fi

shopt -s nullglob
for src in "${OUT_DIR}"/*_annotated.mp4; do
    tmp="${src%.mp4}.h264.mp4"
    before=$(du -h "${src}" | cut -f1)
    echo "re-encoding $(basename "${src}") (${before}) ..."
    ffmpeg -hide_banner -loglevel error -y \
        -i "${src}" \
        -c:v libx264 -crf "${CRF}" -preset veryfast \
        -pix_fmt yuv420p -movflags +faststart \
        "${tmp}"
    mv -f "${tmp}" "${src}"
    echo "  -> $(du -h "${src}" | cut -f1)"
done

echo "done"
