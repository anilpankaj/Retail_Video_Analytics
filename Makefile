PY ?= python3
export PYTHONPATH := src

.PHONY: help setup models test run entrance interior smoke zones grid compress clean docker docker-run

help:
	@echo "make setup      install dependencies and download model weights"
	@echo "make test       run the unit tests (no video or model needed)"
	@echo "make run        produce every deliverable from both videos"
	@echo "make entrance   Task 1 + Task 3 only"
	@echo "make interior   Task 2 only"
	@echo "make smoke      60-second end-to-end sanity run"
	@echo "make zones      render the configured zones onto reference frames"
	@echo "make compress   re-encode the annotated videos to H.264 (needs ffmpeg)"
	@echo "make docker     build the reproducible container"

setup: models
	$(PY) -m pip install -r requirements.txt

models:
	bash models/download_models.sh

test:
	$(PY) -m pytest

run:
	$(PY) -m rva.cli all

entrance:
	$(PY) -m rva.cli entrance

interior:
	$(PY) -m rva.cli interior

smoke:
	$(PY) -m rva.cli entrance --max-seconds 60 --outputs outputs/smoke
	$(PY) -m rva.cli interior --max-seconds 60 --outputs outputs/smoke

zones:
	$(PY) -m rva.cli zones --config configs/entrance.yaml --video data/entrance.mp4 \
		--at 20 --out outputs/zones_entrance.jpg
	$(PY) -m rva.cli zones --config configs/interior.yaml --video data/interior.mp4 \
		--at 25 --out outputs/zones_interior.jpg

grid:
	$(PY) -m rva.cli grid --video data/entrance.mp4 --at 20 --out outputs/grid_entrance.jpg
	$(PY) -m rva.cli grid --video data/interior.mp4 --at 25 --out outputs/grid_interior.jpg

docker:
	docker build -t rva .

docker-run:
	docker run --rm -v "$$PWD/data:/app/data" -v "$$PWD/outputs:/app/outputs" rva all

compress:
	bash tools/compress_outputs.sh outputs

clean:
	rm -rf outputs/*.mp4 outputs/*.csv outputs/audit
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
