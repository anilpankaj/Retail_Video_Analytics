"""Feature cache: run inference once, re-tune every threshold in seconds.

Inference is by far the most expensive part of the pipeline — on CPU a full
7.5-minute clip takes about 50 minutes — yet nothing about it depends on any
of the thresholds a reviewer might want to question.  Detection, tracking,
keypoints and the two per-frame appearance measurements used by the staff
classifier are all threshold-independent.

So a run can write them to a JSONL cache (``--dump-features``), and a
``replay`` run then feeds that cache through exactly the same task state
machines with a different config, recomputing every count in a second or two
without touching the video.

This is what makes the threshold choices *auditable* rather than merely
documented: any claim in the README of the form "raising X to Y would change
the answer to Z" can be checked directly.

    python -m rva.cli entrance --dump-features outputs/cache/entrance.jsonl
    python -m rva.cli replay --features outputs/cache/entrance.jsonl \\
           --config configs/entrance.yaml \\
           --set task3.staff.score_threshold=0.70

The cache stores the *canonical* (post-fragment-linking) track id, so a replay
reproduces the original run exactly when the config is unchanged — a property
worth checking after any change to this module.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .tracking import Sample, TrackState, derive_signals
from .geometry import bbox_ground_point, bbox_height


class FeatureWriter:
    """Appends one JSON record per processed frame."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self.frames = 0

    def write(
        self,
        t: float,
        frame_idx: int,
        states: List[TrackState],
        appearance: Dict[int, Optional[Tuple[float, float]]],
    ) -> None:
        detections = []
        for state in states:
            kps = state.keypoints
            app = appearance.get(state.track_id)
            detections.append(
                {
                    "id": state.track_id,
                    "box": [round(float(v), 2) for v in state.box],
                    "kp": None if kps is None else [[round(float(x), 1), round(float(y), 1),
                                                     round(float(c), 3)] for x, y, c in kps],
                    "apron": None if app is None else round(float(app[0]), 4),
                    "badge": None if app is None else float(app[1]),
                }
            )
        self._handle.write(
            json.dumps({"t": round(t, 4), "f": frame_idx, "det": detections}) + "\n"
        )
        self.frames += 1

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


def read_features(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class ReplayTracks:
    """Rebuilds ``TrackState`` objects from a feature cache.

    The derived per-track signals (velocity, fused orientation, reach, wrist)
    are recomputed with :func:`rva.core.tracking.derive_signals`, i.e. the very
    same function the live pipeline uses, so replayed state is identical to
    live state apart from the colour signature, which only the fragment linker
    needs and which the cache has already applied.
    """

    def __init__(
        self,
        smoothing_window: int = 7,
        walk_speed_norm: float = 0.9,
        motion_bias: float = 0.65,
        kp_conf_min: float = 0.35,
        history_len: int = 120,
    ) -> None:
        self.smoothing_window = int(smoothing_window)
        self.walk_speed_norm = float(walk_speed_norm)
        self.motion_bias = float(motion_bias)
        self.kp_conf_min = float(kp_conf_min)
        self.history_len = int(history_len)
        self.tracks: Dict[int, TrackState] = {}

    def step(self, record: dict) -> Tuple[List[TrackState], Dict[int, Optional[Tuple[float, float]]]]:
        t = float(record["t"])
        frame_idx = int(record["f"])
        visible: List[TrackState] = []
        appearance: Dict[int, Optional[Tuple[float, float]]] = {}

        for det in record["det"]:
            track_id = int(det["id"])
            box = tuple(float(v) for v in det["box"])
            kps = None
            if det.get("kp") is not None:
                kps = np.asarray(det["kp"], dtype=np.float32)

            state = self.tracks.get(track_id)
            if state is None:
                state = TrackState(track_id=track_id, history=deque(maxlen=self.history_len))
                state.first_t = t
                self.tracks[track_id] = state

            state.history.append(
                Sample(
                    t=t,
                    frame_idx=frame_idx,
                    box=box,  # type: ignore[arg-type]
                    ground=bbox_ground_point(box),
                    height=max(bbox_height(box), 1.0),
                    keypoints=kps,
                    score=1.0,
                )
            )
            state.last_t = t
            state.n_obs += 1
            derive_signals(
                state,
                smoothing_window=self.smoothing_window,
                walk_speed_norm=self.walk_speed_norm,
                motion_bias=self.motion_bias,
                kp_conf_min=self.kp_conf_min,
            )
            visible.append(state)
            appearance[track_id] = (
                None if det.get("apron") is None
                else (float(det["apron"]), float(det.get("badge") or 0.0))
            )

        return visible, appearance
