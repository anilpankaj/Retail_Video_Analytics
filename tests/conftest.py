"""Shared fixtures: synthetic tracks so task logic can be tested without a model."""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np
import pytest

from rva.config import Config, load_config
from rva.core.orientation import Orientation
from rva.core.tracking import Sample, TrackState

ROOT_CONFIGS = "configs"


def make_track(
    track_id: int,
    ground: Tuple[float, float],
    height: float = 120.0,
    facing: Optional[Tuple[float, float]] = None,
    facing_conf: float = 0.9,
    speed_norm: float = 0.0,
    t: float = 0.0,
    duration: float = 5.0,
    role: str = "customer",
    reach: Optional[Tuple[float, float]] = None,
) -> TrackState:
    """Build a TrackState with exactly the fields the task logic reads."""
    x, y = ground
    box = (x - height * 0.2, y - height, x + height * 0.2, y)
    state = TrackState(track_id=track_id, history=deque(maxlen=90))
    state.history.append(
        Sample(t=t, frame_idx=int(t * 30), box=box, ground=(x, y), height=height,
               keypoints=None, score=0.9)
    )
    state.first_t = t - duration
    state.last_t = t
    state.n_obs = 10
    state.role = role
    state.speed_norm = speed_norm
    state.reach = reach
    if facing is not None:
        state.facing = Orientation(vector=facing, confidence=facing_conf, source="fused")
    return state


def move(state: TrackState, ground: Tuple[float, float], t: float,
         facing: Optional[Tuple[float, float]] = None, speed_norm: Optional[float] = None) -> TrackState:
    """Advance an existing synthetic track to a new position/time."""
    x, y = ground
    h = state.height
    box = (x - h * 0.2, y - h, x + h * 0.2, y)
    state.history.append(
        Sample(t=t, frame_idx=int(t * 30), box=box, ground=(x, y), height=h,
               keypoints=None, score=0.9)
    )
    state.last_t = t
    state.n_obs += 1
    if facing is not None:
        state.facing = Orientation(vector=facing, confidence=0.9, source="fused")
    if speed_norm is not None:
        state.speed_norm = speed_norm
    return state


@pytest.fixture
def entrance_cfg() -> Config:
    return load_config(f"{ROOT_CONFIGS}/entrance.yaml")


@pytest.fixture
def interior_cfg() -> Config:
    return load_config(f"{ROOT_CONFIGS}/interior.yaml")


def make_keypoints(
    shoulder_left: Tuple[float, float],
    shoulder_right: Tuple[float, float],
    nose: Optional[Tuple[float, float]] = None,
    wrist: Optional[Tuple[float, float]] = None,
    conf: float = 0.9,
) -> np.ndarray:
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[5] = (*shoulder_left, conf)
    kps[6] = (*shoulder_right, conf)
    kps[11] = (shoulder_left[0], shoulder_left[1] + 40, conf)
    kps[12] = (shoulder_right[0], shoulder_right[1] + 40, conf)
    if nose is not None:
        kps[0] = (*nose, conf)
    if wrist is not None:
        kps[9] = (*wrist, conf)
    return kps
