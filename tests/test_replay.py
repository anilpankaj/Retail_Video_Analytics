"""The feature cache: a replay must reproduce the live run exactly.

The cache exists so a reviewer can re-tune any threshold in seconds instead of
re-running an hour of CPU inference.  That is only worth anything if replaying
an *unchanged* config reproduces the original numbers bit for bit, which is
what these tests pin down.
"""

from __future__ import annotations

from collections import deque

import pytest

from conftest import make_keypoints
from rva.config import Config
from rva.core.replay import FeatureWriter, ReplayTracks, read_features
from rva.core.tracking import Sample, TrackState, derive_signals, velocity_of
from rva.pipeline import replay_entrance


def synthetic_state(track_id: int, x: float, y: float, t: float, height: float = 120.0) -> TrackState:
    box = (x - height * 0.2, y - height, x + height * 0.2, y)
    state = TrackState(track_id=track_id, history=deque(maxlen=90))
    state.history.append(
        Sample(t=t, frame_idx=int(t * 10), box=box, ground=(x, y), height=height,
               keypoints=make_keypoints((x + 12, y - height * 0.75), (x - 12, y - height * 0.75)),
               score=0.9)
    )
    state.first_t = t
    state.last_t = t
    state.n_obs = 1
    return state


def test_writer_round_trips_boxes_and_keypoints(tmp_path) -> None:
    path = tmp_path / "cache.jsonl"
    writer = FeatureWriter(path)
    state = synthetic_state(1, 700.0, 340.0, 0.0)
    writer.write(0.0, 0, [state], {1: (0.55, 1.0)})
    writer.close()

    records = list(read_features(path))
    assert len(records) == 1
    det = records[0]["det"][0]
    assert det["id"] == 1
    assert det["apron"] == pytest.approx(0.55)
    assert det["badge"] == 1.0
    assert len(det["kp"]) == 17


def test_replay_rebuilds_identical_track_geometry(tmp_path) -> None:
    path = tmp_path / "cache.jsonl"
    writer = FeatureWriter(path)
    live = None
    for step in range(12):
        t = step * 0.1
        x = 700.0 + step * 6.0
        state = synthetic_state(1, x, 340.0, t)
        if live is None:
            live = state
        else:
            live.history.append(state.history[-1])
            live.last_t = t
            live.n_obs += 1
        derive_signals(live)
        writer.write(t, step, [live], {1: (0.4, 0.0)})
    writer.close()

    tracks = ReplayTracks()
    replayed = None
    for record in read_features(path):
        visible, appearance = tracks.step(record)
        replayed = visible[0]
        assert appearance[1] == (0.4, 0.0)

    assert replayed is not None and live is not None
    assert replayed.ground == pytest.approx(live.ground)
    assert replayed.height == pytest.approx(live.height)
    assert replayed.speed_norm == pytest.approx(live.speed_norm, rel=1e-6)
    assert replayed.facing.vector == pytest.approx(live.facing.vector, abs=1e-6)


def test_replay_produces_the_same_counts_as_a_direct_run(tmp_path, entrance_cfg) -> None:
    """A person who stands at the storefront looking in, then walks in."""
    path = tmp_path / "cache.jsonl"
    writer = FeatureWriter(path)

    live = None
    step = 0
    for point, count in [((700.0, 340.0), 30), ((700.0, 620.0), 30)]:
        for _ in range(count):
            t = step * 0.1
            x, y = point
            state = synthetic_state(1, x, y, t, height=130.0)
            if live is None:
                live = state
            else:
                live.history.append(state.history[-1])
                live.last_t = t
                live.n_obs += 1
            derive_signals(live)
            writer.write(t, step, [live], {1: (0.1, 0.0)})
            step += 1
    writer.close()

    summary = replay_entrance(entrance_cfg, str(path))
    assert summary["task1"]["total_interested"] == 1
    assert summary["task1"]["interested_entered"] == 1
    assert (
        summary["task1"]["interested_entered"] + summary["task1"]["interested_passed_by"]
        == summary["task1"]["total_interested"]
    )


def test_replay_reacts_to_a_config_override(tmp_path, entrance_cfg) -> None:
    """Raising the interest threshold must be able to change the answer -
    otherwise the cache would be useless for sensitivity analysis."""
    path = tmp_path / "cache.jsonl"
    writer = FeatureWriter(path)
    live = None
    for step in range(40):
        t = step * 0.1
        state = synthetic_state(1, 700.0, 340.0, t, height=130.0)
        if live is None:
            live = state
        else:
            live.history.append(state.history[-1])
            live.last_t = t
            live.n_obs += 1
        derive_signals(live)
        writer.write(t, step, [live], {1: (0.1, 0.0)})
    writer.close()

    base = replay_entrance(entrance_cfg, str(path))
    strict = dict(entrance_cfg)
    strict["task1"] = {
        **strict["task1"],
        "interest": {**strict["task1"]["interest"], "score_threshold": 99.0},
    }
    tightened = replay_entrance(Config(strict), str(path))

    assert base["task1"]["total_interested"] == 1
    assert tightened["task1"]["total_interested"] == 0


def test_velocity_is_zero_for_a_stationary_track() -> None:
    state = synthetic_state(1, 100.0, 200.0, 0.0)
    for step in range(1, 8):
        state.history.append(
            Sample(t=step * 0.1, frame_idx=step, box=state.box, ground=(100.0, 200.0),
                   height=120.0, keypoints=None, score=0.9)
        )
    _, speed = velocity_of(state)
    assert speed == pytest.approx(0.0)
