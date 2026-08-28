"""Config loading, inheritance and the sanity of the shipped scene files."""

from __future__ import annotations

import pytest

from rva.config import Config, as_points, deep_merge, parse_cli_overrides
from rva.core.geometry import Polygon, Polyline


def test_deep_merge_is_recursive_and_non_destructive() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20, "z": 30}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}
    assert base["a"]["y"] == 2, "the base must not be mutated"


def test_dotted_access() -> None:
    cfg = Config({"task1": {"entry": {"confirm_s": 1.2}}})
    assert cfg.get_path("task1.entry.confirm_s") == 1.2
    assert cfg.get_path("task1.missing.key", "fallback") == "fallback"
    with pytest.raises(KeyError):
        cfg.require("nope.at.all")


def test_cli_overrides_parse_into_nested_values() -> None:
    parsed = parse_cli_overrides(["task1.entry.confirm_s=2.5", "model.imgsz=640"])
    assert parsed == {"task1": {"entry": {"confirm_s": 2.5}}, "model": {"imgsz": 640}}


def test_entrance_config_inherits_common(entrance_cfg) -> None:
    assert entrance_cfg.get_path("model.weights")          # from common.yaml
    assert entrance_cfg.get_path("scene.storefront_line")  # from entrance.yaml


def test_shipped_polygons_are_valid(entrance_cfg, interior_cfg) -> None:
    scene = entrance_cfg.require("scene")
    store = Polygon(as_points(scene["store_polygon"]))
    walkway = Polygon(as_points(scene["walkway_polygon"]))
    front = Polyline(as_points(scene["storefront_line"]))
    assert store.area > 10000
    assert walkway.area > 10000
    assert len(front.points) >= 2

    for shelf in interior_cfg.require("scene")["shelves"]:
        polygon = Polygon(as_points(shelf["polygon"]), shelf["id"])
        assert polygon.area > 1000, f"shelf {shelf['id']} polygon looks degenerate"


def test_all_four_shelves_are_configured(interior_cfg) -> None:
    ids = sorted(s["id"] for s in interior_cfg.require("scene")["shelves"])
    assert ids == ["A", "B", "C", "D"]


def test_store_and_walkway_share_the_storefront_line(entrance_cfg) -> None:
    """The lease line must actually separate the two zones, not float free."""
    scene = entrance_cfg.require("scene")
    store = Polygon(as_points(scene["store_polygon"]))
    walkway = Polygon(as_points(scene["walkway_polygon"]))
    for point in as_points(scene["storefront_line"]):
        assert store.distance(point) < 2.0, f"{point} is not on the store boundary"
        assert walkway.distance(point) < 2.0, f"{point} is not on the walkway boundary"


def test_thresholds_are_in_seconds_not_frames(entrance_cfg, interior_cfg) -> None:
    """Guards the invariant that frame_stride can never change a count."""
    assert entrance_cfg.get_path("task1.entry.confirm_s") > 0
    assert entrance_cfg.get_path("task3.interaction.min_session_s") > 0
    assert interior_cfg.get_path("task2.interest.min_dwell_s") > 0
    assert interior_cfg.get_path("task2.interest.min_gap_between_events_s") > \
        interior_cfg.get_path("task2.interest.flicker_tolerance_s"), \
        "a return must be harder to trigger than bridging a flicker"
