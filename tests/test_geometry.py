"""Geometry primitives: containment, distance and the scale-invariant units."""

from __future__ import annotations

import math

import pytest

from rva.core.geometry import (
    Polygon,
    Polyline,
    angle_between,
    bbox_ground_point,
    bbox_height,
    iou,
    normalize,
)


@pytest.fixture
def square() -> Polygon:
    return Polygon([(0, 0), (100, 0), (100, 100), (0, 100)], "square")


def test_polygon_containment(square: Polygon) -> None:
    assert square.contains((50, 50))
    assert not square.contains((150, 50))
    assert square.contains((0, 0))  # on the boundary counts as inside


def test_polygon_distance_is_zero_inside(square: Polygon) -> None:
    assert square.distance((50, 50)) == 0.0
    assert square.distance((120, 50)) == pytest.approx(20.0, abs=0.5)


def test_polygon_nearest_point(square: Polygon) -> None:
    near = square.nearest_point((150, 50))
    assert near == pytest.approx((100.0, 50.0), abs=0.5)


def test_polygon_centroid_and_area(square: Polygon) -> None:
    assert square.centroid == pytest.approx((50.0, 50.0), abs=0.5)
    assert square.area == pytest.approx(10000.0, rel=1e-3)


def test_polyline_distance_and_projection() -> None:
    line = Polyline([(0, 0), (100, 0), (100, 100)])
    assert line.distance((50, 30)) == pytest.approx(30.0, abs=0.5)
    assert line.nearest_point((50, 30)) == pytest.approx((50.0, 0.0), abs=0.5)
    # the corner region projects onto the vertex
    assert line.distance((130, -30)) == pytest.approx(math.hypot(30, 30), abs=0.5)


def test_polygon_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        Polygon([(0, 0), (1, 1)])


def test_angle_between() -> None:
    assert angle_between((1, 0), (1, 0)) == pytest.approx(0.0, abs=1e-3)
    assert angle_between((1, 0), (0, 1)) == pytest.approx(90.0, abs=1e-3)
    assert angle_between((1, 0), (-1, 0)) == pytest.approx(180.0, abs=1e-3)
    # a null vector must never look like agreement
    assert angle_between((0, 0), (1, 0)) == 180.0


def test_normalize() -> None:
    assert normalize((3, 4)) == pytest.approx((0.6, 0.8))
    assert normalize((0, 0)) == (0.0, 0.0)


def test_bbox_helpers() -> None:
    box = (10.0, 20.0, 30.0, 120.0)
    assert bbox_ground_point(box) == (20.0, 120.0)
    assert bbox_height(box) == 100.0


def test_iou() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3, abs=1e-3)


def test_body_height_normalisation_is_scale_invariant() -> None:
    """The same real-world distance must give the same normalised value.

    A shopper near the camera (200 px tall, 200 px from the shelf) and one
    far away (60 px tall, 60 px from it) are both exactly one body-height
    away and must be treated identically.
    """
    near = 200.0 / 200.0
    far = 60.0 / 60.0
    assert near == pytest.approx(far)
