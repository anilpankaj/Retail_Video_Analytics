"""Image-space geometry helpers.

Design note — why image space and not a homography
--------------------------------------------------
A ground-plane homography would give us metric distances, but it needs four
known-length references on the floor which the assessment does not provide,
and both cameras are wide-angle with visible barrel distortion, so a single
planar homography would be wrong at the edges anyway.

Instead every distance used for a decision is normalised by the *observed
pixel height of the person it concerns*.  A standing adult is ~1.7 m, so
``pixel_distance / person_pixel_height`` is a scale-invariant proxy for
"distance in body-heights", and it degrades gracefully under lens distortion
because both quantities are measured in the same local neighbourhood of the
image.  Every threshold expressed in these units is therefore readable as
"within N body-heights", which is also how a human would describe it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def _to_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


@dataclass
class Polygon:
    """A closed polygon in image coordinates."""

    points: np.ndarray
    name: str = ""

    def __init__(self, points: Sequence[Sequence[float]], name: str = "") -> None:
        self.points = _to_array(points)
        self.name = name
        if len(self.points) < 3:
            raise ValueError(f"polygon {name!r} needs at least 3 points")

    # -- containment / distance ------------------------------------------- #
    def contains(self, point: Point) -> bool:
        return cv2.pointPolygonTest(self.points, (float(point[0]), float(point[1])), False) >= 0

    def signed_distance(self, point: Point) -> float:
        """Positive inside, negative outside, in pixels."""
        return float(cv2.pointPolygonTest(self.points, (float(point[0]), float(point[1])), True))

    def distance(self, point: Point) -> float:
        """0.0 when inside, otherwise the pixel distance to the nearest edge."""
        return max(0.0, -self.signed_distance(point))

    # -- derived quantities ------------------------------------------------ #
    @property
    def centroid(self) -> Point:
        moments = cv2.moments(self.points)
        if abs(moments["m00"]) < 1e-6:
            mean = self.points.mean(axis=0)
            return float(mean[0]), float(mean[1])
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])

    @property
    def area(self) -> float:
        return float(abs(cv2.contourArea(self.points)))

    def nearest_point(self, point: Point) -> Point:
        """Closest point on the polygon boundary."""
        best: Point = (float(self.points[0][0]), float(self.points[0][1]))
        best_d = math.inf
        pts = self.points
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            cand = _closest_on_segment(point, (float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
            d = math.dist(point, cand)
            if d < best_d:
                best_d, best = d, cand
        return best

    def as_int(self) -> np.ndarray:
        return self.points.astype(np.int32)

    def mask(self, shape: Tuple[int, int]) -> np.ndarray:
        canvas = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(canvas, [self.as_int()], 255)
        return canvas


@dataclass
class Polyline:
    """An open polyline — used for the storefront / entrance boundary."""

    points: np.ndarray
    name: str = ""

    def __init__(self, points: Sequence[Sequence[float]], name: str = "") -> None:
        self.points = _to_array(points)
        self.name = name
        if len(self.points) < 2:
            raise ValueError(f"polyline {name!r} needs at least 2 points")

    def distance(self, point: Point) -> float:
        return math.dist(point, self.nearest_point(point))

    def nearest_point(self, point: Point) -> Point:
        best: Point = (float(self.points[0][0]), float(self.points[0][1]))
        best_d = math.inf
        for i in range(len(self.points) - 1):
            a = self.points[i]
            b = self.points[i + 1]
            cand = _closest_on_segment(point, (float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
            d = math.dist(point, cand)
            if d < best_d:
                best_d, best = d, cand
        return best

    def as_int(self) -> np.ndarray:
        return self.points.astype(np.int32)


def _closest_on_segment(p: Point, a: Point, b: Point) -> Point:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < 1e-9:
        return a
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    return ax + t * dx, ay + t * dy


# --------------------------------------------------------------------------- #
# vector helpers
# --------------------------------------------------------------------------- #
def normalize(vec: Sequence[float]) -> Tuple[float, float]:
    x, y = float(vec[0]), float(vec[1])
    n = math.hypot(x, y)
    if n < 1e-9:
        return 0.0, 0.0
    return x / n, y / n


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Angle in degrees between two 2-D vectors (0-180); 180 if either is null."""
    ax, ay = normalize(a)
    bx, by = normalize(b)
    if (ax == 0 and ay == 0) or (bx == 0 and by == 0):
        return 180.0
    dot = max(-1.0, min(1.0, ax * bx + ay * by))
    return math.degrees(math.acos(dot))


def bbox_ground_point(box: Sequence[float]) -> Point:
    """Feet position proxy: bottom-centre of the bounding box."""
    x1, y1, x2, y2 = box
    return (float(x1) + float(x2)) / 2.0, float(y2)


def bbox_height(box: Sequence[float]) -> float:
    return float(box[3]) - float(box[1])


def bbox_center(box: Sequence[float]) -> Point:
    x1, y1, x2, y2 = box
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def build_polygons(raw: Iterable[Sequence[Sequence[float]]], names: Iterable[str]) -> List[Polygon]:
    return [Polygon(pts, name) for pts, name in zip(raw, names)]
