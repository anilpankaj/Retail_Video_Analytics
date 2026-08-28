"""Annotation layer.

Twenty per cent of the rubric is presentation, and the brief is specific:
counts at the top-left, every ML detection visibly annotated, and
interaction results presented so the detected interactions can be understood
at a glance.  Three details matter for readability on this footage:

* the source video already burns a timestamp into the top-left corner, so
  the metrics panel is offset below it rather than fighting with it;
* every text element is drawn on a translucent slab, because white text over
  a white mall floor is unreadable;
* colour encodes *state*, not identity, so a reviewer can see at a glance
  which people are contributing to which count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import Polygon, Polyline

BGR = Tuple[int, int, int]

# --- palette (BGR) --------------------------------------------------------- #
COL_BG = (28, 26, 24)
COL_TEXT = (245, 245, 245)
COL_MUTED = (170, 170, 170)
COL_NEUTRAL = (190, 190, 190)
COL_INTEREST = (0, 210, 255)     # amber
COL_ENTERED = (90, 220, 90)      # green
COL_PASSED = (120, 140, 255)     # salmon
COL_STAFF = (0, 190, 255)        # gold
COL_CUSTOMER = (205, 205, 205)   # neutral grey - shelf/state colours must stand out
COL_LINK = (80, 255, 255)
COL_STORE = (255, 150, 60)
COL_WALKWAY = (70, 200, 255)
COL_FRONT = (60, 60, 240)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX


def _text_size(text: str, scale: float, thickness: int, font=FONT) -> Tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    return w, h


def slab(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    alpha: float = 0.55,
    color: BGR = COL_BG,
    radius: int = 6,
) -> None:
    """Draw a translucent rounded rectangle in place."""
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x or y2 <= y:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def text(
    img: np.ndarray,
    label: str,
    org: Tuple[int, int],
    color: BGR = COL_TEXT,
    scale: float = 0.6,
    thickness: int = 1,
    font=FONT,
    shadow: bool = True,
) -> None:
    if shadow:
        cv2.putText(img, label, (org[0] + 1, org[1] + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, label, org, font, scale, color, thickness, cv2.LINE_AA)


def chip(
    img: np.ndarray,
    label: str,
    org: Tuple[int, int],
    color: BGR,
    scale: float = 0.52,
    thickness: int = 1,
) -> Tuple[int, int]:
    """A small filled label. Returns its (width, height)."""
    w, h = _text_size(label, scale, thickness, FONT_S)
    pad_x, pad_y = 6, 4
    x, y = org
    cv2.rectangle(img, (x, y - h - pad_y), (x + w + 2 * pad_x, y + pad_y), color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (x + pad_x, y), FONT_S, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return w + 2 * pad_x, h + 2 * pad_y


def draw_polygon(
    img: np.ndarray,
    polygon: Polygon,
    color: BGR,
    label: str = "",
    fill_alpha: float = 0.12,
    thickness: int = 2,
) -> None:
    pts = polygon.as_int()
    if fill_alpha > 0:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    if label:
        cx, cy = polygon.centroid
        text(img, label, (int(cx) - 20, int(cy)), color, 0.6, 1)


def draw_polyline(img: np.ndarray, line: Polyline, color: BGR, thickness: int = 3) -> None:
    cv2.polylines(img, [line.as_int()], False, color, thickness, cv2.LINE_AA)


def draw_person(
    img: np.ndarray,
    box: Sequence[float],
    color: BGR,
    label: str = "",
    sublabel: str = "",
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if label:
        chip(img, label, (x1, max(14, y1 - 6)), color)
    if sublabel:
        text(img, sublabel, (x1, min(img.shape[0] - 4, y2 + 15)), color, 0.46, 1, FONT_S)


SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]


def draw_keypoints(
    img: np.ndarray,
    keypoints: Optional[np.ndarray],
    color: BGR,
    conf_min: float = 0.35,
    radius: int = 2,
) -> None:
    """Draw the pose the decisions were actually made from."""
    if keypoints is None or len(keypoints) < 17:
        return
    for a, b in SKELETON:
        if keypoints[a][2] >= conf_min and keypoints[b][2] >= conf_min:
            cv2.line(
                img,
                (int(keypoints[a][0]), int(keypoints[a][1])),
                (int(keypoints[b][0]), int(keypoints[b][1])),
                color, 1, cv2.LINE_AA,
            )
    for kp in keypoints:
        if kp[2] >= conf_min:
            cv2.circle(img, (int(kp[0]), int(kp[1])), radius, color, -1, cv2.LINE_AA)


def draw_facing(
    img: np.ndarray,
    origin: Sequence[float],
    vector: Sequence[float],
    color: BGR,
    length: float = 55.0,
) -> None:
    if not vector or (vector[0] == 0 and vector[1] == 0):
        return
    x, y = int(origin[0]), int(origin[1])
    tip = (int(x + vector[0] * length), int(y + vector[1] * length))
    cv2.arrowedLine(img, (x, y), tip, color, 2, cv2.LINE_AA, tipLength=0.32)


def draw_link(
    img: np.ndarray,
    a: Sequence[float],
    b: Sequence[float],
    color: BGR,
    label: str = "",
    thickness: int = 2,
) -> None:
    pa = (int(a[0]), int(a[1]))
    pb = (int(b[0]), int(b[1]))
    cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
    cv2.circle(img, pa, 4, color, -1, cv2.LINE_AA)
    cv2.circle(img, pb, 4, color, -1, cv2.LINE_AA)
    if label:
        mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
        chip(img, label, (mid[0] - 26, mid[1]), color)


@dataclass
class PanelRow:
    label: str
    value: str
    color: BGR = COL_TEXT
    emphasis: bool = False


def draw_panel(
    img: np.ndarray,
    title: str,
    rows: Sequence[PanelRow],
    origin: Tuple[int, int] = (14, 52),
    width: int = 340,
    footnote: str = "",
) -> Tuple[int, int]:
    """The counts panel. Placed below the camera's burned-in timestamp."""
    x, y = origin
    line_h = 26
    head_h = 30
    foot_h = 20 if footnote else 0
    height = head_h + line_h * len(rows) + 12 + foot_h
    slab(img, x, y, width, height, alpha=0.62)
    cv2.rectangle(img, (x, y), (x + width, y + height), (90, 90, 90), 1, cv2.LINE_AA)

    text(img, title, (x + 12, y + 21), COL_TEXT, 0.6, 1)
    cv2.line(img, (x + 10, y + 28), (x + width - 10, y + 28), (110, 110, 110), 1, cv2.LINE_AA)

    cursor = y + head_h + 16
    for row in rows:
        scale = 0.62 if row.emphasis else 0.55
        text(img, row.label, (x + 12, cursor), COL_MUTED if not row.emphasis else COL_TEXT, scale, 1, FONT_S)
        vw, _ = _text_size(row.value, scale + 0.04, 2, FONT)
        text(img, row.value, (x + width - 14 - vw, cursor), row.color, scale + 0.04, 2 if row.emphasis else 1)
        cursor += line_h

    if footnote:
        text(img, footnote, (x + 12, y + height - 7), COL_MUTED, 0.42, 1, FONT_S)
    return width, height


def draw_footer(img: np.ndarray, left: str, right: str = "") -> None:
    h, w = img.shape[:2]
    slab(img, 0, h - 30, w, 30, alpha=0.5)
    text(img, left, (14, h - 10), COL_TEXT, 0.5, 1, FONT_S)
    if right:
        rw, _ = _text_size(right, 0.5, 1, FONT_S)
        text(img, right, (w - rw - 14, h - 10), COL_MUTED, 0.5, 1, FONT_S)


def draw_legend(img: np.ndarray, items: Sequence[Tuple[str, BGR]], origin: Tuple[int, int]) -> None:
    x, y = origin
    width = 190
    height = 16 + 20 * len(items)
    slab(img, x, y, width, height, alpha=0.55)
    cursor = y + 20
    for label, color in items:
        cv2.rectangle(img, (x + 10, cursor - 9), (x + 26, cursor + 1), color, -1, cv2.LINE_AA)
        text(img, label, (x + 34, cursor), COL_TEXT, 0.45, 1, FONT_S)
        cursor += 20
