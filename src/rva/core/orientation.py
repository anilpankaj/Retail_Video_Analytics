"""Where is this person looking?

At CCTV resolution a passer-by in the mall walkway is 60-110 px tall, so a
face/gaze model has nothing to work with: the head is a dozen pixels and is
often seen from above or behind.  We therefore estimate *body* orientation,
which is what the brief actually asks for ("turning their head or body
toward it").

Two independent estimates are fused:

1. **Pose normal.**  COCO-17 keypoints give the shoulder line.  The facing
   direction is its perpendicular.  The two-fold sign ambiguity is resolved
   from image-space shoulder order: for a person facing the camera the *left*
   shoulder appears on the *right* of the image.  Concretely, with
   ``s = right_shoulder - left_shoulder`` the facing vector is
   ``(s.y, -s.x)`` normalised.  Head yaw is then added as a small correction
   from the nose offset relative to the shoulder mid-point, which recovers
   "walking one way while looking sideways at the window".

2. **Motion direction.**  A walking person faces where they are going.  This
   is far more reliable than pose when the person is small, but meaningless
   when they are standing still.

The two are blended with a weight driven by normalised speed: fast movers
trust motion, near-stationary people trust pose.  Every estimate carries a
confidence so downstream rules can require evidence rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import math

import numpy as np

from .geometry import normalize

# COCO-17 keypoint indices produced by the YOLO pose models.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

Vector = Tuple[float, float]


@dataclass
class Orientation:
    """Fused facing estimate for one person at one instant."""

    vector: Vector = (0.0, 0.0)
    confidence: float = 0.0
    source: str = "none"  # pose | motion | fused | none

    @property
    def is_valid(self) -> bool:
        return self.confidence > 0.0 and (self.vector[0] or self.vector[1])


def pose_facing(
    keypoints: Optional[np.ndarray],
    kp_conf_min: float = 0.35,
    head_yaw_gain_deg: float = 55.0,
) -> Orientation:
    """Facing direction from the shoulder line, corrected by head yaw.

    ``keypoints`` is an ``(17, 3)`` array of ``(x, y, confidence)``.
    """
    if keypoints is None or len(keypoints) < 17:
        return Orientation()

    left = keypoints[L_SHOULDER]
    right = keypoints[R_SHOULDER]
    if left[2] < kp_conf_min or right[2] < kp_conf_min:
        return Orientation()

    shoulder_vec = (float(right[0] - left[0]), float(right[1] - left[1]))
    shoulder_width = math.hypot(*shoulder_vec)
    if shoulder_width < 4.0:  # degenerate: person seen edge-on or too small
        return Orientation()

    # Perpendicular with the sign convention derived in the module docstring.
    facing = normalize((shoulder_vec[1], -shoulder_vec[0]))

    # Head-yaw correction: how far the nose sits off the shoulder mid-point,
    # projected onto the shoulder axis, in units of half the shoulder width.
    nose = keypoints[NOSE]
    if nose[2] >= kp_conf_min:
        mid = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
        axis = normalize(shoulder_vec)
        offset = (float(nose[0]) - mid[0]) * axis[0] + (float(nose[1]) - mid[1]) * axis[1]
        ratio = max(-1.0, min(1.0, offset / (shoulder_width / 2.0)))
        yaw = math.radians(head_yaw_gain_deg) * ratio
        cos_a, sin_a = math.cos(yaw), math.sin(yaw)
        facing = (
            facing[0] * cos_a - facing[1] * sin_a,
            facing[0] * sin_a + facing[1] * cos_a,
        )
        facing = normalize(facing)

    confidence = float(min(left[2], right[2]))
    # Very narrow shoulder lines mean the person is nearly edge-on and the
    # perpendicular is unstable; damp the confidence accordingly.
    hips_present = keypoints[L_HIP][2] >= kp_conf_min and keypoints[R_HIP][2] >= kp_conf_min
    if hips_present:
        confidence = min(1.0, confidence * 1.15)
    return Orientation(vector=facing, confidence=confidence, source="pose")


def motion_facing(velocity: Sequence[float], speed_norm: float, walk_speed_norm: float) -> Orientation:
    """Facing direction inferred from where the person is walking."""
    vec = normalize(velocity)
    if vec == (0.0, 0.0) or speed_norm <= 0:
        return Orientation()
    # Confidence ramps up between "shuffling" and "walking".
    confidence = max(0.0, min(1.0, speed_norm / max(walk_speed_norm, 1e-6)))
    return Orientation(vector=vec, confidence=confidence, source="motion")


def fuse(pose: Orientation, motion: Orientation, motion_bias: float = 0.65) -> Orientation:
    """Blend the pose and motion estimates into a single facing vector."""
    if not pose.is_valid and not motion.is_valid:
        return Orientation()
    if not pose.is_valid:
        return motion
    if not motion.is_valid:
        return pose

    w_motion = motion.confidence * motion_bias
    w_pose = pose.confidence * (1.0 - motion_bias) + pose.confidence * (1.0 - motion.confidence)
    total = w_motion + w_pose
    if total < 1e-6:
        return pose
    vec = normalize(
        (
            (pose.vector[0] * w_pose + motion.vector[0] * w_motion) / total,
            (pose.vector[1] * w_pose + motion.vector[1] * w_motion) / total,
        )
    )
    return Orientation(vector=vec, confidence=max(pose.confidence, motion.confidence), source="fused")


def best_wrist(keypoints: Optional[np.ndarray], kp_conf_min: float = 0.35) -> Optional[Tuple[float, float]]:
    """Image position of the most confident wrist, if either is visible.

    Used as a second reference point for shelf proximity: a customer bent
    over a low gondola has their hand *in* the shelf while their feet are
    still in the aisle, and on this footage the bounding box is often
    truncated by the frame edge, which makes the foot position the less
    honest of the two signals.
    """
    if keypoints is None or len(keypoints) < 17:
        return None
    best: Optional[Tuple[float, float]] = None
    best_conf = kp_conf_min
    for index in (L_WRIST, R_WRIST):
        kp = keypoints[index]
        if kp[2] >= best_conf:
            best_conf = float(kp[2])
            best = (float(kp[0]), float(kp[1]))
    return best


def reach_vector(keypoints: Optional[np.ndarray], kp_conf_min: float = 0.35) -> Optional[Vector]:
    """Direction the person is reaching, from shoulder mid-point to wrists.

    Reaching for a product is the single strongest evidence of shelf
    engagement, so Task 2 uses it as a tie-break when a customer stands
    between two shelves.
    """
    if keypoints is None or len(keypoints) < 17:
        return None
    left, right = keypoints[L_SHOULDER], keypoints[R_SHOULDER]
    if left[2] < kp_conf_min or right[2] < kp_conf_min:
        return None
    mid = np.array([(left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0])

    wrists = [keypoints[L_WRIST], keypoints[R_WRIST]]
    best: Optional[np.ndarray] = None
    best_conf = kp_conf_min
    for wrist in wrists:
        if wrist[2] >= best_conf:
            best_conf = float(wrist[2])
            best = np.array([float(wrist[0]), float(wrist[1])])
    if best is None:
        return None
    return normalize((float(best[0] - mid[0]), float(best[1] - mid[1])))
