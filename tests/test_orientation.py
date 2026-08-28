"""Body-orientation estimation from pose keypoints and motion."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_keypoints
from rva.core.geometry import angle_between
from rva.core.orientation import Orientation, fuse, motion_facing, pose_facing, reach_vector


def test_facing_camera_points_down_the_image() -> None:
    """A person facing the camera has their LEFT shoulder on the image right."""
    kps = make_keypoints(shoulder_left=(120, 100), shoulder_right=(80, 100))
    result = pose_facing(kps)
    assert result.is_valid
    assert angle_between(result.vector, (0, 1)) < 15.0


def test_facing_away_points_up_the_image() -> None:
    kps = make_keypoints(shoulder_left=(80, 100), shoulder_right=(120, 100))
    result = pose_facing(kps)
    assert result.is_valid
    assert angle_between(result.vector, (0, -1)) < 15.0


def test_profile_view_points_sideways() -> None:
    """Shoulders stacked vertically in the image means the person is side-on.

    Facing image-left: with your head up and your body turned west, your left
    shoulder is the southern one, i.e. the one lower down the image.
    """
    kps = make_keypoints(shoulder_left=(100, 140), shoulder_right=(100, 100))
    result = pose_facing(kps)
    assert result.is_valid
    assert angle_between(result.vector, (-1, 0)) < 20.0

    # ...and swapping the shoulders flips the facing direction.
    flipped = pose_facing(make_keypoints(shoulder_left=(100, 100), shoulder_right=(100, 140)))
    assert flipped.is_valid
    assert angle_between(flipped.vector, (1, 0)) < 20.0


def test_head_yaw_rotates_the_estimate() -> None:
    """Walking one way while looking sideways must shift the facing vector."""
    straight = pose_facing(make_keypoints((120, 100), (80, 100), nose=(100, 88)))
    turned = pose_facing(make_keypoints((120, 100), (80, 100), nose=(118, 88)))
    assert straight.is_valid and turned.is_valid
    delta = angle_between(straight.vector, turned.vector)
    assert delta > 15.0, "a clear head turn must move the facing estimate"


def test_low_confidence_keypoints_are_rejected() -> None:
    kps = make_keypoints((120, 100), (80, 100), conf=0.1)
    assert not pose_facing(kps).is_valid


def test_degenerate_shoulder_line_is_rejected() -> None:
    kps = make_keypoints((100, 100), (101, 100))
    assert not pose_facing(kps).is_valid


def test_missing_keypoints_are_safe() -> None:
    assert not pose_facing(None).is_valid
    assert not pose_facing(np.zeros((3, 3))).is_valid


def test_motion_facing_tracks_velocity_and_scales_confidence() -> None:
    slow = motion_facing((10.0, 0.0), speed_norm=0.1, walk_speed_norm=0.9)
    fast = motion_facing((10.0, 0.0), speed_norm=1.2, walk_speed_norm=0.9)
    assert slow.vector == pytest.approx((1.0, 0.0))
    assert fast.confidence > slow.confidence
    assert fast.confidence == pytest.approx(1.0)


def test_fuse_prefers_motion_for_a_walking_person() -> None:
    pose = Orientation(vector=(0.0, 1.0), confidence=0.5, source="pose")
    motion = Orientation(vector=(1.0, 0.0), confidence=1.0, source="motion")
    result = fuse(pose, motion, motion_bias=0.8)
    assert angle_between(result.vector, (1.0, 0.0)) < 45.0


def test_fuse_falls_back_when_one_side_is_missing() -> None:
    pose = Orientation(vector=(0.0, 1.0), confidence=0.8, source="pose")
    assert fuse(pose, Orientation()).source == "pose"
    assert fuse(Orientation(), Orientation()).is_valid is False


def test_reach_vector_points_at_the_wrist() -> None:
    kps = make_keypoints((120, 100), (80, 100), wrist=(100, 40))
    reach = reach_vector(kps)
    assert reach is not None
    assert angle_between(reach, (0, -1)) < 10.0


def test_reach_vector_absent_without_a_confident_wrist() -> None:
    assert reach_vector(make_keypoints((120, 100), (80, 100))) is None
