"""The apron / badge / residency staff classifier.

The badge test is a shape test, so it can be exercised with synthetic torso
patches: a dark field with a small bright disc on it is a uniform; the same
dark field with a large bright region is a light garment; a bright sliver is a
highlight or a bag strap.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rva.core.staff import StaffClassifier, StaffFeatures


def torso(dark_value: int = 40, size: int = 80) -> np.ndarray:
    """A plain dark, desaturated torso patch (BGR)."""
    return np.full((size, size, 3), dark_value, dtype=np.uint8)


def with_disc(patch: np.ndarray, radius: int, value: int = 220) -> np.ndarray:
    out = patch.copy()
    centre = (out.shape[1] // 2, out.shape[0] // 2)
    cv2.circle(out, centre, radius, (value, value, value), -1)
    return out


def with_rect(patch: np.ndarray, w: int, h: int, value: int = 220) -> np.ndarray:
    out = patch.copy()
    x = (out.shape[1] - w) // 2
    y = (out.shape[0] - h) // 2
    cv2.rectangle(out, (x, y), (x + w, y + h), (value, value, value), -1)
    return out


@pytest.fixture
def clf() -> StaffClassifier:
    return StaffClassifier()


def test_plain_dark_torso_is_an_apron_but_has_no_badge(clf: StaffClassifier) -> None:
    apron, badge = clf._appearance(torso())
    assert apron > 0.95
    assert badge == 0.0


def test_light_torso_is_not_an_apron(clf: StaffClassifier) -> None:
    apron, badge = clf._appearance(np.full((80, 80, 3), 200, dtype=np.uint8))
    assert apron < 0.05
    assert badge == 0.0


def test_small_bright_disc_on_a_dark_field_reads_as_a_badge(clf: StaffClassifier) -> None:
    # radius 8 on an 80x80 patch => ~3.1% of the area, matching the real badge
    apron, badge = clf._appearance(with_disc(torso(), radius=8))
    assert badge == 1.0
    assert apron > 0.9


def test_a_large_bright_area_is_a_garment_not_a_badge(clf: StaffClassifier) -> None:
    _, badge = clf._appearance(with_disc(torso(), radius=30))
    assert badge == 0.0, "a bright region covering ~44% of the torso is clothing"


def test_a_bright_sliver_is_a_highlight_not_a_badge(clf: StaffClassifier) -> None:
    _, badge = clf._appearance(with_rect(torso(), w=40, h=3))
    assert badge == 0.0, "a long thin blob fails the aspect-ratio test"


def test_a_tiny_speck_is_noise_not_a_badge(clf: StaffClassifier) -> None:
    _, badge = clf._appearance(with_disc(torso(), radius=1))
    assert badge == 0.0


def test_no_badge_search_on_a_light_torso(clf: StaffClassifier) -> None:
    """The badge search is gated on the torso being dark in the first place."""
    light = np.full((80, 80, 3), 200, dtype=np.uint8)
    _, badge = clf._appearance(with_disc(light, radius=8, value=255))
    assert badge == 0.0


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def make_features(apron: float, badge_rate: float, residency: float, n_obs: int = 200) -> StaffFeatures:
    feat = StaffFeatures(track_id=1)
    feat.apron_ema = apron
    feat.badge_rate = badge_rate
    feat.residency_s = residency
    feat.n_obs = n_obs
    return feat


def test_staff_profile_scores_above_threshold(clf: StaffClassifier) -> None:
    feat = make_features(apron=0.55, badge_rate=0.30, residency=40)
    feat.score = clf._score(feat)
    assert feat.score > 0.62
    assert clf._decide(feat) == "staff"


def test_dark_clothed_customer_without_a_badge_stays_a_customer(clf: StaffClassifier) -> None:
    """The measured failure mode: shoppers in this footage score 0.83-0.96 apron."""
    feat = make_features(apron=0.90, badge_rate=0.0, residency=300)
    feat.score = clf._score(feat)
    assert feat.score < 0.62
    assert clf._decide(feat) == "customer"


def test_a_stray_badge_false_positive_does_not_promote_a_customer(clf: StaffClassifier) -> None:
    feat = make_features(apron=0.90, badge_rate=0.05, residency=300)
    feat.score = clf._score(feat)
    assert clf._decide(feat) == "customer"


def test_a_short_track_is_never_called_staff(clf: StaffClassifier) -> None:
    """Precision guard: 1-3 second fragments produced 19 false 'staff' at
    min_obs=12, so the verdict now needs sustained observation."""
    feat = make_features(apron=0.95, badge_rate=1.0, residency=30, n_obs=clf.min_obs - 1)
    feat.score = clf._score(feat)
    assert feat.score > 0.62
    assert clf._decide(feat) == "unknown"
    assert clf._decide(feat, final=True) == "customer"


def test_manual_overrides_win(clf: StaffClassifier) -> None:
    clf.force_staff_ids = {7}
    clf.force_customer_ids = {8}
    staff = make_features(apron=0.0, badge_rate=0.0, residency=0)
    staff.track_id = 7
    customer = make_features(apron=1.0, badge_rate=1.0, residency=300)
    customer.track_id = 8
    assert clf._decide(staff) == "staff"
    assert clf._decide(customer) == "customer"


def test_residency_alone_cannot_create_staff(clf: StaffClassifier) -> None:
    feat = make_features(apron=0.0, badge_rate=0.0, residency=10_000)
    feat.score = clf._score(feat)
    assert feat.score <= clf.weights["residency"]
    assert clf._decide(feat) == "customer"
