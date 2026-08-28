"""Telling staff from customers.

The brief says staff are identifiable by their aprons, and inspection of the
entrance footage confirms it: the uniform is a **dark navy bib apron with a
small circular light-coloured badge on the chest**.  Staff also stand close
to the camera (120-250 px tall) whenever they are serving, so the torso is
large enough to measure directly.

Three independent, cheap and explainable signals are combined per track.
None of them is trustworthy alone in this scene — the store's armchairs and
carpet are the same navy as the apron, and plenty of customers wear black —
so the classifier only commits after it has seen enough observations.

``apron``      fraction of torso pixels that are dark *and* desaturated
``badge``      a small bright blob inside that dark region
``residency``  how long the track has lived inside the store polygon;
               customers transit, staff stay for the whole clip

The score is a fixed weighted sum, thresholded once per track with a minimum
observation count, and then latched — a track's role never flips mid-video,
which keeps the Task 3 denominator stable.

``--staff-report`` dumps the per-track features so the thresholds can be
audited or re-tuned without re-running inference blind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from .geometry import Polygon
from .tracking import TrackState, crop, torso_box


@dataclass
class StaffFeatures:
    """Running per-track evidence used to decide the role."""

    track_id: int
    n_obs: int = 0
    apron_ema: float = 0.0
    badge_rate: float = 0.0
    badge_hits: int = 0
    residency_s: float = 0.0
    observed_s: float = 0.0
    badge_obs: int = 0
    mean_height: float = 0.0
    score: float = 0.0
    role: str = "unknown"
    #: score at the moment the verdict was latched (diagnostic)
    decided_score: float = 0.0
    decided: bool = False

    def as_row(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "observations": self.n_obs,
            "observed_seconds": round(self.observed_s, 2),
            "in_store_seconds": round(self.residency_s, 2),
            "apron_dark_ratio": round(self.apron_ema, 4),
            "badge_detection_rate": round(self.badge_rate, 4),
            "badge_hits": self.badge_hits,
            "badge_observations": self.badge_obs,
            "mean_person_height_px": round(self.mean_height, 1),
            "staff_score": round(self.score, 4),
            "score_at_decision": round(self.decided_score, 4),
            "role": self.role,
        }


class StaffClassifier:
    """Scores every track and latches a ``customer`` / ``staff`` verdict."""

    def __init__(
        self,
        dark_v_max: int = 95,
        dark_s_max: int = 120,
        badge_v_min: int = 140,
        badge_min_fraction: float = 0.005,
        badge_max_fraction: float = 0.06,
        badge_min_aspect: float = 0.60,
        badge_min_fill: float = 0.42,
        badge_requires_apron: float = 0.40,
        badge_rate_target: float = 0.25,
        min_badge_height_px: int = 110,
        residency_target_s: float = 30.0,
        min_obs: int = 60,
        min_torso_px: int = 12,
        weights: Optional[Dict[str, float]] = None,
        score_threshold: float = 0.62,
        force_staff_ids: Optional[List[int]] = None,
        force_customer_ids: Optional[List[int]] = None,
    ) -> None:
        self.dark_v_max = int(dark_v_max)
        self.dark_s_max = int(dark_s_max)
        self.badge_v_min = int(badge_v_min)
        self.badge_min_fraction = float(badge_min_fraction)
        self.badge_min_aspect = float(badge_min_aspect)
        self.badge_min_fill = float(badge_min_fill)
        self.badge_requires_apron = float(badge_requires_apron)
        self.badge_rate_target = float(badge_rate_target)
        self.badge_max_fraction = float(badge_max_fraction)
        self.min_badge_height_px = int(min_badge_height_px)
        self.residency_target_s = float(residency_target_s)
        self.min_obs = int(min_obs)
        self.min_torso_px = int(min_torso_px)
        self.weights = weights or {"apron": 0.35, "badge": 0.45, "residency": 0.20}
        self.score_threshold = float(score_threshold)
        self.force_staff_ids = set(force_staff_ids or [])
        self.force_customer_ids = set(force_customer_ids or [])
        self.features: Dict[int, StaffFeatures] = {}

    # ------------------------------------------------------------------ API #
    def measure(
        self,
        state: TrackState,
        frame: np.ndarray,
        kp_conf_min: float = 0.35,
    ) -> Optional[tuple[float, float]]:
        """Per-frame appearance measurement: (apron ratio, badge 0/1).

        Split out from :meth:`observe` so a run can cache these two numbers per
        detection and every threshold downstream can then be re-tuned by
        replaying the cache, with no second inference pass over the video.
        """
        patch = crop(frame, torso_box(state.box, state.keypoints, kp_conf_min))
        if patch is None or min(patch.shape[:2]) < self.min_torso_px:
            return None
        return self._appearance(patch)

    def observe(
        self,
        state: TrackState,
        frame: np.ndarray,
        dt: float,
        inside_store: bool,
        kp_conf_min: float = 0.35,
    ) -> str:
        return self.observe_features(
            state, self.measure(state, frame, kp_conf_min), dt, inside_store
        )

    def observe_features(
        self,
        state: TrackState,
        appearance: Optional[tuple[float, float]],
        dt: float,
        inside_store: bool,
    ) -> str:
        feat = self.features.setdefault(state.track_id, StaffFeatures(track_id=state.track_id))
        feat.observed_s += dt
        if inside_store:
            feat.residency_s += dt
        feat.mean_height = (feat.mean_height * feat.n_obs + state.height) / (feat.n_obs + 1)

        if appearance is not None:
            apron, badge = appearance
            # A badge is only ~15 px across on a 150 px person; below that the
            # shape test cannot mean anything, so we record "no evidence"
            # rather than "no badge".
            resolvable = state.height >= self.min_badge_height_px
            alpha = 0.2
            if feat.n_obs == 0:
                feat.apron_ema = apron
            else:
                feat.apron_ema = (1 - alpha) * feat.apron_ema + alpha * apron
            feat.n_obs += 1
            if resolvable:
                # The badge is scored by DETECTION RATE, not by an EMA. It is
                # occluded by an arm or a customer on many frames, so what
                # matters is that it is seen repeatedly - not that it is seen
                # every time. A stray false positive on a customer produces a
                # rate near zero and is diluted away.
                feat.badge_obs += 1
                feat.badge_hits += 1 if badge > 0 else 0
                feat.badge_rate = feat.badge_hits / feat.badge_obs

        feat.score = self._score(feat)
        feat.role = self._decide(feat)
        state.role = feat.role
        state.role_score = feat.score
        return feat.role

    def finalize(self) -> None:
        """Take the verdict for every track, once, from whole-clip evidence.

        The pipeline is deliberately two-pass (see ``rva.pipeline``): the model
        runs first and accumulates evidence, this is called, and only then are
        the task state machines driven. Any provisional verdict reached mid-clip
        is discarded here, because a track's early frames are exactly when the
        badge detection rate and the apron average are least settled.
        """
        for feat in self.features.values():
            feat.decided = False
            feat.role = self._decide(feat, final=True)

    def role_of(self, track_id: int) -> str:
        feat = self.features.get(track_id)
        return feat.role if feat else "customer"

    def score_of(self, track_id: int) -> float:
        feat = self.features.get(track_id)
        return feat.score if feat else 0.0

    def report_rows(self) -> List[Dict[str, object]]:
        return [f.as_row() for f in sorted(self.features.values(), key=lambda f: -f.score)]

    # ------------------------------------------------------------ internals #
    def _appearance(self, patch: np.ndarray) -> tuple[float, float]:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        dark = ((v <= self.dark_v_max) & (s <= self.dark_s_max)).astype(np.uint8)
        total = dark.size
        apron = float(dark.sum()) / max(total, 1)

        # Badge detection is a *shape* test, not a brightness fraction.
        #
        # Plenty of customers wear dark clothing, so "dark torso" alone is a
        # weak signal (measured dark ratios of 0.83-0.96 on genuine shoppers).
        # What is specific to the uniform is a small, compact, round, bright
        # badge sitting on that dark field. We therefore look for a connected
        # bright component whose area is a small fraction of the torso and
        # whose shape is close to circular, and whose centroid falls inside
        # the dark region. A light-coloured shirt fails the area test; a
        # highlight on a shoulder fails the circularity test.
        badge = 0.0
        if apron >= self.badge_requires_apron:
            bright = (v >= self.badge_v_min).astype(np.uint8)
            n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(bright, 8)
            for index in range(1, n_labels):
                area = float(stats[index, cv2.CC_STAT_AREA])
                ratio = area / max(total, 1)
                if not (self.badge_min_fraction <= ratio <= self.badge_max_fraction):
                    continue
                w = float(stats[index, cv2.CC_STAT_WIDTH])
                h = float(stats[index, cv2.CC_STAT_HEIGHT])
                if w < 2 or h < 2:
                    continue
                aspect = min(w, h) / max(w, h)
                fill = area / (w * h)
                # A solid disc fills ~0.79 of its bounding box, but this badge
                # carries a dark logo, so its bright mask is a ring: 0.42 is
                # the measured floor on real staff crops, while a lit shoulder
                # or a bag strap comes in well below it.
                if aspect < self.badge_min_aspect or fill < self.badge_min_fill:
                    continue
                cx, cy = int(centroids[index][0]), int(centroids[index][1])
                if 0 <= cy < dark.shape[0] and 0 <= cx < dark.shape[1] and not dark[cy, cx]:
                    # the blob's own pixels are bright; check its surroundings
                    y0, y1 = max(0, cy - int(h)), min(dark.shape[0], cy + int(h) + 1)
                    x0, x1 = max(0, cx - int(w)), min(dark.shape[1], cx + int(w) + 1)
                    neighbourhood = dark[y0:y1, x0:x1]
                    if neighbourhood.size == 0 or neighbourhood.mean() < 0.4:
                        continue
                badge = 1.0
                break
        return apron, badge

    def _score(self, feat: StaffFeatures) -> float:
        residency = min(1.0, feat.residency_s / max(self.residency_target_s, 1e-6))
        badge = min(1.0, feat.badge_rate / max(self.badge_rate_target, 1e-6))
        return (
            self.weights["apron"] * feat.apron_ema
            + self.weights["badge"] * badge
            + self.weights["residency"] * residency
        )

    def _decide(self, feat: StaffFeatures, final: bool = False) -> str:
        """One verdict per track, taken once and never revisited.

        Latching matters for consistency, not just stability: Task 1 excludes
        staff and Task 3 counts them, both while the video is being annotated,
        so a role that changed mid-clip would leave the on-screen counters, the
        CSVs and the audit trail disagreeing about who was who. The verdict is
        therefore taken at the first frame where the track has enough evidence
        (``min_obs`` torso observations, i.e. 6 s at the 10 Hz analysis rate)
        and then frozen.
        """
        if feat.track_id in self.force_staff_ids:
            return "staff"
        if feat.track_id in self.force_customer_ids:
            return "customer"
        if feat.decided:
            return feat.role
        if feat.n_obs < self.min_obs:
            # Not enough evidence to call anyone staff. Before the end of the
            # video that is "undecided"; at the end it resolves to customer,
            # because a person we barely saw cannot be *shown* to be staff and
            # admitting them would add a spurious zero to the Task 3 average.
            if final:
                feat.decided = True
                feat.decided_score = feat.score
                return "customer"
            return "unknown"
        feat.decided = True
        feat.decided_score = feat.score
        return "staff" if feat.score >= self.score_threshold else "customer"


def inside_any(polygons: List[Polygon], point) -> bool:
    return any(poly.contains(point) for poly in polygons)
