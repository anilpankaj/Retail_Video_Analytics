"""Task 2 — Per-shelf customer interest (``interior.mp4``).

What has to be decided
----------------------
For each of the four designated shelves A-D, how many *interest events* were
observed.  Two sub-problems dominate: deciding which shelf a customer is
engaging with when they stand between two of them, and deciding when one
long visit ends and a separate return begins.

Shelf assignment — exclusive, scored, with a documented tie-break
----------------------------------------------------------------
On every processed frame each visible customer is scored against every shelf
and assigned to **at most one** — the highest scorer above threshold.  An
exclusive assignment is what makes the per-shelf totals meaningful: one
person at one instant is engaged with one shelf, so the same second of
attention can never be counted for two shelves.

The score has three terms, all scale-invariant:

``proximity``  distance from the customer's feet to the shelf fixture,
               divided by their pixel height.  Standing at a shelf puts you
               well under one body-height from it.
``facing``     angle between the fused body/head facing vector and the
               direction to the nearest point of the shelf.  A customer
               walking down the aisle with their back to a shelf is not
               interested in it.
``reach``      angle between the shoulder-to-wrist vector and the direction
               to the shelf.  Reaching for a product is the strongest
               possible evidence of engagement, and it is precisely what
               disambiguates the "standing in the gap between two shelves"
               case the brief calls out.  It only contributes when the wrist
               keypoint is actually confident, so it never invents evidence.

Continuous episode vs a genuine return
--------------------------------------
Assignment is fed into :class:`~rva.core.events.EpisodeTracker` per
(customer, shelf) pair with three temporal parameters:

* ``min_dwell_s`` — attention must be sustained this long before it counts
  at all, so walking past a shelf is not an event;
* ``flicker_tolerance_s`` — a shorter break (a missed detection, the person
  glancing away, an occlusion behind a gondola) does not end the episode;
* ``min_gap_between_events_s`` — after an episode ends, re-engagement within
  this window is merged back into it; re-engagement after it counts as a new
  event.  This is the explicit answer to "how do you distinguish a continuous
  interest episode from a later, separate return".

Staff working the floor are excluded when ``exclude_staff`` is set, since the
metric is *customer* interest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math

from ..config import Config, as_points
from ..core.events import EpisodeTracker
from ..core.geometry import Polygon, angle_between, normalize
from ..core.tracking import TrackState


@dataclass
class Shelf:
    shelf_id: str
    polygon: Polygon
    color: Tuple[int, int, int]
    label_anchor: Tuple[float, float]
    max_norm_dist: float


@dataclass
class ShelfScore:
    shelf_id: str
    score: float
    dist_norm: float
    facing_deg: float
    reach_deg: Optional[float]


class ShelfInterestTask:
    """Task 2 pipeline stage."""

    name = "task2"

    def __init__(self, cfg: Config) -> None:
        scene = cfg.require("scene")
        self.roi = Polygon(as_points(scene["roi_polygon"]), "roi")

        interest = cfg.require("task2.interest")
        self.default_max_dist = float(interest["max_norm_dist"])
        self.facing_angle = float(interest["facing_angle_deg"])
        self.reach_angle = float(interest.get("reach_angle_deg", 70.0))
        self.threshold = float(interest["score_threshold"])
        self.weights = dict(interest["score_weights"])
        self.min_dwell_s = float(interest["min_dwell_s"])
        self.flicker_s = float(interest["flicker_tolerance_s"])
        self.min_gap_s = float(interest["min_gap_between_events_s"])
        self.min_track_s = float(interest.get("min_track_seconds", 1.0))
        self.exclude_staff = bool(cfg.get_path("task2.exclude_staff", True))

        self.shelves: List[Shelf] = []
        for entry in scene["shelves"]:
            polygon = Polygon(as_points(entry["polygon"]), entry["id"])
            anchor = entry.get("label_anchor") or polygon.centroid
            self.shelves.append(
                Shelf(
                    shelf_id=str(entry["id"]),
                    polygon=polygon,
                    color=tuple(int(c) for c in entry.get("color", (0, 200, 255))),  # type: ignore[arg-type]
                    label_anchor=(float(anchor[0]), float(anchor[1])),
                    max_norm_dist=float(entry.get("max_norm_dist", self.default_max_dist)),
                )
            )
        self.shelf_ids = [s.shelf_id for s in self.shelves]

        # (track_id, shelf_id) -> EpisodeTracker
        self.trackers: Dict[Tuple[int, str], EpisodeTracker] = {}
        self.counts: Dict[str, int] = {s: 0 for s in self.shelf_ids}
        self.assignment: Dict[int, Optional[ShelfScore]] = {}
        self.events: List[dict] = []
        self.episodes: List[dict] = []

    # ------------------------------------------------------------------ API #
    def in_scene(self, state: TrackState) -> bool:
        return self.roi.contains(state.ground)

    def update(self, t: float, tracks: List[TrackState]) -> None:
        live_ids = set()
        for state in tracks:
            if not self.in_scene(state):
                self.assignment[state.track_id] = None
                continue
            if self.exclude_staff and state.role == "staff":
                self.assignment[state.track_id] = None
                continue
            live_ids.add(state.track_id)

            best = self._best_shelf(state)
            self.assignment[state.track_id] = best

            for shelf in self.shelves:
                key = (state.track_id, shelf.shelf_id)
                tracker = self.trackers.get(key)
                if tracker is None:
                    tracker = EpisodeTracker(
                        min_duration_s=self.min_dwell_s,
                        flicker_tolerance_s=self.flicker_s,
                        min_gap_s=self.min_gap_s,
                        label=shelf.shelf_id,
                    )
                    self.trackers[key] = tracker

                active = (
                    best is not None
                    and best.shelf_id == shelf.shelf_id
                    and state.duration >= self.min_track_s
                )
                episode = tracker.update(t, active, meta={"track_id": state.track_id})
                if episode is not None:
                    self.counts[shelf.shelf_id] += 1
                    self.events.append(
                        {
                            "t": round(t, 2),
                            "event": "shelf_interest",
                            "shelf": shelf.shelf_id,
                            "track_id": state.track_id,
                            "dist_norm": round(best.dist_norm, 3) if best else None,
                            "facing_deg": round(best.facing_deg, 1) if best else None,
                        }
                    )

        # tracks that vanished this frame get a false update so their episodes
        # can time out naturally
        for (track_id, shelf_id), tracker in self.trackers.items():
            if track_id not in live_ids:
                tracker.update(t, False)

    def finalize(self, t_end: float) -> None:
        for (track_id, shelf_id), tracker in self.trackers.items():
            tracker.finalize(t_end)
            for episode in tracker.episodes:
                self.episodes.append(
                    {
                        "shelf": shelf_id,
                        "track_id": track_id,
                        "start_s": round(episode.start_t, 2),
                        "end_s": round(episode.end_t, 2),
                        "duration_s": round(episode.duration, 2),
                        "bridged_gaps": episode.merges,
                    }
                )
        self.episodes.sort(key=lambda e: (e["shelf"], e["start_s"]))

    # -------------------------------------------------------------- results #
    def summary(self) -> Dict[str, int]:
        return dict(self.counts)

    def active_for(self, track_id: int, now: float) -> Optional[Tuple[str, float]]:
        """(shelf_id, elapsed seconds) for the interaction being displayed."""
        best = self.assignment.get(track_id)
        if best is None:
            return None
        tracker = self.trackers.get((track_id, best.shelf_id))
        if tracker is None or not tracker.is_active:
            return None
        return best.shelf_id, tracker.active_duration(now)

    def candidate_for(self, track_id: int) -> Optional[ShelfScore]:
        return self.assignment.get(track_id)

    def shelf_by_id(self, shelf_id: str) -> Shelf:
        for shelf in self.shelves:
            if shelf.shelf_id == shelf_id:
                return shelf
        raise KeyError(shelf_id)

    # ------------------------------------------------------------ internals #
    def _best_shelf(self, state: TrackState) -> Optional[ShelfScore]:
        ground = state.ground
        scored: List[ShelfScore] = []
        for shelf in self.shelves:
            nearest = shelf.polygon.nearest_point(ground)
            dist_px = 0.0 if shelf.polygon.contains(ground) else math.dist(ground, nearest)

            # Proximity is measured to the closest of the customer's feet and
            # their hand. A shopper bent over a low gondola has their hand IN
            # the shelf while their feet are still in the aisle, and on this
            # footage bounding boxes are frequently truncated by the frame
            # edge, which makes the foot position the less honest of the two.
            if state.wrist is not None:
                wrist_dist = (
                    0.0 if shelf.polygon.contains(state.wrist)
                    else math.dist(state.wrist, shelf.polygon.nearest_point(state.wrist))
                )
                dist_px = min(dist_px, wrist_dist)

            dist_norm = state.norm(dist_px)
            if dist_norm > shelf.max_norm_dist:
                continue

            to_shelf = normalize((nearest[0] - ground[0], nearest[1] - ground[1]))
            facing_deg = 180.0
            facing_term = 0.0
            if state.facing.is_valid and to_shelf != (0.0, 0.0):
                facing_deg = angle_between(state.facing.vector, to_shelf)
                facing_term = max(0.0, 1.0 - facing_deg / max(self.facing_angle, 1e-6))
                facing_term *= max(0.35, state.facing.confidence)

            reach_deg: Optional[float] = None
            reach_term = 0.0
            if state.reach is not None and to_shelf != (0.0, 0.0):
                reach_deg = angle_between(state.reach, to_shelf)
                reach_term = max(0.0, 1.0 - reach_deg / max(self.reach_angle, 1e-6))

            prox_term = max(0.0, 1.0 - dist_norm / max(shelf.max_norm_dist, 1e-6))

            score = (
                self.weights.get("proximity", 0.0) * prox_term
                + self.weights.get("facing", 0.0) * facing_term
                + self.weights.get("reach", 0.0) * reach_term
            )
            scored.append(
                ShelfScore(shelf.shelf_id, score, dist_norm, facing_deg, reach_deg)
            )

        if not scored:
            return None
        best = max(scored, key=lambda s: s.score)
        return best if best.score >= self.threshold else None
