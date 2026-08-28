"""Task 1 — Store interest and conversion (``entrance.mp4``).

What has to be decided
----------------------
Three numbers: how many unique passers-by showed interest in the store, how
many of those then entered, and how many walked on.  The brief deliberately
leaves "interest" undefined and explicitly says it must **not** be reduced to
"the person stopped".

Definition of interest used here
-------------------------------
A passer-by is scored on four observable behaviours, each of which the brief
names, each measured in scale-invariant units so it works equally well for a
person near the camera and one at the far end of the walkway:

===============  ==========================================================
signal           measurement
===============  ==========================================================
``proximity``    distance from the storefront line, divided by the person's
                 own pixel height (a "body-heights" distance).  This is a
                 hard gate: someone on the far side of the walkway is not
                 looking at *this* store in any meaningful sense.
``facing``       angle between the fused body/head facing vector and the
                 vector from the person to the nearest point on the
                 storefront line.  This is "turning their head or body
                 toward it".
``slowing``      normalised speed below a walking threshold — "slowing
                 down".  Note it is *slowing*, not *stopped*: the threshold
                 sits at roughly half a normal walking pace.
``approach``     the rate of change of the distance to the storefront is
                 negative — "approaching the entrance".
===============  ==========================================================

The four are combined as a weighted sum and compared with a threshold, so no
single behaviour is necessary or sufficient: a person who slows and turns
qualifies, and so does a person who walks straight at the entrance without
ever slowing.  The score must stay above threshold for ``min_evidence_s``
before the person is latched as interested, which is what stops a single
noisy frame from creating a customer.

Interest is *latched* per person: once shown, it is never withdrawn, so
"Total Interested" is a count of unique people.

Entered vs passed by
--------------------
``ENTERED``  the person's feet are inside the store polygon, at least
``min_depth_norm`` body-heights past the storefront line, continuously for
``confirm_s`` seconds.  The depth and time requirements together implement
"crosses the entrance boundary **and continues into the store**" and reject
someone who leans in to look at a display and steps back out.

``PASSED_BY``  an interested person who left the walkway, or reached the end
of the clip, without ever satisfying the entry condition.

Because the two outcomes are mutually exclusive and every interested person
receives exactly one of them, ``entered + passed_by == total_interested`` by
construction — a property asserted in the unit tests and re-checked before
the CSV is written.

Staff are excluded from all three counts: they loiter at the storefront all
day and would otherwise dominate the "interested" number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math

from ..config import Config, as_points
from ..core.events import Latch
from ..core.geometry import Polygon, Polyline, angle_between, normalize
from ..core.tracking import TrackState

INTERESTED = "INTERESTED"
ENTERED = "ENTERED"
PASSED_BY = "PASSED_BY"
NEUTRAL = "-"


@dataclass
class PersonOutcome:
    track_id: int
    interested: bool = False
    interested_at: Optional[float] = None
    outcome: str = NEUTRAL          # ENTERED | PASSED_BY | NEUTRAL
    outcome_at: Optional[float] = None
    peak_score: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    max_depth_norm: float = 0.0
    reason: str = ""
    origin: str = ""
    frames_scored: int = 0
    frames_above: int = 0
    longest_run_s: float = 0.0
    entered_without_interest: bool = False


@dataclass
class PersonRuntime:
    """Per-track scratch state for the interest/entry machines."""

    interest_latch: Latch
    entry_latch: Latch
    outside_latch: Latch
    prev_dist_norm: Optional[float] = None
    prev_t: Optional[float] = None
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    dist_norm: float = math.inf
    depth_norm: float = 0.0
    #: "walkway" or "store" - where this person was first seen
    origin: str = ""
    walkway_s: float = 0.0
    last_update_t: Optional[float] = None
    run_start_t: Optional[float] = None


class StoreInterestTask:
    """Task 1 pipeline stage."""

    name = "task1"

    def __init__(self, cfg: Config) -> None:
        scene = cfg.require("scene")
        self.store = Polygon(as_points(scene["store_polygon"]), "store")
        self.walkway = Polygon(as_points(scene["walkway_polygon"]), "walkway")
        self.front = Polyline(as_points(scene["storefront_line"]), "storefront")
        self.roi_top_y = float(scene.get("roi_top_y", 0))

        interest = cfg.require("task1.interest")
        self.max_dist = float(interest["max_norm_dist_to_front"])
        self.facing_angle = float(interest["facing_angle_deg"])
        self.slow_speed = float(interest["slow_speed_norm"])
        self.approach_rate = float(interest["approach_rate_norm"])
        # Closing speed (body-heights/second) at which "approaching" scores 1.0.
        self.approach_full_rate = float(interest.get("approach_full_rate_norm", 0.8))
        self.min_evidence_s = float(interest["min_evidence_s"])
        self.evidence_tolerance_s = float(interest.get("evidence_tolerance_s", 0.5))
        self.weights = dict(interest["score_weights"])
        self.threshold = float(interest["score_threshold"])
        self.min_track_s = float(interest.get("min_track_seconds", 0.6))
        self.require_walkway_origin = bool(interest.get("require_walkway_origin", True))
        self.min_walkway_s = float(interest.get("min_walkway_seconds", 0.4))

        entry = cfg.require("task1.entry")
        self.entry_confirm_s = float(entry["confirm_s"])
        self.min_depth_norm = float(entry["min_depth_norm"])

        exit_cfg = cfg.get_path("task1.exit", {}) or {}
        self.passby_confirm_s = float(exit_cfg.get("passby_confirm_s", 1.0))

        self.exclude_staff = bool(cfg.get_path("task1.exclude_staff", True))

        self.runtime: Dict[int, PersonRuntime] = {}
        self.outcomes: Dict[int, PersonOutcome] = {}
        self.events: List[dict] = []

    # ------------------------------------------------------------------ API #
    def in_scene(self, state: TrackState) -> bool:
        """Is this track a person on our floor at all?"""
        ground = state.ground
        if ground[1] < self.roi_top_y:
            return False
        return self.walkway.contains(ground) or self.store.contains(ground)

    def update(self, t: float, tracks: List[TrackState]) -> None:
        for state in tracks:
            if not self.in_scene(state):
                continue
            if self.exclude_staff and state.role == "staff":
                continue

            runtime = self.runtime.get(state.track_id)
            if runtime is None:
                runtime = PersonRuntime(
                    interest_latch=Latch(self.min_evidence_s, self.evidence_tolerance_s),
                    entry_latch=Latch(self.entry_confirm_s, 0.4),
                    outside_latch=Latch(self.passby_confirm_s, 0.4),
                )
                self.runtime[state.track_id] = runtime
            outcome = self.outcomes.setdefault(
                state.track_id, PersonOutcome(track_id=state.track_id, first_seen=t)
            )
            outcome.last_seen = t

            on_walkway = self.walkway.contains(state.ground) and not self.store.contains(state.ground)
            if not runtime.origin:
                runtime.origin = "walkway" if on_walkway else "store"
                outcome.origin = runtime.origin
            if on_walkway and runtime.last_update_t is not None:
                runtime.walkway_s += max(0.0, t - runtime.last_update_t)
            runtime.last_update_t = t

            # Task 1 is about PASSERS-BY converting. Someone whose track begins
            # already inside the store was shopping before the clip started (or
            # walked in off-camera); they are not a passer-by and counting them
            # would inflate both "interested" and "entered".
            if self.require_walkway_origin and runtime.origin != "walkway":
                continue

            # The brief asks for "those interested people who SUBSEQUENTLY
            # entered", so interest has to be established while the person is
            # still a passer-by. Once entry is confirmed the interest score is
            # frozen: wandering around inside the store afterwards must not be
            # able to retro-fit the evidence that justified the conversion.
            if outcome.outcome != ENTERED:
                self._score(t, state, runtime)
            else:
                runtime.score = 0.0
            outcome.peak_score = max(outcome.peak_score, runtime.score)

            # Diagnostics: how much of this person's time was spent above the
            # interest threshold, and what their longest unbroken run was.
            # Written to the audit CSV so a reviewer can see exactly how close
            # a borderline person came to being counted.
            outcome.frames_scored += 1
            if runtime.score >= self.threshold:
                outcome.frames_above += 1
                if runtime.run_start_t is None:
                    runtime.run_start_t = t
                outcome.longest_run_s = max(outcome.longest_run_s, t - runtime.run_start_t)
            else:
                runtime.run_start_t = None

            outcome.max_depth_norm = max(outcome.max_depth_norm, runtime.depth_norm)

            # ---- interest latch ------------------------------------------ #
            long_enough = (
                state.duration >= self.min_track_s and runtime.walkway_s >= self.min_walkway_s
            )
            if runtime.interest_latch.update(t, runtime.score >= self.threshold and long_enough):
                if not outcome.interested:
                    outcome.interested = True
                    outcome.interested_at = runtime.interest_latch.triggered_at
                    outcome.reason = self._explain(runtime)
                    self._log(t, "interest", state.track_id, runtime)

            # ---- conversion ---------------------------------------------- #
            inside = self.store.contains(state.ground) and runtime.depth_norm >= self.min_depth_norm
            if runtime.entry_latch.update(t, inside):
                if outcome.outcome != ENTERED:
                    outcome.outcome = ENTERED
                    outcome.outcome_at = runtime.entry_latch.triggered_at
                    # A person who walks in without ever showing observable
                    # interest first is not counted in ANY of the three metrics:
                    # they are a customer, but not one this camera saw convert.
                    outcome.entered_without_interest = not outcome.interested
                    self._log(t, "entered", state.track_id, runtime)

            if outcome.outcome != ENTERED:
                left_scene = not self.walkway.contains(state.ground) and not self.store.contains(state.ground)
                if runtime.outside_latch.update(t, left_scene) and outcome.interested:
                    outcome.outcome = PASSED_BY
                    outcome.outcome_at = runtime.outside_latch.triggered_at
                    self._log(t, "passed_by", state.track_id, runtime)

    def finalize(self, t_end: float) -> None:
        """Resolve anyone still pending when the clip ends."""
        for outcome in self.outcomes.values():
            if outcome.interested and outcome.outcome == NEUTRAL:
                # Never confirmed inside the store => they did not enter within
                # the observed window.  Documented assumption: unresolved
                # interested people are attributed to "passed by".
                outcome.outcome = PASSED_BY
                outcome.outcome_at = outcome.last_seen
                self.events.append(
                    {"t": round(outcome.last_seen, 2), "event": "passed_by_eof",
                     "track_id": outcome.track_id}
                )

    # -------------------------------------------------------------- results #
    @property
    def total_interested(self) -> int:
        return sum(1 for o in self.outcomes.values() if o.interested)

    @property
    def interested_entered(self) -> int:
        return sum(1 for o in self.outcomes.values() if o.interested and o.outcome == ENTERED)

    @property
    def interested_passed_by(self) -> int:
        return sum(1 for o in self.outcomes.values() if o.interested and o.outcome == PASSED_BY)

    def summary(self) -> Dict[str, int]:
        return {
            "total_interested": self.total_interested,
            "interested_entered": self.interested_entered,
            "interested_passed_by": self.interested_passed_by,
        }

    def per_person_rows(self) -> List[Dict[str, object]]:
        rows = []
        for outcome in sorted(self.outcomes.values(), key=lambda o: o.first_seen):
            rows.append(
                {
                    "track_id": outcome.track_id,
                    "origin": outcome.origin,
                    "first_seen_s": round(outcome.first_seen, 2),
                    "last_seen_s": round(outcome.last_seen, 2),
                    "interested": int(outcome.interested),
                    "interested_at_s": round(outcome.interested_at, 2) if outcome.interested_at else "",
                    "outcome": outcome.outcome if outcome.interested else (
                        "ENTERED_NO_PRIOR_INTEREST" if outcome.entered_without_interest else NEUTRAL
                    ),
                    "outcome_at_s": round(outcome.outcome_at, 2) if outcome.outcome_at else "",
                    "peak_interest_score": round(outcome.peak_score, 3),
                    "frames_scored": outcome.frames_scored,
                    "frames_above_threshold": outcome.frames_above,
                    "longest_run_above_s": round(outcome.longest_run_s, 2),
                    "max_depth_into_store_norm": round(outcome.max_depth_norm, 3),
                    "evidence": outcome.reason,
                }
            )
        return rows

    def state_of(self, track_id: int) -> Tuple[str, float]:
        """Display state and current score for the annotation layer."""
        outcome = self.outcomes.get(track_id)
        runtime = self.runtime.get(track_id)
        score = runtime.score if runtime else 0.0
        if outcome is None or not outcome.interested:
            return NEUTRAL, score
        if outcome.outcome == ENTERED:
            return ENTERED, score
        if outcome.outcome == PASSED_BY:
            return PASSED_BY, score
        return INTERESTED, score

    # ------------------------------------------------------------ internals #
    def _score(self, t: float, state: TrackState, runtime: PersonRuntime) -> None:
        ground = state.ground
        nearest = self.front.nearest_point(ground)
        dist_px = math.dist(ground, nearest)
        dist_norm = state.norm(dist_px)
        runtime.dist_norm = dist_norm

        inside_store = self.store.contains(ground)
        runtime.depth_norm = dist_norm if inside_store else 0.0

        # proximity gate — beyond this the person is not a candidate at all
        if dist_norm > self.max_dist and not inside_store:
            runtime.score = 0.0
            runtime.components = {}
            runtime.prev_dist_norm = dist_norm
            runtime.prev_t = t
            return

        components: Dict[str, float] = {}

        # proximity: 1 at the threshold line, rising as they get closer
        components["proximity"] = max(0.0, 1.0 - dist_norm / max(self.max_dist, 1e-6))

        # facing: the brief lists "looking toward the storefront, turning their
        # head OR body toward it" as separate behaviours, so we take the better
        # of the two estimates. The fused vector is dominated by walking
        # direction for a fast mover, which would erase a sideways glance; the
        # pose-only estimate (shoulder normal + head yaw) preserves it.
        to_front = normalize((nearest[0] - ground[0], nearest[1] - ground[1]))
        facing_term = 0.0
        if to_front != (0.0, 0.0):
            for estimate in (state.facing, state.head):
                if not estimate.is_valid:
                    continue
                angle = angle_between(estimate.vector, to_front)
                term = max(0.0, 1.0 - angle / max(self.facing_angle, 1e-6))
                facing_term = max(facing_term, term * max(0.35, estimate.confidence))
        components["facing"] = facing_term

        # slowing: below the "half walking pace" threshold
        components["slow"] = max(0.0, 1.0 - state.speed_norm / max(self.slow_speed, 1e-6))
        components["slow"] = min(1.0, components["slow"])

        # approach: closing on the storefront
        approach = 0.0
        if runtime.prev_dist_norm is not None and runtime.prev_t is not None:
            dt = t - runtime.prev_t
            if dt > 1e-6:
                rate = (dist_norm - runtime.prev_dist_norm) / dt  # body-heights per second
                if rate < self.approach_rate:
                    # Ramp from "barely closing" (approach_rate_norm) to
                    # "walking purposefully at the door" (approach_full_rate).
                    span = max(self.approach_full_rate - abs(self.approach_rate), 1e-6)
                    approach = min(1.0, (abs(rate) - abs(self.approach_rate)) / span)
        components["approach"] = approach

        runtime.prev_dist_norm = dist_norm
        runtime.prev_t = t
        runtime.components = components
        runtime.score = sum(self.weights.get(k, 0.0) * v for k, v in components.items())

    @staticmethod
    def _explain(runtime: PersonRuntime) -> str:
        if not runtime.components:
            return ""
        parts = [f"{k}={v:.2f}" for k, v in sorted(runtime.components.items(), key=lambda kv: -kv[1])]
        return " ".join(parts)

    def _log(self, t: float, event: str, track_id: int, runtime: PersonRuntime) -> None:
        self.events.append(
            {
                "t": round(t, 2),
                "event": event,
                "track_id": track_id,
                "score": round(runtime.score, 3),
                "dist_norm": round(runtime.dist_norm, 3),
                "components": {k: round(v, 3) for k, v in runtime.components.items()},
            }
        )
