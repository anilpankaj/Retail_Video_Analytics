"""Task 3 — Staff-customer interaction (``entrance.mp4``).

The metric
----------
``average interaction sessions per staff member`` = total sessions / number
of staff instances, **including staff instances with zero sessions**.  Per
the brief a staff instance is one continuous appearance in the camera view;
re-identification across separate appearances is optional, so one track id
(after fragment linking) is exactly one staff instance.

What counts as an interaction
-----------------------------
An interaction requires evidence that a staff member and a customer are
*actively engaging with one another*, which is stronger than merely being
near each other in a small shop.  Two routes qualify:

**Engaged-facing** — the pair are within ``max_norm_dist`` body-heights of
each other *and* at least one of them is oriented towards the other within
``facing_angle_deg``.  Requiring only one direction rather than both is a
deliberate choice: in a fitting interaction the staff member is usually
kneeling and facing the customer while the customer looks down at the shoe,
so a mutual-gaze rule would miss the single most common interaction in this
footage.

**Co-stationary** — the pair are close *and* both are nearly stationary for
the duration.  This is the fallback for the seating area at the bottom of the
frame, where the staff member kneels, is heavily truncated by the frame edge
and occluded by the armchair, and pose keypoints are unreliable.  Without it
the pipeline would systematically under-count exactly the interactions the
store cares most about.

Sessions
--------
Per (staff, customer) pair the per-frame engagement flag is fed to an
:class:`~rva.core.events.EpisodeTracker`:

* ``min_session_s`` — engagement must be sustained before it is a session,
  which rejects a customer merely walking past a staff member;
* ``flicker_tolerance_s`` — brief detection dropouts do not split a session;
* ``end_gap_s`` — once engagement stops for longer than this the session
  closes; a later re-engagement between the same pair is counted as a new
  session, which is precisely the behaviour the brief specifies.

Because sessions are tracked per pair, the same staff member serving two
customers at once accrues two sessions, and the same customer returning to
the same staff member later accrues a second session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math

from ..config import Config
from ..core.events import EpisodeTracker
from ..core.geometry import angle_between, normalize
from ..core.tracking import TrackState


@dataclass
class ActiveLink:
    staff_id: int
    customer_id: int
    started_at: float
    duration: float
    reason: str


class StaffInteractionTask:
    """Task 3 pipeline stage."""

    name = "task3"

    def __init__(self, cfg: Config) -> None:
        interaction = cfg.require("task3.interaction")
        self.max_norm_dist = float(interaction["max_norm_dist"])
        self.facing_angle = float(interaction["facing_angle_deg"])
        self.require_mutual = bool(interaction.get("require_mutual", False))
        self.min_session_s = float(interaction["min_session_s"])
        self.flicker_s = float(interaction.get("flicker_tolerance_s", 1.0))
        self.end_gap_s = float(interaction["end_gap_s"])
        self.static_speed_norm = float(interaction.get("costationary_speed_norm", 0.22))
        self.costationary_dist = float(
            interaction.get("costationary_max_norm_dist", self.max_norm_dist * 0.8)
        )
        self.allow_costationary = bool(interaction.get("allow_costationary", True))
        self.min_staff_track_s = float(cfg.get_path("task3.staff.min_track_seconds", 2.0))

        self.trackers: Dict[Tuple[int, int], EpisodeTracker] = {}
        self.sessions_per_staff: Dict[int, int] = {}
        self.staff_seen: Dict[int, Tuple[float, float]] = {}  # id -> (first_t, last_t)
        self.active_links: List[ActiveLink] = []
        self.events: List[dict] = []
        self.session_log: List[dict] = []

    # ------------------------------------------------------------------ API #
    def update(self, t: float, tracks: List[TrackState], in_scene) -> None:
        staff = [s for s in tracks if s.role == "staff" and in_scene(s)]
        customers = [s for s in tracks if s.role != "staff" and in_scene(s)]

        for state in staff:
            first, _ = self.staff_seen.get(state.track_id, (t, t))
            self.staff_seen[state.track_id] = (first, t)
            self.sessions_per_staff.setdefault(state.track_id, 0)

        self.active_links = []
        live_pairs = set()

        for staff_state in staff:
            for customer_state in customers:
                key = (staff_state.track_id, customer_state.track_id)
                live_pairs.add(key)
                engaged, reason = self._engaged(staff_state, customer_state)

                tracker = self.trackers.get(key)
                if tracker is None:
                    tracker = EpisodeTracker(
                        min_duration_s=self.min_session_s,
                        flicker_tolerance_s=self.flicker_s,
                        min_gap_s=self.end_gap_s,
                        label=f"{key[0]}->{key[1]}",
                    )
                    self.trackers[key] = tracker

                episode = tracker.update(t, engaged, meta={"reason": reason})
                if episode is not None:
                    self.sessions_per_staff[staff_state.track_id] = (
                        self.sessions_per_staff.get(staff_state.track_id, 0) + 1
                    )
                    self.events.append(
                        {
                            "t": round(t, 2),
                            "event": "interaction_session",
                            "staff_id": staff_state.track_id,
                            "customer_id": customer_state.track_id,
                            "reason": reason,
                        }
                    )

                if tracker.is_active:
                    self.active_links.append(
                        ActiveLink(
                            staff_id=staff_state.track_id,
                            customer_id=customer_state.track_id,
                            started_at=tracker.active_since or t,
                            duration=tracker.active_duration(t),
                            reason=reason,
                        )
                    )

        for key, tracker in self.trackers.items():
            if key not in live_pairs:
                tracker.update(t, False)

    def finalize(self, t_end: float) -> None:
        for (staff_id, customer_id), tracker in self.trackers.items():
            tracker.finalize(t_end)
            for episode in tracker.episodes:
                self.session_log.append(
                    {
                        "staff_id": staff_id,
                        "customer_id": customer_id,
                        "start_s": round(episode.start_t, 2),
                        "end_s": round(episode.end_t, 2),
                        "duration_s": round(episode.duration, 2),
                        "bridged_gaps": episode.merges,
                    }
                )
        self.session_log.sort(key=lambda s: (s["staff_id"], s["start_s"]))

        # Drop staff instances that were only ever a flicker: they are almost
        # certainly a detection artefact, and including them would deflate the
        # average with a spurious zero.
        for staff_id, (first, last) in list(self.staff_seen.items()):
            if (last - first) < self.min_staff_track_s and self.sessions_per_staff.get(staff_id, 0) == 0:
                self.staff_seen.pop(staff_id, None)
                self.sessions_per_staff.pop(staff_id, None)

    # -------------------------------------------------------------- results #
    @property
    def staff_ids(self) -> List[int]:
        return sorted(self.staff_seen.keys())

    @property
    def total_sessions(self) -> int:
        return sum(self.sessions_per_staff.values())

    @property
    def average_sessions(self) -> float:
        n = len(self.staff_ids)
        return (self.total_sessions / n) if n else 0.0

    def summary(self) -> Dict[str, object]:
        return {
            "staff_instances": len(self.staff_ids),
            "total_sessions": self.total_sessions,
            "average_sessions_per_staff": round(self.average_sessions, 3),
        }

    def per_staff_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for index, staff_id in enumerate(self.staff_ids, start=1):
            first, last = self.staff_seen[staff_id]
            rows.append(
                {
                    "staff_instance": f"Staff {index}",
                    "track_id": staff_id,
                    "first_seen_s": round(first, 2),
                    "last_seen_s": round(last, 2),
                    "interaction_sessions": self.sessions_per_staff.get(staff_id, 0),
                }
            )
        return rows

    def display_index(self, staff_id: int) -> int:
        try:
            return self.staff_ids.index(staff_id) + 1
        except ValueError:
            return 0

    # ------------------------------------------------------------ internals #
    def _engaged(self, staff: TrackState, customer: TrackState) -> Tuple[bool, str]:
        scale = max((staff.height + customer.height) / 2.0, 1.0)
        dist_norm = math.dist(staff.ground, customer.ground) / scale
        if dist_norm > self.max_norm_dist:
            return False, ""

        to_customer = normalize(
            (customer.ground[0] - staff.ground[0], customer.ground[1] - staff.ground[1])
        )
        to_staff = (-to_customer[0], -to_customer[1])

        staff_faces = (
            staff.facing.is_valid
            and angle_between(staff.facing.vector, to_customer) <= self.facing_angle
        )
        customer_faces = (
            customer.facing.is_valid
            and angle_between(customer.facing.vector, to_staff) <= self.facing_angle
        )

        if self.require_mutual:
            if staff_faces and customer_faces:
                return True, "mutual-facing"
        elif staff_faces or customer_faces:
            return True, "facing"

        if (
            self.allow_costationary
            and dist_norm <= self.costationary_dist
            and staff.speed_norm <= self.static_speed_norm
            and customer.speed_norm <= self.static_speed_norm
        ):
            return True, "co-stationary"

        return False, ""
