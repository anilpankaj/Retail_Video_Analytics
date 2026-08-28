"""Turning a noisy per-frame boolean into countable events.

All three tasks reduce to the same question: *a condition is true on some
frames and false on others — how many distinct "things" happened?*  Detection
noise, brief occlusions and a person shifting their weight all make a raw
per-frame boolean flicker, so counting raw transitions would massively
over-count.

``EpisodeTracker`` implements the shared answer with three explicitly named
temporal parameters:

``min_duration_s``
    How long the condition must hold before we believe it at all.  Filters
    out someone merely walking past a shelf.

``flicker_tolerance_s``
    A gap shorter than this does **not** end the episode.  Covers a missed
    detection or the person turning away for a moment.

``min_gap_s``
    After an episode has ended, a *new* episode that starts within this
    window is merged back into the previous one rather than counted again.
    This is the "is this a continuation or a genuine return?" rule the brief
    asks us to define.  A return after a longer absence counts as a new
    event, which is exactly the requested behaviour.

The state machine is deliberately dependency-free and unit-tested against
synthetic sequences (see ``tests/test_events.py``), because every headline
number in the submission flows through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Episode:
    """One counted event: a continuous period during which the condition held."""

    start_t: float
    end_t: float
    #: number of raw sub-runs merged into this episode (diagnostic)
    merges: int = 0
    #: free-form payload, e.g. which shelf or which partner track
    label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_t - self.start_t)


class EpisodeTracker:
    """Hysteresis state machine converting a per-frame flag into episodes."""

    def __init__(
        self,
        min_duration_s: float,
        flicker_tolerance_s: float,
        min_gap_s: float,
        label: str = "",
    ) -> None:
        self.min_duration_s = float(min_duration_s)
        self.flicker_tolerance_s = float(flicker_tolerance_s)
        self.min_gap_s = float(min_gap_s)
        self.label = label

        self.episodes: List[Episode] = []
        self._run_start: Optional[float] = None  # start of the current candidate run
        self._last_active_t: Optional[float] = None
        self._open: Optional[Episode] = None  # confirmed, still-running episode

    # -- introspection ----------------------------------------------------- #
    @property
    def count(self) -> int:
        """Number of confirmed episodes, including one currently in progress."""
        return len(self.episodes) + (1 if self._open is not None else 0)

    @property
    def is_active(self) -> bool:
        return self._open is not None

    @property
    def active_since(self) -> Optional[float]:
        return self._open.start_t if self._open else None

    def active_duration(self, now: float) -> float:
        return 0.0 if self._open is None else max(0.0, now - self._open.start_t)

    # -- the state machine -------------------------------------------------- #
    def update(self, t: float, active: bool, meta: Optional[dict] = None) -> Optional[Episode]:
        """Advance the machine to time ``t``.

        Returns an :class:`Episode` on the frame where one is *confirmed*
        (i.e. where the count increments), otherwise ``None``.
        """
        confirmed: Optional[Episode] = None

        if active:
            if self._run_start is None:
                self._run_start = t
            self._last_active_t = t

            if self._open is None and (t - self._run_start) >= self.min_duration_s:
                # A candidate run has lasted long enough to be believed.
                episode = Episode(start_t=self._run_start, end_t=t, label=self.label,
                                  meta=dict(meta or {}))
                merged = self._try_merge(episode)
                if merged is None:
                    self._open = episode
                    confirmed = episode
                else:
                    self._open = merged
            elif self._open is not None:
                self._open.end_t = t
                if meta:
                    self._open.meta.update(meta)
        else:
            if self._last_active_t is not None and (t - self._last_active_t) > self.flicker_tolerance_s:
                self._close(self._last_active_t)
                self._run_start = None
                self._last_active_t = None
            elif self._last_active_t is None:
                self._run_start = None

        return confirmed

    def finalize(self, t: float) -> None:
        """Close any episode still open at the end of the video."""
        end = self._last_active_t if self._last_active_t is not None else t
        self._close(end)
        self._run_start = None
        self._last_active_t = None

    # -- internals ---------------------------------------------------------- #
    def _close(self, end_t: float) -> None:
        if self._open is not None:
            self._open.end_t = max(self._open.end_t, end_t)
            self.episodes.append(self._open)
            self._open = None

    def _try_merge(self, candidate: Episode) -> Optional[Episode]:
        """Re-open the previous episode if the gap was too short to be a return."""
        if not self.episodes:
            return None
        previous = self.episodes[-1]
        if (candidate.start_t - previous.end_t) < self.min_gap_s:
            self.episodes.pop()
            previous.end_t = candidate.end_t
            previous.merges += 1
            return previous
        return None


class Latch:
    """A sticky boolean that requires sustained evidence to switch on.

    Used for one-way decisions such as "this person showed interest" or
    "this track is a staff member": once enough evidence has accumulated the
    verdict never flips back, which keeps unique-person counts stable.
    """

    def __init__(self, min_duration_s: float, flicker_tolerance_s: float = 0.0) -> None:
        self.min_duration_s = float(min_duration_s)
        self.flicker_tolerance_s = float(flicker_tolerance_s)
        self._run_start: Optional[float] = None
        self._last_active_t: Optional[float] = None
        self.value = False
        self.triggered_at: Optional[float] = None

    def update(self, t: float, active: bool) -> bool:
        if self.value:
            return True
        if active:
            if self._run_start is None:
                self._run_start = t
            self._last_active_t = t
            if (t - self._run_start) >= self.min_duration_s:
                self.value = True
                self.triggered_at = t
        else:
            if self._last_active_t is None or (t - self._last_active_t) > self.flicker_tolerance_s:
                self._run_start = None
                self._last_active_t = None
        return self.value
