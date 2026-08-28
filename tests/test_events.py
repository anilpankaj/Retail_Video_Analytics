"""The episode state machine - the component every headline number flows through."""

from __future__ import annotations

import pytest

from rva.core.events import EpisodeTracker, Latch


def feed(tracker: EpisodeTracker, timeline, dt: float = 0.1):
    """Drive the tracker with a list of (duration_s, active) segments."""
    t = 0.0
    confirmations = []
    for duration, active in timeline:
        steps = int(round(duration / dt))
        for _ in range(steps):
            episode = tracker.update(t, active)
            if episode is not None:
                confirmations.append(round(t, 2))
            t += dt
    tracker.finalize(t)
    return confirmations, t


def make(min_duration=2.0, flicker=1.5, gap=6.0) -> EpisodeTracker:
    return EpisodeTracker(min_duration_s=min_duration, flicker_tolerance_s=flicker, min_gap_s=gap)


def test_brief_touch_is_not_an_event() -> None:
    """Walking past a shelf must not count."""
    tracker = make()
    feed(tracker, [(1.0, False), (1.0, True), (3.0, False)])
    assert tracker.count == 0


def test_sustained_attention_is_one_event() -> None:
    tracker = make()
    feed(tracker, [(1.0, False), (10.0, True), (3.0, False)])
    assert tracker.count == 1
    assert tracker.episodes[0].duration == pytest.approx(9.9, abs=0.3)


def test_continuous_interaction_is_not_counted_repeatedly() -> None:
    """The explicit double-counting requirement in the brief."""
    tracker = make()
    # 30 seconds of continuous engagement broken only by detection flicker
    timeline = [(1.0, False)]
    for _ in range(10):
        timeline += [(2.5, True), (0.4, False)]
    timeline += [(4.0, False)]
    feed(tracker, timeline)
    assert tracker.count == 1, "flicker inside an episode must not create new events"


def test_short_gap_is_bridged_not_split() -> None:
    tracker = make(min_duration=2.0, flicker=1.5, gap=6.0)
    feed(tracker, [(4.0, True), (1.0, False), (4.0, True), (8.0, False)])
    assert tracker.count == 1


def test_return_after_long_absence_is_a_new_event() -> None:
    """A customer coming back to the same shelf later must count again."""
    tracker = make(min_duration=2.0, flicker=1.5, gap=6.0)
    feed(tracker, [(4.0, True), (10.0, False), (4.0, True), (8.0, False)])
    assert tracker.count == 2


def test_gap_just_under_threshold_merges() -> None:
    tracker = make(min_duration=2.0, flicker=1.0, gap=6.0)
    # gap of ~4 s: longer than flicker tolerance so the episode closes, but
    # shorter than min_gap so the return is merged back in.
    feed(tracker, [(3.0, True), (4.0, False), (3.0, True), (8.0, False)])
    assert tracker.count == 1
    assert tracker.episodes[0].merges == 1


def test_episode_open_at_end_of_video_is_still_counted() -> None:
    tracker = make()
    feed(tracker, [(1.0, False), (10.0, True)])
    assert tracker.count == 1
    assert len(tracker.episodes) == 1


def test_active_duration_reports_live_elapsed_time() -> None:
    tracker = make(min_duration=1.0, flicker=1.0, gap=5.0)
    t = 0.0
    for _ in range(50):  # 5 s of engagement
        tracker.update(t, True)
        t += 0.1
    assert tracker.is_active
    assert tracker.active_duration(t) == pytest.approx(4.9, abs=0.2)


def test_count_includes_the_in_progress_episode() -> None:
    tracker = make(min_duration=1.0)
    t = 0.0
    for _ in range(30):
        tracker.update(t, True)
        t += 0.1
    assert tracker.count == 1


# --------------------------------------------------------------------------- #
# Latch
# --------------------------------------------------------------------------- #
def test_latch_requires_sustained_evidence() -> None:
    latch = Latch(min_duration_s=0.8, flicker_tolerance_s=0.5)
    t = 0.0
    for _ in range(5):  # 0.5 s - not enough
        latch.update(t, True)
        t += 0.1
    assert latch.value is False
    for _ in range(5):
        latch.update(t, True)
        t += 0.1
    assert latch.value is True


def test_latch_never_flips_back() -> None:
    latch = Latch(min_duration_s=0.2)
    t = 0.0
    for _ in range(5):
        latch.update(t, True)
        t += 0.1
    assert latch.value is True
    for _ in range(50):
        latch.update(t, False)
        t += 0.1
    assert latch.value is True, "unique-person counts must never decrease"


def test_latch_resets_candidate_run_on_a_long_gap() -> None:
    latch = Latch(min_duration_s=1.0, flicker_tolerance_s=0.2)
    t = 0.0
    for _ in range(5):
        latch.update(t, True)
        t += 0.1
    for _ in range(10):
        latch.update(t, False)
        t += 0.1
    for _ in range(5):
        latch.update(t, True)
        t += 0.1
    assert latch.value is False, "evidence must be contiguous, not cumulative"
