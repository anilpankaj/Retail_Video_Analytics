"""End-to-end behaviour of the three task state machines on synthetic tracks.

These are the tests that matter most: they encode, as executable
specifications, the definitions the brief asks the candidate to defend.
Each one describes a scenario in plain language and asserts the count the
methodology says it should produce.
"""

from __future__ import annotations

import pytest

from conftest import make_track, move
from rva.tasks.task1_interest import StoreInterestTask
from rva.tasks.task2_shelf import ShelfInterestTask
from rva.tasks.task3_staff import StaffInteractionTask


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def walk(task, track, points, facing, speed_norm, t0=0.0, dt=0.1):
    """Move a track through a list of ground points, one per timestep."""
    t = t0
    for point in points:
        move(track, point, t, facing=facing, speed_norm=speed_norm)
        task.update(t, [track])
        t += dt
    return t


def hold(task, track, point, facing, seconds, t0=0.0, dt=0.1, speed_norm=0.0):
    steps = int(round(seconds / dt))
    return walk(task, track, [point] * steps, facing, speed_norm, t0, dt)


# --------------------------------------------------------------------------- #
# Task 1
# --------------------------------------------------------------------------- #
def test_disinterested_passer_by_is_not_counted(entrance_cfg) -> None:
    """Someone striding across the walkway looking straight ahead."""
    task = StoreInterestTask(entrance_cfg)
    track = make_track(1, (500, 250), height=100, facing=(1.0, 0.0), speed_norm=1.4)
    points = [(500 + i * 8, 250) for i in range(60)]
    t = walk(task, track, points, facing=(1.0, 0.0), speed_norm=1.4)
    task.finalize(t)
    assert task.total_interested == 0


def test_slowing_and_turning_towards_the_store_counts_as_interest(entrance_cfg) -> None:
    task = StoreInterestTask(entrance_cfg)
    track = make_track(2, (700, 330), height=110, facing=(0.0, 1.0), speed_norm=0.1)
    t = hold(task, track, (700, 330), facing=(0.0, 1.0), seconds=3.0)
    task.finalize(t)
    assert task.total_interested == 1


def test_interest_requires_sustained_evidence(entrance_cfg) -> None:
    """A single frame of favourable geometry must not mint a customer."""
    task = StoreInterestTask(entrance_cfg)
    track = make_track(3, (700, 330), height=110, facing=(0.0, 1.0), speed_norm=0.1)
    t = hold(task, track, (700, 330), facing=(0.0, 1.0), seconds=0.3)
    task.finalize(t)
    assert task.total_interested == 0


def test_far_side_of_the_walkway_is_gated_out(entrance_cfg) -> None:
    """Attention has to be plausible: a small figure far from the frontage."""
    task = StoreInterestTask(entrance_cfg)
    # 60 px tall at 2.5 body-heights from the storefront line
    track = make_track(4, (700, 230), height=60, facing=(0.0, 1.0), speed_norm=0.05)
    t = hold(task, track, (700, 230), facing=(0.0, 1.0), seconds=4.0)
    task.finalize(t)
    assert task.total_interested == 0


def test_interested_person_who_walks_in_is_counted_as_entered(entrance_cfg) -> None:
    task = StoreInterestTask(entrance_cfg)
    track = make_track(5, (700, 340), height=130, facing=(0.0, 1.0), speed_norm=0.2)
    t = hold(task, track, (700, 340), facing=(0.0, 1.0), seconds=2.0)
    # walk deep into the store and stay there
    t = hold(task, track, (700, 600), facing=(0.0, 1.0), seconds=3.0, t0=t)
    task.finalize(t)
    assert task.total_interested == 1
    assert task.interested_entered == 1
    assert task.interested_passed_by == 0


def test_leaning_in_and_stepping_back_is_not_an_entry(entrance_cfg) -> None:
    """The depth + dwell requirement rejects a browse at the threshold."""
    task = StoreInterestTask(entrance_cfg)
    track = make_track(6, (700, 340), height=130, facing=(0.0, 1.0), speed_norm=0.2)
    t = hold(task, track, (700, 340), facing=(0.0, 1.0), seconds=2.0)
    t = hold(task, track, (700, 445), facing=(0.0, 1.0), seconds=0.6, t0=t)  # a toe over the line
    t = hold(task, track, (700, 330), facing=(0.0, 1.0), seconds=1.0, t0=t)
    t = hold(task, track, (250, 250), facing=(-1.0, 0.0), seconds=2.0, t0=t)  # leaves the scene
    task.finalize(t)
    assert task.interested_entered == 0
    assert task.interested_passed_by == 1


def test_interested_person_who_walks_on_is_passed_by(entrance_cfg) -> None:
    task = StoreInterestTask(entrance_cfg)
    track = make_track(7, (700, 340), height=120, facing=(0.0, 1.0), speed_norm=0.2)
    t = hold(task, track, (700, 340), facing=(0.0, 1.0), seconds=2.0)
    t = walk(task, track, [(700 + i * 20, 300) for i in range(30)],
             facing=(1.0, 0.0), speed_norm=1.3, t0=t)
    task.finalize(t)
    assert task.total_interested == 1
    assert task.interested_passed_by == 1


def test_counts_are_internally_consistent(entrance_cfg) -> None:
    """entered + passed_by == total_interested, always."""
    task = StoreInterestTask(entrance_cfg)
    t = 0.0
    for i, (x, y) in enumerate([(650, 340), (760, 350), (900, 380)]):
        track = make_track(20 + i, (x, y), height=125, facing=(0.0, 1.0), speed_norm=0.1)
        t = hold(task, track, (x, y), facing=(0.0, 1.0), seconds=2.5, t0=t)
    task.finalize(t)
    summary = task.summary()
    assert summary["interested_entered"] + summary["interested_passed_by"] == summary["total_interested"]


def test_someone_already_inside_the_store_is_not_a_passer_by(entrance_cfg) -> None:
    """Task 1 counts passers-by converting, not customers already shopping."""
    task = StoreInterestTask(entrance_cfg)
    track = make_track(30, (600, 620), height=200, facing=(0.0, -1.0), speed_norm=0.05)
    t = hold(task, track, (600, 620), facing=(0.0, -1.0), seconds=8.0)
    task.finalize(t)
    assert task.total_interested == 0
    assert task.outcomes[30].origin == "store"


def test_walkway_origin_is_remembered_after_entering(entrance_cfg) -> None:
    """A genuine passer-by keeps their candidacy once they step inside."""
    task = StoreInterestTask(entrance_cfg)
    track = make_track(31, (700, 340), height=130, facing=(0.0, 1.0), speed_norm=0.2)
    t = hold(task, track, (700, 340), facing=(0.0, 1.0), seconds=2.0)
    t = hold(task, track, (700, 620), facing=(0.0, 1.0), seconds=3.0, t0=t)
    task.finalize(t)
    assert task.outcomes[31].origin == "walkway"
    assert task.interested_entered == 1


def test_staff_are_excluded_from_task1(entrance_cfg) -> None:
    task = StoreInterestTask(entrance_cfg)
    track = make_track(8, (700, 340), height=130, facing=(0.0, 1.0), speed_norm=0.0, role="staff")
    t = hold(task, track, (700, 340), facing=(0.0, 1.0), seconds=6.0)
    task.finalize(t)
    assert task.total_interested == 0


def test_people_on_another_mall_level_are_ignored(entrance_cfg) -> None:
    task = StoreInterestTask(entrance_cfg)
    track = make_track(9, (800, 120), height=40, facing=(0.0, 1.0), speed_norm=0.0)
    assert task.in_scene(track) is False


# --------------------------------------------------------------------------- #
# Task 2
# --------------------------------------------------------------------------- #
def test_walking_past_a_shelf_is_not_an_interest_event(interior_cfg) -> None:
    task = ShelfInterestTask(interior_cfg)
    track = make_track(1, (380, 500), height=200, facing=(1.0, 0.0), speed_norm=1.2)
    points = [(380 + i * 6, 500) for i in range(15)]  # 1.5 s of transit
    t = walk(task, track, points, facing=(1.0, 0.0), speed_norm=1.2)
    task.finalize(t)
    assert sum(task.summary().values()) == 0


def test_sustained_attention_creates_one_event(interior_cfg) -> None:
    task = ShelfInterestTask(interior_cfg)
    # standing just left of shelf B, facing it
    track = make_track(2, (400, 520), height=200, facing=(1.0, 0.0), speed_norm=0.05)
    t = hold(task, track, (400, 520), facing=(1.0, 0.0), seconds=12.0)
    task.finalize(t)
    assert task.summary()["B"] == 1


def test_long_continuous_visit_is_still_one_event(interior_cfg) -> None:
    """The explicit no-double-counting requirement."""
    task = ShelfInterestTask(interior_cfg)
    track = make_track(3, (400, 520), height=200, facing=(1.0, 0.0), speed_norm=0.05)
    t = hold(task, track, (400, 520), facing=(1.0, 0.0), seconds=45.0)
    task.finalize(t)
    assert task.summary()["B"] == 1


def test_return_after_a_long_absence_is_a_second_event(interior_cfg) -> None:
    task = ShelfInterestTask(interior_cfg)
    track = make_track(4, (400, 520), height=200, facing=(1.0, 0.0), speed_norm=0.05)
    t = hold(task, track, (400, 520), facing=(1.0, 0.0), seconds=6.0)
    # go elsewhere for 10 s (well beyond min_gap_between_events_s)
    t = hold(task, track, (150, 300), facing=(0.0, -1.0), seconds=10.0, t0=t)
    t = hold(task, track, (400, 520), facing=(1.0, 0.0), seconds=6.0, t0=t)
    task.finalize(t)
    assert task.summary()["B"] == 2


def test_brief_turn_away_does_not_split_the_episode(interior_cfg) -> None:
    task = ShelfInterestTask(interior_cfg)
    track = make_track(5, (400, 520), height=200, facing=(1.0, 0.0), speed_norm=0.05)
    t = hold(task, track, (400, 520), facing=(1.0, 0.0), seconds=5.0)
    t = hold(task, track, (400, 520), facing=(-1.0, 0.0), seconds=1.0, t0=t)  # glances away
    t = hold(task, track, (400, 520), facing=(1.0, 0.0), seconds=5.0, t0=t)
    task.finalize(t)
    assert task.summary()["B"] == 1


def test_assignment_is_exclusive_between_adjacent_shelves(interior_cfg) -> None:
    """A customer in the aisle counts for exactly one shelf, never both."""
    task = ShelfInterestTask(interior_cfg)
    # between shelf B (left) and shelf A (right), clearly facing A
    track = make_track(6, (860, 560), height=190, facing=(0.4, -0.9), speed_norm=0.05)
    t = hold(task, track, (860, 560), facing=(0.4, -0.9), seconds=10.0)
    task.finalize(t)
    counts = task.summary()
    assert sum(counts.values()) <= 1, "the same attention must not be credited twice"


def test_reach_direction_breaks_the_tie(interior_cfg) -> None:
    """Reaching towards a shelf is the strongest evidence of engagement."""
    task = ShelfInterestTask(interior_cfg)
    track = make_track(7, (860, 560), height=190, facing=(0.0, -1.0),
                       speed_norm=0.05, reach=(0.9, -0.4))
    t = hold(task, track, (860, 560), facing=(0.0, -1.0), seconds=10.0)
    task.finalize(t)
    counts = task.summary()
    assert sum(counts.values()) == 1


def test_customer_far_from_every_shelf_scores_nothing(interior_cfg) -> None:
    task = ShelfInterestTask(interior_cfg)
    track = make_track(8, (1150, 690), height=60, facing=(0.0, -1.0), speed_norm=0.0)
    t = hold(task, track, (1150, 690), facing=(0.0, -1.0), seconds=10.0)
    task.finalize(t)
    assert sum(task.summary().values()) == 0


def test_background_customers_are_outside_the_roi(interior_cfg) -> None:
    task = ShelfInterestTask(interior_cfg)
    track = make_track(9, (600, 120), height=45, facing=(0.0, 1.0))
    assert task.in_scene(track) is False


def test_staff_exclusion_mechanism_works_when_enabled(interior_cfg) -> None:
    """The shipped interior config leaves this OFF (see the config comment:
    the apron signal is not reliable in that camera), but the mechanism itself
    must work, because the entrance pipeline depends on it."""
    cfg = dict(interior_cfg)
    cfg["task2"] = {**cfg["task2"], "exclude_staff": True}
    task = ShelfInterestTask(type(interior_cfg)(cfg))
    track = make_track(10, (400, 520), height=200, facing=(1.0, 0.0),
                       speed_norm=0.05, role="staff")
    t = hold(task, track, (400, 520), facing=(1.0, 0.0), seconds=20.0)
    task.finalize(t)
    assert sum(task.summary().values()) == 0


def test_shipped_interior_config_treats_everyone_as_a_customer(interior_cfg) -> None:
    """Documented decision: no reliable apron evidence in the interior view."""
    assert interior_cfg.get_path("task2.exclude_staff") is False


# --------------------------------------------------------------------------- #
# Task 3
# --------------------------------------------------------------------------- #
def _always_in_scene(_state) -> bool:
    return True


def pair_update(task, staff, customer, t, dt=0.1, seconds=1.0):
    steps = int(round(seconds / dt))
    for _ in range(steps):
        task.update(t, [staff, customer], _always_in_scene)
        t += dt
    return t


def test_passing_a_staff_member_is_not_an_interaction(entrance_cfg) -> None:
    task = StaffInteractionTask(entrance_cfg)
    staff = make_track(1, (600, 600), height=200, facing=(1.0, 0.0), role="staff")
    customer = make_track(2, (700, 600), height=200, facing=(1.0, 0.0), speed_norm=1.5)
    t = pair_update(task, staff, customer, 0.0, seconds=1.0)  # under min_session_s
    task.finalize(t)
    assert task.total_sessions == 0


def test_sustained_engagement_is_one_session(entrance_cfg) -> None:
    task = StaffInteractionTask(entrance_cfg)
    staff = make_track(1, (600, 600), height=200, facing=(1.0, 0.0), role="staff")
    customer = make_track(2, (700, 600), height=200, facing=(-1.0, 0.0), speed_norm=0.05)
    t = pair_update(task, staff, customer, 0.0, seconds=20.0)
    task.finalize(t)
    assert task.total_sessions == 1


def test_reengagement_after_a_break_is_a_second_session(entrance_cfg) -> None:
    task = StaffInteractionTask(entrance_cfg)
    staff = make_track(1, (600, 600), height=200, facing=(1.0, 0.0), role="staff")
    customer = make_track(2, (700, 600), height=200, facing=(-1.0, 0.0), speed_norm=0.05)
    t = pair_update(task, staff, customer, 0.0, seconds=6.0)
    move(customer, (1150, 600), t, facing=(1.0, 0.0), speed_norm=1.0)
    t = pair_update(task, staff, customer, t, seconds=8.0)   # far apart
    move(customer, (700, 600), t, facing=(-1.0, 0.0), speed_norm=0.05)
    t = pair_update(task, staff, customer, t, seconds=6.0)
    task.finalize(t)
    assert task.total_sessions == 2


def test_one_staff_serving_two_customers_accrues_two_sessions(entrance_cfg) -> None:
    task = StaffInteractionTask(entrance_cfg)
    staff = make_track(1, (600, 600), height=200, facing=(1.0, 0.0), role="staff")
    a = make_track(2, (700, 600), height=200, facing=(-1.0, 0.0), speed_norm=0.05)
    b = make_track(3, (520, 600), height=200, facing=(1.0, 0.0), speed_norm=0.05)
    t = 0.0
    for _ in range(200):
        task.update(t, [staff, a, b], _always_in_scene)
        t += 0.1
    task.finalize(t)
    assert task.sessions_per_staff[1] == 2


def test_costationary_fallback_catches_the_kneeling_fitting(entrance_cfg) -> None:
    """Pose is unreliable at the frame edge; proximity + stillness must carry."""
    task = StaffInteractionTask(entrance_cfg)
    staff = make_track(1, (600, 700), height=180, facing=None, role="staff")
    customer = make_track(2, (660, 700), height=180, facing=None, speed_norm=0.02)
    staff.speed_norm = 0.02
    t = pair_update(task, staff, customer, 0.0, seconds=10.0)
    task.finalize(t)
    assert task.total_sessions == 1


def test_zero_interaction_staff_are_included_in_the_average(entrance_cfg) -> None:
    """The brief's worked example: (5 + 4 + 0) / 3 = 3.0."""
    task = StaffInteractionTask(entrance_cfg)
    task.staff_seen = {1: (0.0, 60.0), 2: (0.0, 60.0), 3: (0.0, 60.0)}
    task.sessions_per_staff = {1: 5, 2: 4, 3: 0}
    assert task.average_sessions == pytest.approx(3.0)
    assert task.summary()["staff_instances"] == 3


def test_average_is_zero_when_no_staff_are_detected(entrance_cfg) -> None:
    task = StaffInteractionTask(entrance_cfg)
    assert task.average_sessions == 0.0


def test_flicker_staff_instances_are_dropped(entrance_cfg) -> None:
    """A sub-2 s staff 'instance' with no sessions is a detection artefact."""
    task = StaffInteractionTask(entrance_cfg)
    task.staff_seen = {1: (0.0, 60.0), 2: (10.0, 10.4)}
    task.sessions_per_staff = {1: 3, 2: 0}
    task.finalize(60.0)
    assert task.staff_ids == [1]
    assert task.average_sessions == pytest.approx(3.0)
