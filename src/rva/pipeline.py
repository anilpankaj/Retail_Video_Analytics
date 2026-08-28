"""The two runnable pipelines: ``entrance`` (Tasks 1 + 3) and ``interior`` (Task 2).

Why two passes
--------------
The obvious design is one loop: detect, decide, count, draw. It is wrong here,
because whether a track is *staff* can only be judged from the whole clip — the
badge detection rate and the apron average are least settled precisely in a
track's first few seconds, which is when a single-pass design would have to
commit. A first attempt did exactly that and produced a verdict that changed
mid-clip, leaving the on-screen counters, the CSVs and the audit trail
disagreeing about who was who. The brief requires the CSV values to match the
annotated video, so that is a correctness failure, not a cosmetic one.

So:

**Pass 1 — inference.** Detect, track, and measure the two per-frame appearance
numbers the staff classifier needs. Write them all to a feature cache. Produce
no counts. This is the only pass that runs the model, and it dominates runtime.

*Then* the staff/customer verdict is taken once, per track, from the complete
evidence.

**Pass 2 — metrics and annotation.** Replay the cache through the task state
machines with the final roles attached, and draw the annotated video from the
same objects that produce the CSVs. The video is decoded a second time, which
costs a few minutes against the ~50 that inference costs.

The cache that makes this possible is also exposed to the reviewer: any
threshold can be re-tuned and every count recomputed in seconds by replaying it
(``rva.cli replay``), with no second inference pass.

Analysis runs on every ``frame_stride``-th frame while the annotated video is
written at the full source frame rate, holding the most recent state in between.
At the default stride of 3 on 30 fps footage that state is at most 66 ms stale —
invisible — and every temporal threshold is expressed in seconds, so the stride
can never change a count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config, as_points
from .core import report, viz
from .core.geometry import Polygon, bbox_center
from .core.replay import FeatureWriter, ReplayTracks, read_features
from .core.staff import StaffClassifier
from .core.tracking import TrackLinker, TrackManager, TrackState
from .core.video import VideoReader, VideoWriter, format_timecode
from .tasks.task1_interest import ENTERED, INTERESTED, NEUTRAL, PASSED_BY, StoreInterestTask
from .tasks.task2_shelf import ShelfInterestTask
from .tasks.task3_staff import StaffInteractionTask


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _build_tracker(cfg: Config) -> TrackManager:
    model = cfg.require("model")
    runtime = cfg.require("runtime")
    link_cfg = cfg.get_path("runtime.fragment_linker", {}) or {}
    linker = TrackLinker(
        max_gap_s=float(link_cfg.get("max_gap_s", 2.5)),
        max_dist_norm=float(link_cfg.get("max_dist_norm", 2.0)),
        min_similarity=float(link_cfg.get("min_similarity", 0.55)),
        enabled=bool(link_cfg.get("enabled", True)),
        link_within_obs=int(link_cfg.get("link_within_obs", 3)),
    )
    return TrackManager(
        weights=model["weights"],
        imgsz=model.get("imgsz", 960),
        conf=model.get("conf", 0.3),
        iou=model.get("iou", 0.55),
        device=model.get("device", "cpu"),
        tracker_cfg=model.get("tracker", "botsort.yaml"),
        smoothing_window=runtime.get("smoothing_window", 7),
        walk_speed_norm=runtime.get("walk_speed_norm", 0.9),
        motion_bias=runtime.get("motion_bias", 0.65),
        kp_conf_min=runtime.get("kp_conf_min", 0.35),
        half=bool(model.get("half", False)),
        linker=linker,
        retire_after_s=float(link_cfg.get("retire_after_s", 0.5)),
    )


def _progress(prefix: str, idx: int, total: int, started: float, every: int = 300) -> None:
    if total <= 0 or idx % every:
        return
    done = idx + 1
    elapsed = time.time() - started
    rate = done / max(elapsed, 1e-6)
    eta = (total - done) / max(rate, 1e-6)
    pct = 100.0 * done / total
    print(f"  {prefix} {done}/{total} ({pct:5.1f}%)  {rate:5.1f} fps  eta {eta/60:5.1f} min", flush=True)


def _staff_classifier(staff_cfg: dict, residency_default: float, min_obs_default: int) -> StaffClassifier:
    """Build the staff classifier from a config block (shared by run and replay)."""
    return StaffClassifier(
        dark_v_max=staff_cfg.get("dark_v_max", 95),
        dark_s_max=staff_cfg.get("dark_s_max", 120),
        badge_v_min=staff_cfg.get("badge_v_min", 140),
        badge_min_fraction=staff_cfg.get("badge_min_fraction", 0.005),
        badge_max_fraction=staff_cfg.get("badge_max_fraction", 0.06),
        badge_min_aspect=staff_cfg.get("badge_min_aspect", 0.60),
        badge_min_fill=staff_cfg.get("badge_min_fill", 0.42),
        badge_requires_apron=staff_cfg.get("badge_requires_apron", 0.40),
        badge_rate_target=staff_cfg.get("badge_rate_target", 0.25),
        min_badge_height_px=staff_cfg.get("min_badge_height_px", 110),
        residency_target_s=staff_cfg.get("residency_target_s", residency_default),
        min_obs=staff_cfg.get("min_obs", min_obs_default),
        weights=staff_cfg.get("score_weights"),
        score_threshold=staff_cfg.get("score_threshold", 0.62),
        force_staff_ids=staff_cfg.get("force_staff_ids"),
        force_customer_ids=staff_cfg.get("force_customer_ids"),
    )


@dataclass
class RunResult:
    video: Path
    csvs: List[Path]
    summary: Dict[str, object]


# --------------------------------------------------------------------------- #
# entrance: Task 1 + Task 3 in one annotated video (as the brief requires)
# --------------------------------------------------------------------------- #
def _evidence_from_cache(cfg: Config, cache_path: Path, classifier: StaffClassifier,
                         inside_store) -> None:
    """Rebuild the staff evidence from an existing cache (no model, seconds)."""
    tracks = _replay_tracks(cfg)
    prev_t = None
    for record in read_features(cache_path):
        t = float(record["t"])
        visible, appearance = tracks.step(record)
        dt = (t - prev_t) if prev_t is not None else 0.1
        prev_t = t
        for state in visible:
            classifier.observe_features(
                state, appearance.get(state.track_id), dt, inside_store=inside_store(state)
            )


def _pass_one(
    cfg: Config,
    video_path: str,
    outputs: Path,
    classifier: StaffClassifier,
    inside_store,
    start_s: float,
    max_seconds: Optional[float],
    dump_features: Optional[str],
    label: str,
    reuse_cache: bool = False,
) -> Path:
    """Inference pass: detect, track, measure appearance, write the cache.

    This is the only pass that runs the model. It produces no counts, because
    the staff/customer verdict it feeds is only trustworthy once the whole clip
    has been seen (see the module docstring).

    With ``reuse_cache`` and an existing cache the model is skipped entirely and
    the evidence is rebuilt from the cache instead - which is how the annotated
    video can be re-rendered after a threshold change in minutes rather than the
    best part of an hour.
    """
    cache_path = Path(dump_features) if dump_features else outputs / "cache" / f"{label}_features.jsonl"
    if reuse_cache and cache_path.exists():
        print(f"  reusing feature cache {cache_path} (skipping inference)", flush=True)
        _evidence_from_cache(cfg, cache_path, classifier, inside_store)
        return cache_path

    tracker = _build_tracker(cfg)
    stride = int(cfg.get_path("runtime.frame_stride", 3))
    kp_conf_min = float(cfg.get_path("runtime.kp_conf_min", 0.35))

    cache = FeatureWriter(cache_path)

    reader = VideoReader(video_path, start_s=start_s, max_seconds=max_seconds)
    info = reader.info
    total = info.frame_count if max_seconds is None else int(max_seconds * info.fps)
    started = time.time()
    prev_infer_t: Optional[float] = None

    for frame_idx, t, frame in reader:
        if (frame_idx - reader.start_frame) % stride:
            continue
        visible = tracker.update(frame, t, frame_idx)
        dt = (t - prev_infer_t) if prev_infer_t is not None else (stride / info.fps)
        prev_infer_t = t

        appearance = {}
        for state in visible:
            measured = classifier.measure(state, frame, kp_conf_min)
            appearance[state.track_id] = measured
            classifier.observe_features(state, measured, dt, inside_store=inside_store(state))
        cache.write(t, frame_idx, visible, appearance)
        _progress(f"{label}:detect", frame_idx - reader.start_frame, total, started)

    reader.close()
    cache.close()
    print(f"  feature cache -> {cache.path} ({cache.frames} frames)", flush=True)
    return cache_path


def run_entrance(
    cfg: Config,
    video_path: str,
    outputs: Path,
    start_s: float = 0.0,
    max_seconds: Optional[float] = None,
    show_zones: bool = True,
    write_video: bool = True,
    dump_features: Optional[str] = None,
    reuse_cache: bool = False,
) -> RunResult:
    outputs.mkdir(parents=True, exist_ok=True)
    classifier = _staff_classifier(cfg.require("task3.staff"),
                                   residency_default=30.0, min_obs_default=60)
    store = Polygon(as_points(cfg.require("scene")["store_polygon"]), "store")

    # ---- pass 1: inference. Populates the feature cache and accumulates the
    # staff evidence over the WHOLE clip.
    cache_path = _pass_one(
        cfg, video_path, outputs, classifier,
        inside_store=lambda state: store.contains(state.ground),
        start_s=start_s, max_seconds=max_seconds,
        dump_features=dump_features, label="entrance", reuse_cache=reuse_cache,
    )
    classifier.finalize()

    # ---- pass 2: metrics + annotation, replayed from the cache with the FINAL
    # roles. See the module docstring for why this is two passes.
    task1 = StoreInterestTask(cfg)
    task3 = StaffInteractionTask(cfg)
    kp_conf_min = float(cfg.get_path("runtime.kp_conf_min", 0.35))
    draw_pose = bool(cfg.get_path("output.draw_keypoints", True))

    reader = VideoReader(video_path, start_s=start_s, max_seconds=max_seconds)
    info = reader.info
    out_video = outputs / cfg.get_path("output.entrance_video", "entrance_annotated.mp4")
    writer = VideoWriter(out_video, info.fps, cfg.get_path("output.codec", "mp4v")) if write_video else None

    total = info.frame_count if max_seconds is None else int(max_seconds * info.fps)
    started = time.time()
    tracks = _replay_tracks(cfg)
    records = read_features(cache_path)
    pending = next(records, None)
    last_visible: List[TrackState] = []
    last_t = 0.0

    for frame_idx, t, frame in reader:
        last_t = t
        if pending is not None and int(pending["f"]) == frame_idx:
            visible, _ = tracks.step(pending)
            for state in visible:
                state.role = classifier.role_of(state.track_id)
                state.role_score = classifier.score_of(state.track_id)
            task1.update(t, visible)
            task3.update(t, visible, in_scene=task1.in_scene)
            last_visible = visible
            pending = next(records, None)

        if writer is not None:
            canvas = frame.copy()
            _render_entrance(canvas, t, last_visible, task1, task3, show_zones, draw_pose, kp_conf_min)
            writer.write(canvas)

        _progress("entrance:annotate", frame_idx - reader.start_frame, total, started)

    reader.close()
    if writer is not None:
        writer.close()

    task1.finalize(last_t)
    task3.finalize(last_t)

    summary = {"task1": task1.summary(), "task3": task3.summary()}
    csvs = report.write_task1(outputs, task1.summary(), task1.per_person_rows(), task1.events)
    csvs += report.write_task3(
        outputs, task3.per_staff_rows(), task3.summary(), task3.session_log,
        classifier.report_rows(), task3.events,
    )

    # Consistency invariant asserted before anything is reported.
    s = task1.summary()
    assert s["interested_entered"] + s["interested_passed_by"] == s["total_interested"], (
        "Task 1 invariant violated: entered + passed_by != total_interested"
    )

    report.write_json(
        outputs / "audit" / "entrance_summary.json",
        {
            "video": str(video_path),
            "frames_processed": total,
            "fps": info.fps,
            "task1": s,
            "task3": task3.summary(),
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        },
    )
    return RunResult(video=out_video, csvs=csvs, summary=summary)


def _render_entrance(
    canvas: np.ndarray,
    t: float,
    visible: List[TrackState],
    task1: StoreInterestTask,
    task3: StaffInteractionTask,
    show_zones: bool,
    draw_pose: bool,
    kp_conf_min: float,
) -> None:
    if show_zones:
        viz.draw_polygon(canvas, task1.walkway, viz.COL_WALKWAY, "", fill_alpha=0.06, thickness=1)
        viz.draw_polygon(canvas, task1.store, viz.COL_STORE, "", fill_alpha=0.06, thickness=1)
        viz.draw_polyline(canvas, task1.front, viz.COL_FRONT, 3)
        viz.text(canvas, "ENTRANCE BOUNDARY", (int(task1.front.points[-1][0]) - 230,
                                               int(task1.front.points[-1][1]) - 8),
                 viz.COL_FRONT, 0.5, 1, viz.FONT_S)

    positions: Dict[int, Tuple[float, float]] = {}
    for state in visible:
        positions[state.track_id] = bbox_center(state.box)
        is_staff = state.role == "staff"
        state_label, score = task1.state_of(state.track_id)

        if is_staff:
            color = viz.COL_STAFF
            index = task3.display_index(state.track_id)
            sessions = task3.sessions_per_staff.get(state.track_id, 0)
            label = f"STAFF {index} | sessions {sessions}"
            sub = f"apron score {state.role_score:.2f}"
        else:
            color = {
                ENTERED: viz.COL_ENTERED,
                PASSED_BY: viz.COL_PASSED,
                INTERESTED: viz.COL_INTEREST,
            }.get(state_label, viz.COL_CUSTOMER)
            label = f"ID {state.track_id}" + (f" | {state_label}" if state_label != NEUTRAL else "")
            sub = f"interest {score:.2f}"

        if draw_pose:
            viz.draw_keypoints(canvas, state.keypoints, color, kp_conf_min)
        viz.draw_person(canvas, state.box, color, label, sub)
        if state.facing.is_valid:
            viz.draw_facing(canvas, state.ground, state.facing.vector, color, 46)

    for link in task3.active_links:
        a = positions.get(link.staff_id)
        b = positions.get(link.customer_id)
        if a and b:
            viz.draw_link(canvas, a, b, viz.COL_LINK, f"{link.duration:4.1f}s {link.reason}")

    s = task1.summary()
    rows = [
        viz.PanelRow("Total Interested", str(s["total_interested"]), viz.COL_INTEREST, True),
        viz.PanelRow("Interested Entered", str(s["interested_entered"]), viz.COL_ENTERED, True),
        viz.PanelRow("Interested Passed By", str(s["interested_passed_by"]), viz.COL_PASSED, True),
    ]
    viz.draw_panel(canvas, "TASK 1 - STORE INTEREST & CONVERSION", rows, origin=(14, 52), width=352)

    t3 = task3.summary()
    staff_rows = [
        viz.PanelRow(f"Staff {task3.display_index(sid)}  (id {sid})",
                     str(task3.sessions_per_staff.get(sid, 0)), viz.COL_STAFF)
        for sid in task3.staff_ids[:6]
    ] or [viz.PanelRow("no staff detected yet", "-", viz.COL_MUTED)]
    staff_rows.append(
        viz.PanelRow("AVG SESSIONS / STAFF", f"{t3['average_sessions_per_staff']:.2f}", viz.COL_LINK, True)
    )
    viz.draw_panel(
        canvas, "TASK 3 - STAFF-CUSTOMER INTERACTION", staff_rows,
        origin=(14, 52 + 150), width=352,
        footnote=f"{t3['total_sessions']} sessions / {t3['staff_instances']} staff instances",
    )

    viz.draw_legend(
        canvas,
        [
            ("interested", viz.COL_INTEREST),
            ("entered", viz.COL_ENTERED),
            ("passed by", viz.COL_PASSED),
            ("staff", viz.COL_STAFF),
            ("interaction", viz.COL_LINK),
        ],
        origin=(canvas.shape[1] - 205, 52),
    )
    viz.draw_footer(
        canvas,
        f"t={format_timecode(t)}   tracked={len(visible)}",
        "entrance.mp4 - Task 1 + Task 3",
    )


# --------------------------------------------------------------------------- #
# interior: Task 2
# --------------------------------------------------------------------------- #
def run_interior(
    cfg: Config,
    video_path: str,
    outputs: Path,
    start_s: float = 0.0,
    max_seconds: Optional[float] = None,
    show_zones: bool = True,
    write_video: bool = True,
    dump_features: Optional[str] = None,
    reuse_cache: bool = False,
) -> RunResult:
    outputs.mkdir(parents=True, exist_ok=True)
    # The classifier always runs so its evidence lands in the audit CSV; the
    # verdict is only *acted on* when task2.exclude_staff is set.
    classifier = _staff_classifier(cfg.get_path("task2.staff", {}) or {},
                                   residency_default=90.0, min_obs_default=20)

    # ---- pass 1: inference (see the module docstring)
    cache_path = _pass_one(
        cfg, video_path, outputs, classifier,
        inside_store=lambda state: True,
        start_s=start_s, max_seconds=max_seconds,
        dump_features=dump_features, label="interior", reuse_cache=reuse_cache,
    )
    classifier.finalize()

    # ---- pass 2: metrics + annotation from the cache, with final roles
    task2 = ShelfInterestTask(cfg)
    draw_pose = bool(cfg.get_path("output.draw_keypoints", True))
    kp_conf_min = float(cfg.get_path("runtime.kp_conf_min", 0.35))

    reader = VideoReader(video_path, start_s=start_s, max_seconds=max_seconds)
    info = reader.info
    out_video = outputs / cfg.get_path("output.interior_video", "interior_annotated.mp4")
    writer = VideoWriter(out_video, info.fps, cfg.get_path("output.codec", "mp4v")) if write_video else None

    total = info.frame_count if max_seconds is None else int(max_seconds * info.fps)
    started = time.time()
    tracks = _replay_tracks(cfg)
    records = read_features(cache_path)
    pending = next(records, None)
    last_visible: List[TrackState] = []
    last_t = 0.0

    for frame_idx, t, frame in reader:
        last_t = t
        if pending is not None and int(pending["f"]) == frame_idx:
            visible, _ = tracks.step(pending)
            for state in visible:
                state.role = classifier.role_of(state.track_id)
                state.role_score = classifier.score_of(state.track_id)
            task2.update(t, visible)
            last_visible = visible
            pending = next(records, None)

        if writer is not None:
            canvas = frame.copy()
            _render_interior(canvas, t, last_visible, task2, show_zones, draw_pose, kp_conf_min)
            writer.write(canvas)

        _progress("interior:annotate", frame_idx - reader.start_frame, total, started)

    reader.close()
    if writer is not None:
        writer.close()
    task2.finalize(last_t)

    csvs = report.write_task2(outputs, task2.summary(), task2.episodes, task2.events)
    report.write_rows(outputs / "audit" / "task2_staff_features.csv", classifier.report_rows())
    report.write_json(
        outputs / "audit" / "interior_summary.json",
        {
            "video": str(video_path),
            "frames_processed": total,
            "fps": info.fps,
            "task2": task2.summary(),
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        },
    )
    return RunResult(video=out_video, csvs=csvs, summary={"task2": task2.summary()})


def _render_interior(
    canvas: np.ndarray,
    t: float,
    visible: List[TrackState],
    task2: ShelfInterestTask,
    show_zones: bool,
    draw_pose: bool,
    kp_conf_min: float,
) -> None:
    if show_zones:
        for shelf in task2.shelves:
            viz.draw_polygon(canvas, shelf.polygon, shelf.color, "", fill_alpha=0.14, thickness=2)
            viz.chip(canvas, f"SHELF {shelf.shelf_id}: {task2.counts[shelf.shelf_id]}",
                     (int(shelf.label_anchor[0]) - 45, int(shelf.label_anchor[1])), shelf.color, 0.56)

    for state in visible:
        if state.role == "staff" and task2.exclude_staff:
            viz.draw_person(canvas, state.box, viz.COL_STAFF, f"STAFF {state.track_id} (excluded)")
            continue

        active = task2.active_for(state.track_id, t)
        if active is not None:
            shelf_id, elapsed = active
            shelf = task2.shelf_by_id(shelf_id)
            color = shelf.color
            label = f"ID {state.track_id} | SHELF {shelf_id} | {elapsed:4.1f}s"
            anchor = shelf.polygon.nearest_point(state.ground)
            viz.draw_link(canvas, bbox_center(state.box), anchor, color, f"{shelf_id}  {elapsed:4.1f}s")
        else:
            candidate = task2.candidate_for(state.track_id)
            color = viz.COL_CUSTOMER
            label = f"ID {state.track_id}"
            if candidate is not None:
                label += f" | near {candidate.shelf_id}"

        if draw_pose:
            viz.draw_keypoints(canvas, state.keypoints, color, kp_conf_min)
        viz.draw_person(canvas, state.box, color, label)
        if state.facing.is_valid:
            viz.draw_facing(canvas, state.ground, state.facing.vector, color, 46)

    counts = task2.summary()
    rows = [
        viz.PanelRow(f"Shelf {shelf_id}", str(counts[shelf_id]),
                     task2.shelf_by_id(shelf_id).color, True)
        for shelf_id in task2.shelf_ids
    ]
    rows.append(viz.PanelRow("TOTAL EVENTS", str(sum(counts.values())), viz.COL_TEXT, True))
    viz.draw_panel(
        canvas, "TASK 2 - PER-SHELF CUSTOMER INTEREST", rows, origin=(14, 52), width=320,
        footnote="one event = a sustained, non-repeating interest episode",
    )
    viz.draw_legend(
        canvas,
        [(f"shelf {s.shelf_id}", s.color) for s in task2.shelves],
        origin=(canvas.shape[1] - 205, 52),
    )
    viz.draw_footer(
        canvas,
        f"t={format_timecode(t)}   tracked={len(visible)}",
        "interior.mp4 - Task 2",
    )


# --------------------------------------------------------------------------- #
# replay: recompute every metric from a feature cache, with no inference
# --------------------------------------------------------------------------- #
def _replay_tracks(cfg: Config) -> ReplayTracks:
    runtime = cfg.require("runtime")
    return ReplayTracks(
        smoothing_window=runtime.get("smoothing_window", 7),
        walk_speed_norm=runtime.get("walk_speed_norm", 0.9),
        motion_bias=runtime.get("motion_bias", 0.65),
        kp_conf_min=runtime.get("kp_conf_min", 0.35),
    )


def replay_entrance(cfg: Config, features: str, outputs: Optional[Path] = None) -> Dict[str, object]:
    """Recompute Tasks 1 and 3 from a cached run. Seconds, not an hour.

    Mirrors :func:`run_entrance` exactly - the same two passes over the same
    data, minus the model and the video - so replaying an unchanged config
    reproduces the run's numbers.
    """
    task1 = StoreInterestTask(cfg)
    task3 = StaffInteractionTask(cfg)
    classifier = _staff_classifier(cfg.require("task3.staff"), residency_default=30.0, min_obs_default=60)

    # pass 1 equivalent: accumulate staff evidence over the whole clip
    tracks = _replay_tracks(cfg)
    prev_t = None
    for record in read_features(features):
        t = float(record["t"])
        visible, appearance = tracks.step(record)
        dt = (t - prev_t) if prev_t is not None else 0.1
        prev_t = t
        for state in visible:
            classifier.observe_features(
                state, appearance.get(state.track_id), dt,
                inside_store=task1.store.contains(state.ground),
            )
    classifier.finalize()

    # pass 2 equivalent: drive the task state machines with the final roles
    tracks = _replay_tracks(cfg)
    last_t = 0.0
    for record in read_features(features):
        t = float(record["t"])
        visible, _ = tracks.step(record)
        last_t = t
        for state in visible:
            state.role = classifier.role_of(state.track_id)
            state.role_score = classifier.score_of(state.track_id)
        task1.update(t, visible)
        task3.update(t, visible, in_scene=task1.in_scene)

    task1.finalize(last_t)
    task3.finalize(last_t)

    if outputs is not None:
        outputs.mkdir(parents=True, exist_ok=True)
        report.write_task1(outputs, task1.summary(), task1.per_person_rows(), task1.events)
        report.write_task3(outputs, task3.per_staff_rows(), task3.summary(), task3.session_log,
                           classifier.report_rows(), task3.events)
    return {"task1": task1.summary(), "task3": task3.summary()}


def replay_interior(cfg: Config, features: str, outputs: Optional[Path] = None) -> Dict[str, object]:
    """Recompute Task 2 from a cached run."""
    task2 = ShelfInterestTask(cfg)
    classifier = _staff_classifier(cfg.get_path("task2.staff", {}) or {},
                                   residency_default=90.0, min_obs_default=20)

    tracks = _replay_tracks(cfg)
    prev_t = None
    for record in read_features(features):
        t = float(record["t"])
        visible, appearance = tracks.step(record)
        dt = (t - prev_t) if prev_t is not None else 0.1
        prev_t = t
        for state in visible:
            classifier.observe_features(state, appearance.get(state.track_id), dt, inside_store=True)
    classifier.finalize()

    tracks = _replay_tracks(cfg)
    last_t = 0.0
    for record in read_features(features):
        t = float(record["t"])
        visible, _ = tracks.step(record)
        last_t = t
        for state in visible:
            state.role = classifier.role_of(state.track_id)
            state.role_score = classifier.score_of(state.track_id)
        task2.update(t, visible)

    task2.finalize(last_t)

    if outputs is not None:
        outputs.mkdir(parents=True, exist_ok=True)
        report.write_task2(outputs, task2.summary(), task2.episodes, task2.events)
        report.write_rows(outputs / "audit" / "task2_staff_features.csv", classifier.report_rows())
    return {"task2": task2.summary()}
