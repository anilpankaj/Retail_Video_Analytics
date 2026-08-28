"""Person detection, tracking and the per-track state every task reads.

Model choice
------------
A single **YOLO11 pose** model is used rather than a detector plus a separate
pose network.  One forward pass yields the box *and* the 17 COCO keypoints,
which we need for body orientation, reach direction and the torso crop used
by the staff classifier.  On a CPU-only machine that is roughly half the cost
of running two models, and it guarantees box/keypoint association for free.

Tracking is delegated to Ultralytics' BoT-SORT.  It is a motion + appearance
tracker with a Kalman filter, which suits fixed-camera retail footage where
people are frequently occluded by fixtures for a second or two.

Track identity and the fragment linker
--------------------------------------
Every headline metric is a count of *unique people*, so an identity switch
inflates the answer.  Two defences are in place:

* short-lived tracks (< ``min_track_frames``) are ignored entirely;
* :class:`TrackLinker` stitches a newly-born track onto a recently-dead one
  when they are close in time and space and their torso colour histograms
  agree.  This recovers the common failure mode of a shopper disappearing
  behind a gondola and re-emerging with a new id.

The linker is conservative by design: it would rather leave two fragments
separate (over-count by one) than merge two different people (which would
silently corrupt several metrics at once).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import bbox_ground_point, bbox_height, normalize
from .orientation import Orientation, best_wrist, fuse, motion_facing, pose_facing, reach_vector

Box = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# per-track state
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    t: float
    frame_idx: int
    box: Box
    ground: Tuple[float, float]
    height: float
    keypoints: Optional[np.ndarray]
    score: float


@dataclass
class TrackState:
    """Everything the task logic is allowed to know about one person."""

    track_id: int
    history: Deque[Sample] = field(default_factory=lambda: deque(maxlen=90))
    first_t: float = 0.0
    last_t: float = 0.0
    n_obs: int = 0

    facing: Orientation = field(default_factory=Orientation)
    #: pose-only estimate (shoulder normal + head yaw). Kept separately because
    #: "turning their HEAD toward the store while walking past" is a distinct
    #: behaviour from "turning their BODY", and the fused vector - which a fast
    #: walker's motion direction dominates - would erase it.
    head: Orientation = field(default_factory=Orientation)
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed_norm: float = 0.0
    reach: Optional[Tuple[float, float]] = None
    #: image position of the most confident wrist (second reference point)
    wrist: Optional[Tuple[float, float]] = None

    #: role assigned by the staff classifier: "customer" | "staff" | "unknown"
    role: str = "unknown"
    role_score: float = 0.0

    #: colour signature used by the fragment linker
    signature: Optional[np.ndarray] = None
    #: ids that were merged into this one
    merged_ids: List[int] = field(default_factory=list)
    #: free-form per-task scratch space (keeps task logic out of this class)
    scratch: Dict[str, object] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------- #
    @property
    def last(self) -> Sample:
        return self.history[-1]

    @property
    def box(self) -> Box:
        return self.history[-1].box

    @property
    def ground(self) -> Tuple[float, float]:
        return self.history[-1].ground

    @property
    def height(self) -> float:
        return self.history[-1].height

    @property
    def keypoints(self) -> Optional[np.ndarray]:
        return self.history[-1].keypoints

    @property
    def duration(self) -> float:
        return max(0.0, self.last_t - self.first_t)

    def norm(self, pixels: float) -> float:
        """Convert a pixel distance into body-heights for this person."""
        h = max(self.height, 1e-3)
        return pixels / h


def velocity_of(state: "TrackState", smoothing_window: int = 7) -> Tuple[Tuple[float, float], float]:
    """Smoothed image-space velocity and its body-height-normalised magnitude."""
    hist = state.history
    if len(hist) < 2:
        return (0.0, 0.0), 0.0
    window = min(smoothing_window, len(hist))
    a = hist[-window]
    b = hist[-1]
    dt = b.t - a.t
    if dt <= 1e-6:
        return (0.0, 0.0), 0.0
    vx = (b.ground[0] - a.ground[0]) / dt
    vy = (b.ground[1] - a.ground[1]) / dt
    height = max(float(np.median([s.height for s in list(hist)[-window:]])), 1.0)
    return (vx, vy), float(math.hypot(vx, vy) / height)


def derive_signals(
    state: "TrackState",
    smoothing_window: int = 7,
    walk_speed_norm: float = 0.9,
    motion_bias: float = 0.65,
    kp_conf_min: float = 0.35,
) -> None:
    """Recompute velocity, orientation, reach and wrist for the newest sample.

    Module-level (rather than a TrackManager method) so the replay path can
    reuse it verbatim: replayed track state must be bit-identical to live
    track state, or a cached re-run would not reproduce the original numbers.
    """
    state.velocity, state.speed_norm = velocity_of(state, smoothing_window)
    pose = pose_facing(state.keypoints, kp_conf_min=kp_conf_min)
    motion = motion_facing(state.velocity, state.speed_norm, walk_speed_norm)
    state.head = pose
    state.facing = _smooth_facing_state(state, fuse(pose, motion, motion_bias))
    state.reach = reach_vector(state.keypoints, kp_conf_min=kp_conf_min)
    state.wrist = best_wrist(state.keypoints, kp_conf_min=kp_conf_min)


def _smooth_facing_state(state: "TrackState", current: Orientation) -> Orientation:
    """Exponential smoothing so the facing arrow does not jitter frame to frame."""
    previous: Optional[Orientation] = state.scratch.get("_facing_prev")  # type: ignore[assignment]
    if not current.is_valid:
        return previous or Orientation()
    if previous is None or not previous.is_valid:
        state.scratch["_facing_prev"] = current
        return current
    alpha = 0.55
    vec = normalize(
        (
            alpha * current.vector[0] + (1 - alpha) * previous.vector[0],
            alpha * current.vector[1] + (1 - alpha) * previous.vector[1],
        )
    )
    smoothed = Orientation(vector=vec, confidence=current.confidence, source=current.source)
    state.scratch["_facing_prev"] = smoothed
    return smoothed


class TrackManager:
    """Owns detection, tracking, smoothing and the derived per-track signals."""

    def __init__(
        self,
        weights: str,
        imgsz: int = 960,
        conf: float = 0.3,
        iou: float = 0.55,
        device: str = "cpu",
        tracker_cfg: str = "botsort.yaml",
        smoothing_window: int = 7,
        walk_speed_norm: float = 0.9,
        motion_bias: float = 0.65,
        kp_conf_min: float = 0.35,
        history_len: int = 120,
        half: bool = False,
        linker: Optional["TrackLinker"] = None,
        retire_after_s: float = 1.0,
    ) -> None:
        from ultralytics import YOLO  # imported lazily so unit tests need no torch

        self.model = YOLO(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.tracker_cfg = tracker_cfg
        self.smoothing_window = int(smoothing_window)
        self.walk_speed_norm = float(walk_speed_norm)
        self.motion_bias = float(motion_bias)
        self.kp_conf_min = float(kp_conf_min)
        self.history_len = int(history_len)
        self.half = bool(half)

        self.tracks: Dict[int, TrackState] = {}
        self.active_ids: List[int] = []
        self.linker = linker or TrackLinker(enabled=False)
        self.retire_after_s = float(retire_after_s)
        self._retired_ids: set[int] = set()

    # ---------------------------------------------------------------- update #
    def update(self, frame: np.ndarray, t: float, frame_idx: int) -> List[TrackState]:
        """Run one inference step and return the tracks visible in this frame."""
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=[0],  # person
            persist=True,
            tracker=self.tracker_cfg,
            verbose=False,
            half=self.half,
        )
        result = results[0]
        self.active_ids = []
        # Retire quiet tracks BEFORE matching: a track that died two frames ago
        # must be linkable by a track that is born on this frame, otherwise the
        # fragment linker can never fire on the shortest (commonest) gaps.
        self._retire_stale(t)

        if result.boxes is None or result.boxes.id is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        kps = None
        if getattr(result, "keypoints", None) is not None and result.keypoints is not None:
            data = result.keypoints.data
            if data is not None and len(data):
                kps = data.cpu().numpy()  # (n, 17, 3)

        visible: List[TrackState] = []
        for i, raw_id in enumerate(ids):
            box = tuple(float(v) for v in boxes[i])
            keypoints = kps[i] if kps is not None and i < len(kps) else None
            track_id = self.linker.resolve(int(raw_id))
            state = self.tracks.get(track_id)
            fresh = state is None
            if state is None:
                state = TrackState(track_id=track_id,
                                   history=deque(maxlen=self.history_len))
                state.first_t = t
                self.tracks[track_id] = state

            sample = Sample(
                t=t,
                frame_idx=frame_idx,
                box=box,  # type: ignore[arg-type]
                ground=bbox_ground_point(box),
                height=max(bbox_height(box), 1.0),
                keypoints=keypoints,
                score=float(scores[i]),
            )
            state.history.append(sample)
            state.last_t = t
            state.n_obs += 1
            self._update_derived(state, frame)

            if fresh:
                linked = self.linker.try_link(state)
                if linked is not None and linked in self.tracks and linked != track_id:
                    parent = self.tracks[linked]
                    parent.history.append(sample)
                    parent.last_t = t
                    parent.n_obs += 1
                    parent.merged_ids.append(track_id)
                    self._update_derived(parent, frame)
                    del self.tracks[track_id]
                    self._retired_ids.discard(linked)
                    state = parent
                    track_id = linked

            visible.append(state)
            self.active_ids.append(track_id)

        return visible

    def _retire_stale(self, now: float) -> None:
        """Hand tracks that have gone quiet to the linker so they can be re-used."""
        for track_id, state in self.tracks.items():
            if track_id in self._retired_ids:
                continue
            if (now - state.last_t) > self.retire_after_s and state.n_obs >= 3:
                self.linker.retire(state)
                self._retired_ids.add(track_id)

    # ------------------------------------------------------------ internals #
    def _update_derived(self, state: TrackState, frame: np.ndarray) -> None:
        derive_signals(
            state,
            smoothing_window=self.smoothing_window,
            walk_speed_norm=self.walk_speed_norm,
            motion_bias=self.motion_bias,
            kp_conf_min=self.kp_conf_min,
        )
        if state.signature is None or state.n_obs % 10 == 0:
            sig = torso_signature(frame, state.box, state.keypoints, self.kp_conf_min)
            if sig is not None:
                state.signature = sig if state.signature is None else 0.7 * state.signature + 0.3 * sig

# --------------------------------------------------------------------------- #
# appearance signature + fragment linking
# --------------------------------------------------------------------------- #
def torso_box(box: Box, keypoints: Optional[np.ndarray], kp_conf_min: float = 0.35) -> Box:
    """A box covering the chest/apron area, from keypoints where possible."""
    x1, y1, x2, y2 = box
    if keypoints is not None and len(keypoints) >= 17:
        ls, rs = keypoints[5], keypoints[6]
        lh, rh = keypoints[11], keypoints[12]
        if min(ls[2], rs[2]) >= kp_conf_min and max(lh[2], rh[2]) >= kp_conf_min:
            xs = [ls[0], rs[0], lh[0], rh[0]]
            ys = [ls[1], rs[1], lh[1], rh[1]]
            tx1, tx2 = float(min(xs)), float(max(xs))
            ty1, ty2 = float(min(ys)), float(max(ys))
            pad_x = 0.15 * max(tx2 - tx1, 1.0)
            return (tx1 - pad_x, ty1, tx2 + pad_x, ty2)
    # Fallback: the middle band of the bounding box.
    h = y2 - y1
    w = x2 - x1
    return (x1 + 0.22 * w, y1 + 0.25 * h, x2 - 0.22 * w, y1 + 0.62 * h)


def crop(frame: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1 = max(0, int(round(box[0])))
    y1 = max(0, int(round(box[1])))
    x2 = min(w, int(round(box[2])))
    y2 = min(h, int(round(box[3])))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


def torso_signature(
    frame: np.ndarray,
    box: Box,
    keypoints: Optional[np.ndarray],
    kp_conf_min: float = 0.35,
    bins: Tuple[int, int] = (12, 8),
) -> Optional[np.ndarray]:
    """Normalised H-S histogram of the torso, used to match track fragments."""
    patch = crop(frame, torso_box(box, keypoints, kp_conf_min))
    if patch is None:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


class TrackLinker:
    """Stitch fragmented tracks back together (conservatively)."""

    def __init__(
        self,
        max_gap_s: float = 2.5,
        max_dist_norm: float = 2.0,
        min_similarity: float = 0.55,
        enabled: bool = True,
        link_within_obs: int = 3,
    ) -> None:
        self.max_gap_s = float(max_gap_s)
        self.max_dist_norm = float(max_dist_norm)
        self.min_similarity = float(min_similarity)
        self.enabled = bool(enabled)
        self.link_within_obs = int(link_within_obs)
        self.alias: Dict[int, int] = {}
        self._retired: List[Tuple[int, float, Tuple[float, float], float, Optional[np.ndarray]]] = []

    def resolve(self, track_id: int) -> int:
        """Canonical id for a track, following any merge chain."""
        seen = set()
        current = track_id
        while current in self.alias and current not in seen:
            seen.add(current)
            current = self.alias[current]
        return current

    def retire(self, state: TrackState) -> None:
        self._retired.append(
            (state.track_id, state.last_t, state.ground, state.height, state.signature)
        )
        # keep the retired buffer small
        cutoff = state.last_t - (self.max_gap_s * 4)
        self._retired = [r for r in self._retired if r[1] >= cutoff][-64:]

    def try_link(self, state: TrackState) -> Optional[int]:
        """If ``state`` looks like the continuation of a dead track, return its id."""
        if not self.enabled or state.n_obs > self.link_within_obs or state.signature is None:
            return None
        best_id: Optional[int] = None
        best_sim = self.min_similarity
        for track_id, last_t, ground, height, signature in self._retired:
            gap = state.first_t - last_t
            if gap <= 0 or gap > self.max_gap_s:
                continue
            scale = max(height, state.height, 1.0)
            if math.dist(ground, state.ground) / scale > self.max_dist_norm:
                continue
            if signature is None:
                continue
            similarity = float(
                cv2.compareHist(signature.reshape(-1, 1), state.signature.reshape(-1, 1),
                                cv2.HISTCMP_CORREL)
            )
            if similarity > best_sim:
                best_sim, best_id = similarity, track_id
        if best_id is not None:
            self.alias[state.track_id] = best_id
        return best_id
