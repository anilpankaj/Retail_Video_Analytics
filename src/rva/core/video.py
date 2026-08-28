"""Video reading and annotated-video writing.

The pipeline runs inference on every ``frame_stride``-th frame but writes
*every* source frame to the annotated output, holding the most recent
analysis state in between.  At the default stride of 3 on 30 fps footage the
held state is at most 66 ms stale, which is invisible to a reviewer, and it
cuts CPU inference cost by two-thirds without changing any temporal rule —
all thresholds are expressed in seconds, not frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


class VideoReader:
    """Sequential frame reader that yields ``(frame_idx, timestamp, frame)``."""

    def __init__(self, path: str | Path, start_s: float = 0.0, max_seconds: Optional[float] = None):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.info = VideoInfo(
            path=self.path,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        self.start_frame = int(round(start_s * fps))
        self.max_frames = int(round(max_seconds * fps)) if max_seconds else None
        if self.start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        idx = self.start_frame
        emitted = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            # Timestamps are derived from the frame index rather than
            # CAP_PROP_POS_MSEC: the latter is unreliable on these MPEG-4
            # files, and every temporal rule in the project depends on a
            # monotonically increasing clock.
            yield idx, idx / self.info.fps, frame
            idx += 1
            emitted += 1
            if self.max_frames is not None and emitted >= self.max_frames:
                break

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoWriter:
    """Thin wrapper that lazily opens on the first frame."""

    def __init__(self, path: str | Path, fps: float, codec: str = "mp4v") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.codec = codec
        self._writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {self.path}")
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def format_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
