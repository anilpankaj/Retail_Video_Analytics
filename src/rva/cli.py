"""Command line entry point.

    python -m rva.cli all                       # both videos, all three tasks
    python -m rva.cli entrance --max-seconds 60 # quick smoke test
    python -m rva.cli interior
    python -m rva.cli zones --config configs/entrance.yaml --video data/entrance.mp4
    python -m rva.cli grid   --video data/interior.mp4 --at 25

Any config value can be overridden from the shell without editing YAML::

    python -m rva.cli entrance --set task1.interest.score_threshold=1.6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2

from .config import as_points, load_config, parse_cli_overrides
from .core import viz
from .core.geometry import Polygon, Polyline
from .pipeline import replay_entrance, replay_interior, run_entrance, run_interior

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS = ROOT / "outputs"


def _add_common(parser: argparse.ArgumentParser, default_config: str, default_video: str) -> None:
    parser.add_argument("--config", default=str(ROOT / default_config))
    parser.add_argument("--video", default=str(ROOT / default_video))
    parser.add_argument("--outputs", default=str(DEFAULT_OUTPUTS))
    parser.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="process only the first N seconds (smoke tests)")
    parser.add_argument("--no-video", action="store_true", help="compute metrics without writing video")
    parser.add_argument("--no-zones", action="store_true", help="hide zone overlays in the output video")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override any config value")
    parser.add_argument("--dump-features", default=None, metavar="PATH",
                        help="write a per-frame feature cache so every threshold "
                             "can later be re-tuned with `rva.cli replay` in seconds")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="skip inference and re-render from an existing feature "
                             "cache (minutes instead of an hour)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rva",
        description="Retail behavioural video analytics (Hendricks take-home assessment)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("all", help="run both videos and produce every deliverable")
    _add_common(p_all, "configs/entrance.yaml", "data/entrance.mp4")
    p_all.add_argument("--interior-config", default=str(ROOT / "configs/interior.yaml"))
    p_all.add_argument("--interior-video", default=str(ROOT / "data/interior.mp4"))

    p_ent = sub.add_parser("entrance", help="Task 1 + Task 3 on entrance.mp4")
    _add_common(p_ent, "configs/entrance.yaml", "data/entrance.mp4")

    p_int = sub.add_parser("interior", help="Task 2 on interior.mp4")
    _add_common(p_int, "configs/interior.yaml", "data/interior.mp4")

    p_zones = sub.add_parser("zones", help="render the configured zones onto a reference frame")
    p_zones.add_argument("--config", required=True)
    p_zones.add_argument("--video", required=True)
    p_zones.add_argument("--at", type=float, default=20.0)
    p_zones.add_argument("--out", default=str(DEFAULT_OUTPUTS / "zones_preview.jpg"))

    p_replay = sub.add_parser(
        "replay",
        help="recompute all metrics from a feature cache with a different config "
             "- no model, no video, runs in seconds",
    )
    p_replay.add_argument("--features", required=True, help="JSONL written by --dump-features")
    p_replay.add_argument("--config", required=True)
    p_replay.add_argument("--scene", choices=["entrance", "interior"], default=None,
                          help="defaults to the config's `name`")
    p_replay.add_argument("--outputs", default=None,
                          help="also rewrite the CSVs into this directory")
    p_replay.add_argument("--set", dest="overrides", action="append", default=[],
                          metavar="KEY=VALUE")

    p_grid = sub.add_parser("grid", help="dump a coordinate-grid frame for hand-drawing zones")
    p_grid.add_argument("--video", required=True)
    p_grid.add_argument("--at", type=float, default=20.0)
    p_grid.add_argument("--out", default=str(DEFAULT_OUTPUTS / "grid.jpg"))

    return parser


# --------------------------------------------------------------------------- #
# calibration helpers
# --------------------------------------------------------------------------- #
def _frame_at(video: str, seconds: float):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame at {seconds}s from {video}")
    return frame


def cmd_grid(args) -> int:
    frame = _frame_at(args.video, args.at)
    h, w = frame.shape[:2]
    for x in range(0, w, 80):
        cv2.line(frame, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(frame, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    for y in range(0, h, 60):
        cv2.line(frame, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(frame, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"wrote {out}")
    return 0


def cmd_zones(args) -> int:
    cfg = load_config(args.config)
    frame = _frame_at(args.video, args.at)
    scene = cfg.require("scene")

    if "shelves" in scene:
        for entry in scene["shelves"]:
            poly = Polygon(as_points(entry["polygon"]), entry["id"])
            color = tuple(int(c) for c in entry.get("color", (0, 200, 255)))
            viz.draw_polygon(frame, poly, color, f"SHELF {entry['id']}", fill_alpha=0.22)
        if "roi_polygon" in scene:
            viz.draw_polygon(frame, Polygon(as_points(scene["roi_polygon"])), (255, 255, 255),
                             "ROI", fill_alpha=0.0, thickness=1)
    else:
        viz.draw_polygon(frame, Polygon(as_points(scene["walkway_polygon"])), viz.COL_WALKWAY,
                         "WALKWAY", fill_alpha=0.20)
        viz.draw_polygon(frame, Polygon(as_points(scene["store_polygon"])), viz.COL_STORE,
                         "STORE", fill_alpha=0.20)
        viz.draw_polyline(frame, Polyline(as_points(scene["storefront_line"])), viz.COL_FRONT, 3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------- #
# run commands
# --------------------------------------------------------------------------- #
def _run_entrance(args) -> dict:
    cfg = load_config(args.config, parse_cli_overrides(args.overrides))
    result = run_entrance(
        cfg,
        video_path=args.video,
        outputs=Path(args.outputs),
        start_s=args.start,
        max_seconds=args.max_seconds,
        show_zones=not args.no_zones,
        write_video=not args.no_video,
        dump_features=args.dump_features,
        reuse_cache=args.reuse_cache,
    )
    print("\nTask 1 - Store Interest and Conversion")
    for key, value in result.summary["task1"].items():
        print(f"  {key.replace('_', ' ').title():<24} {value}")
    print("Task 3 - Staff-Customer Interaction")
    for key, value in result.summary["task3"].items():
        print(f"  {key.replace('_', ' ').title():<24} {value}")
    print(f"  annotated video -> {result.video}")
    return result.summary


def _run_interior(args) -> dict:
    cfg = load_config(args.config, parse_cli_overrides(args.overrides))
    result = run_interior(
        cfg,
        video_path=args.video,
        outputs=Path(args.outputs),
        start_s=args.start,
        max_seconds=args.max_seconds,
        show_zones=not args.no_zones,
        write_video=not args.no_video,
        dump_features=args.dump_features,
        reuse_cache=args.reuse_cache,
    )
    print("\nTask 2 - Per-Shelf Customer Interest")
    for shelf, count in sorted(result.summary["task2"].items()):
        print(f"  Shelf {shelf:<20} {count}")
    print(f"  annotated video -> {result.video}")
    return result.summary


def cmd_replay(args) -> int:
    cfg = load_config(args.config, parse_cli_overrides(args.overrides))
    scene = args.scene or cfg.get_path("name", "entrance")
    outputs = Path(args.outputs) if args.outputs else None
    if scene == "interior":
        summary = replay_interior(cfg, args.features, outputs)
    else:
        summary = replay_entrance(cfg, args.features, outputs)
    for task, values in summary.items():
        print(f"{task}:")
        for key, value in values.items():
            print(f"  {key:<32} {value}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()

    if args.command == "grid":
        return cmd_grid(args)
    if args.command == "zones":
        return cmd_zones(args)
    if args.command == "replay":
        return cmd_replay(args)

    if args.command == "entrance":
        _run_entrance(args)
    elif args.command == "interior":
        _run_interior(args)
    elif args.command == "all":
        _run_entrance(args)
        interior_args = argparse.Namespace(**vars(args))
        interior_args.config = args.interior_config
        interior_args.video = args.interior_video
        _run_interior(interior_args)

    print(f"\ntotal wall time {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
