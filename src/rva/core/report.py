"""Writing the deliverables.

Three CSVs are mandated by the brief.  This module writes exactly those,
plus a set of *audit* artefacts that are not required but make the numbers
defensible: a per-person / per-episode breakdown, a JSONL event log with the
timestamp and the evidence behind every increment, and a ``summary.json``
with the resolved configuration.  A reviewer can therefore trace any headline
number back to the second of video that produced it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _ensure(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_rows(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    p = _ensure(path)
    if not rows:
        names = list(fieldnames or [])
    else:
        names = list(fieldnames or rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in names})
    return p


def write_metric_csv(path: str | Path, pairs: Sequence[tuple[str, Any]]) -> Path:
    """The 'metric,count' shape the brief asks for: labelled totals."""
    rows = [{"metric": label, "value": value} for label, value in pairs]
    return write_rows(path, rows, ["metric", "value"])


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    p = _ensure(path)
    with open(p, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
    return p


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = _ensure(path)
    with open(p, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return p


# --------------------------------------------------------------------------- #
# task-specific writers (the required deliverables)
# --------------------------------------------------------------------------- #
def write_task1(outputs: Path, summary: Dict[str, int], per_person: List[Dict[str, Any]],
                events: List[Dict[str, Any]]) -> List[Path]:
    written = [
        write_metric_csv(
            outputs / "task1_store_interest.csv",
            [
                ("Total Interested", summary["total_interested"]),
                ("Interested Entered", summary["interested_entered"]),
                ("Interested Passed By", summary["interested_passed_by"]),
            ],
        ),
        write_rows(outputs / "audit" / "task1_per_person.csv", per_person),
        write_jsonl(outputs / "audit" / "task1_events.jsonl", events),
    ]
    return written


def write_task2(outputs: Path, counts: Dict[str, int], episodes: List[Dict[str, Any]],
                events: List[Dict[str, Any]]) -> List[Path]:
    rows = [{"shelf": shelf, "interest_events": counts[shelf]} for shelf in sorted(counts)]
    rows.append({"shelf": "TOTAL", "interest_events": sum(counts.values())})
    return [
        write_rows(outputs / "task2_shelf_interest.csv", rows, ["shelf", "interest_events"]),
        write_rows(
            outputs / "audit" / "task2_episodes.csv",
            episodes,
            ["shelf", "track_id", "start_s", "end_s", "duration_s", "bridged_gaps"],
        ),
        write_jsonl(outputs / "audit" / "task2_events.jsonl", events),
    ]


def write_task3(outputs: Path, per_staff: List[Dict[str, Any]], summary: Dict[str, Any],
                sessions: List[Dict[str, Any]], staff_features: List[Dict[str, Any]],
                events: List[Dict[str, Any]]) -> List[Path]:
    rows: List[Dict[str, Any]] = []
    for row in per_staff:
        rows.append(
            {
                "staff_instance": row["staff_instance"],
                "track_id": row["track_id"],
                "first_seen_s": row["first_seen_s"],
                "last_seen_s": row["last_seen_s"],
                "interaction_sessions": row["interaction_sessions"],
            }
        )
    rows.append(
        {
            "staff_instance": "AVERAGE",
            "track_id": "",
            "first_seen_s": "",
            "last_seen_s": "",
            "interaction_sessions": summary["average_sessions_per_staff"],
        }
    )
    return [
        write_rows(
            outputs / "task3_staff_interactions.csv",
            rows,
            ["staff_instance", "track_id", "first_seen_s", "last_seen_s", "interaction_sessions"],
        ),
        write_rows(
            outputs / "audit" / "task3_sessions.csv",
            sessions,
            ["staff_id", "customer_id", "start_s", "end_s", "duration_s", "bridged_gaps"],
        ),
        write_rows(outputs / "audit" / "task3_staff_features.csv", staff_features),
        write_jsonl(outputs / "audit" / "task3_events.jsonl", events),
    ]
