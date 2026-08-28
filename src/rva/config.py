"""Configuration loading.

Every tunable number in this project lives in a YAML file under ``configs/``.
Nothing that affects a reported count is hard-coded in the source, so a
reviewer can re-run the pipeline with different thresholds without touching
Python.

A config may declare ``extends: <path>`` to inherit from a base file; the
child is deep-merged over the parent.  Paths in ``extends`` are resolved
relative to the file that declares them.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml


# --------------------------------------------------------------------------- #
# dict helpers
# --------------------------------------------------------------------------- #
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config(dict):
    """A dict with dotted-path access, e.g. ``cfg.get_path("task1.entry.confirm_s")``."""

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        sentinel = object()
        value = self.get_path(path, sentinel)
        if value is sentinel:
            raise KeyError(f"missing required config key: {path}")
        return value


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> Config:
    """Load a YAML config, resolving ``extends`` chains and applying overrides."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle) or {}

    parent_ref = raw.pop("extends", None)
    if parent_ref:
        parent = load_config(path.parent / parent_ref)
        raw = deep_merge(parent, raw)

    if overrides:
        raw = deep_merge(raw, overrides)

    raw.setdefault("_source", str(path))
    return Config(raw)


def parse_cli_overrides(pairs: Iterable[str]) -> Dict[str, Any]:
    """Turn ``["task1.entry.confirm_s=2.0", ...]`` into a nested dict."""
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must look like key.path=value, got {pair!r}")
        key, _, value = pair.partition("=")
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


# --------------------------------------------------------------------------- #
# typed accessors used across the pipeline
# --------------------------------------------------------------------------- #
Point = Tuple[float, float]


def as_points(raw: Sequence[Sequence[float]]) -> List[Point]:
    """Normalise a YAML polygon/polyline into a list of (x, y) float tuples."""
    return [(float(p[0]), float(p[1])) for p in raw]
