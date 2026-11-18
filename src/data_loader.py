"""Readers for the PathSnap trip file formats."""

from pathlib import Path
from typing import Union

import numpy as np


def _read(path: Union[str, Path], columns: int) -> np.ndarray:
    value = np.loadtxt(path, dtype=float, ndmin=2)
    if value.shape[1] != columns:
        raise ValueError(f"{path} must contain {columns} columns")
    return value


def load_track(path: Union[str, Path]) -> np.ndarray:
    return _read(path, 3)


def load_nodes(path: Union[str, Path]) -> np.ndarray:
    return _read(path, 2)


def load_arcs(path: Union[str, Path]) -> np.ndarray:
    arcs = _read(path, 2)
    return arcs.astype(np.int64)


def load_route(path: Union[str, Path]) -> np.ndarray:
    route = _read(path, 1)
    return route[:, 0].astype(np.int64)


def load_trip(trip_dir: Union[str, Path]) -> dict:
    """Load a trip without loading anything except data files on disk."""
    trip_dir = Path(trip_dir)
    def find_file(kind: str, stem: str) -> Path:
        choices = (
            trip_dir / f"{stem}.{kind}",
            trip_dir / f"{stem}_{kind}",
            trip_dir / stem,
            trip_dir / f"{kind}.{kind}",
            trip_dir / f"{kind}.{stem}",
        )
        for choice in choices:
            if choice.exists():
                return choice
        matches = list(trip_dir.glob(f"*.{kind}"))
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"No .{kind} file found in {trip_dir}")

    route_path = None
    for kind, stem in (("route", "route"), ("route", "reference")):
        try:
            route_path = find_file(kind, stem)
            break
        except FileNotFoundError:
            pass
    result = {
        "track": load_track(find_file("track", "track")),
        "nodes": load_nodes(find_file("nodes", "nodes")),
        "arcs": load_arcs(find_file("arcs", "arcs")),
    }
    if route_path.exists():
        result["route"] = load_route(route_path)
    else:
        result["route"] = None
    return result
