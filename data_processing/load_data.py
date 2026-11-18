"""Read raw PathSnap trip files and convert coordinates to metres."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EARTH_RADIUS_M = 6_371_000.0


def list_trips(split):
    """Return sorted trip ids for a split ('train' or 'student_test')."""
    return sorted(p.name for p in (DATA_DIR / split).iterdir() if p.is_dir())


def _read(path, names, dtype):
    if not path.exists():
        return None
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_trip(split, trip_id, with_route=True):
    """Load one trip. Route (ground truth) is only read when with_route=True."""
    d = DATA_DIR / split / trip_id
    trip = {
        "trip_id": trip_id,
        "track": _read(d / f"{trip_id}.track", ["lon", "lat", "t"], float),
        "nodes": _read(d / f"{trip_id}.nodes", ["lon", "lat"], float),
        "arcs": _read(d / f"{trip_id}.arcs", ["u", "v"], int),
    }
    if with_route:
        route = _read(d / f"{trip_id}.route", ["arc"], int)
        trip["route"] = None if route is None else route["arc"].to_numpy()
    return trip


MAX_SPEED_MPS = 50.0  # 180 km/h: faster jumps are GPS errors


def remove_spikes(track):
    """Drop single-point GPS jumps: points that are too fast to reach AND to leave."""
    proj = LocalProjection(track["lon"].mean(), track["lat"].mean())
    xy = proj.to_xy(track["lon"].to_numpy(), track["lat"].to_numpy())
    t = track["t"].to_numpy()
    if len(xy) < 3:
        return track
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
    spike = np.zeros(len(xy), bool)
    spike[1:-1] = (speed[:-1] > MAX_SPEED_MPS) & (speed[1:] > MAX_SPEED_MPS)
    return track[~spike].reset_index(drop=True)


def clean_track(track):
    """Drop invalid rows, sort by time, remove duplicate timestamps and GPS spikes."""
    t = track.dropna()
    t = t[t["lon"].between(-180, 180) & t["lat"].between(-90, 90)]
    t = t.sort_values("t").drop_duplicates("t").reset_index(drop=True)
    return remove_spikes(t)


class LocalProjection:
    """Equirectangular projection centred on the trip.

    The trips cover only a few kilometres, so this local flat projection
    has negligible distortion and gives distances in metres.
    """

    def __init__(self, lon0, lat0):
        self.lon0 = np.radians(lon0)
        self.lat0 = np.radians(lat0)
        self.cos_lat0 = np.cos(self.lat0)

    def to_xy(self, lon, lat):
        x = EARTH_RADIUS_M * (np.radians(lon) - self.lon0) * self.cos_lat0
        y = EARTH_RADIUS_M * (np.radians(lat) - self.lat0)
        return np.column_stack([x, y])


def prepare_trip(split, trip_id, with_route=True):
    """Load, clean and project a trip into metres."""
    trip = load_trip(split, trip_id, with_route)
    trip["track"] = clean_track(trip["track"])
    tr = trip["track"]
    proj = LocalProjection(tr["lon"].mean(), tr["lat"].mean())
    trip["proj"] = proj
    trip["track_xy"] = proj.to_xy(tr["lon"].to_numpy(), tr["lat"].to_numpy())
    trip["nodes_xy"] = proj.to_xy(trip["nodes"]["lon"].to_numpy(), trip["nodes"]["lat"].to_numpy())
    return trip
