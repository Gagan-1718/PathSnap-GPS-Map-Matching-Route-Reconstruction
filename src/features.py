"""Candidate discovery and feature calculation."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class Candidate:
    arc_id: int
    distance_m: float
    projection_fraction: float
    projected_longitude: float
    projected_latitude: float


def haversine_m(point_a, point_b) -> float:
    lon1, lat1 = np.radians(point_a)
    lon2, lat2 = np.radians(point_b)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def point_segment_distance(point, start, end) -> tuple:
    """Local equirectangular point-to-segment projection in metres."""
    lon, lat = np.radians(point)
    lon1, lat1 = np.radians(start)
    lon2, lat2 = np.radians(end)
    cos_lat = np.cos((lat1 + lat2) / 2)
    a = np.array([EARTH_RADIUS_M * (lon1 - lon) * cos_lat, EARTH_RADIUS_M * (lat1 - lat)])
    b = np.array([EARTH_RADIUS_M * (lon2 - lon) * cos_lat, EARTH_RADIUS_M * (lat2 - lat)])
    ab = b - a
    ab_sq = float(ab @ ab)
    if ab_sq == 0:
        fraction = 0.0
        closest = np.array([EARTH_RADIUS_M * (lon1 - lon) * cos_lat, EARTH_RADIUS_M * (lat1 - lat)])
    else:
        fraction = float(np.clip((-a @ ab) / ab_sq, 0.0, 1.0))
        closest = a + fraction * ab
    distance = float(np.linalg.norm(closest))
    projection_lon = (1 - fraction) * start[0] + fraction * end[0]
    projection_lat = (1 - fraction) * start[1] + fraction * end[1]
    return distance, fraction, projection_lon, projection_lat


def find_candidates(track, nodes, arcs, radius_m=100.0, max_candidates=10) -> list:
    candidates = []
    for gps in track:
        point = gps[:2]
        found = []
        for arc_id, arc in enumerate(arcs):
            start = nodes[int(arc[0])]
            end = nodes[int(arc[1])]
            distance, fraction, lon, lat = point_segment_distance(point, start, end)
            if distance <= radius_m:
                found.append(Candidate(arc_id, distance, fraction, lon, lat))
        found.sort(key=lambda c: (c.distance_m, c.arc_id))
        candidates.append(found[:max_candidates])
    return candidates


def candidate_feature_vector(candidate: Candidate, gps_point, nodes, arcs) -> np.ndarray:
    arc = arcs[candidate.arc_id]
    start, end = nodes[int(arc[0])], nodes[int(arc[1])]
    length = haversine_m(start, end)
    bearing = haversine_m(gps_point[:2], [candidate.projected_longitude, candidate.projected_latitude])
    return np.array([candidate.distance_m, length, candidate.projection_fraction, bearing], dtype=float)


def extract_features(track, nodes, arcs, candidates: Iterable) -> list:
    candidates = list(candidates)
    result = []
    for i, gps in enumerate(track):
        result.append([candidate_feature_vector(c, gps, nodes, arcs) for c in candidates[i]])
    return result
