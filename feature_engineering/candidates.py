"""Candidate road-segment generation for every GPS point."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SEARCH_RADIUS_M = 50.0     # candidates must lie within this distance
FALLBACK_RADIUS_M = 200.0  # used only when nothing is inside SEARCH_RADIUS_M
MAX_CANDIDATES = 10        # keep the nearest K arcs (both directions count)
SAMPLE_STEP_M = 10.0       # spacing of points sampled along each arc


def arc_geometry(trip):
    """Start/end coordinates (m), length and bearing of every arc."""
    a = trip["arcs"].to_numpy()
    p, q = trip["nodes_xy"][a[:, 0]], trip["nodes_xy"][a[:, 1]]
    d = q - p
    length = np.linalg.norm(d, axis=1)
    bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360  # 0 = north
    return p, q, length, bearing


def point_segment(pts, p, q):
    """Distance from pts to segments pq and the projection fraction along pq."""
    d = q - p
    l2 = np.einsum("ij,ij->i", d, d)
    frac = np.where(l2 > 0, np.einsum("ij,ij->i", pts - p, d) / np.where(l2 > 0, l2, 1), 0.0)
    frac = np.clip(frac, 0, 1)
    proj = p + frac[:, None] * d
    return np.linalg.norm(pts - proj, axis=1), frac


def _sample_arcs(p, q, length, arc_ids):
    """Points every SAMPLE_STEP_M along each arc, with the arc id of each sample."""
    n = np.maximum(np.ceil(length[arc_ids] / SAMPLE_STEP_M).astype(int), 1) + 1
    owner = np.repeat(arc_ids, n)
    t = np.concatenate([np.linspace(0, 1, k) for k in n])
    xy = p[owner] + t[:, None] * (q[owner] - p[owner])
    return xy, owner


def generate_candidates(trip):
    """Return a DataFrame of (point, arc) candidate pairs with distance info."""
    pts = trip["track_xy"]
    p, q, length, bearing = arc_geometry(trip)

    # keep only arcs close to the trajectory's bounding box (speed-up)
    lo, hi = pts.min(0) - FALLBACK_RADIUS_M, pts.max(0) + FALLBACK_RADIUS_M
    near = np.all((np.minimum(p, q) <= hi) & (np.maximum(p, q) >= lo), axis=1)
    arc_ids = np.flatnonzero(near)

    sxy, owner = _sample_arcs(p, q, length, arc_ids)
    tree = cKDTree(sxy)
    pad = SAMPLE_STEP_M / 2

    def pairs_within(idx, radius):
        hits = tree.query_ball_point(pts[idx], r=radius + pad)
        pi = np.repeat(idx, [len(h) for h in hits])
        ai = owner[np.concatenate(hits).astype(int)] if len(pi) else np.array([], int)
        return pi, ai

    pi, ai = pairs_within(np.arange(len(pts)), SEARCH_RADIUS_M)
    found = np.zeros(len(pts), bool)
    found[pi] = True
    if (~found).any():  # fallback for points far from every road
        fpi, fai = pairs_within(np.flatnonzero(~found), FALLBACK_RADIUS_M)
        pi, ai = np.r_[pi, fpi], np.r_[ai, fai]

    key = np.unique(pi.astype(np.int64) * len(trip["arcs"]) + ai)
    pi, ai = key // len(trip["arcs"]), key % len(trip["arcs"])

    dist, frac = point_segment(pts[pi], p[ai], q[ai])
    df = pd.DataFrame({"point": pi, "arc": ai, "dist_m": dist, "proj_frac": frac,
                       "arc_len_m": length[ai], "arc_bearing": bearing[ai]})
    df = df[(df["dist_m"] <= SEARCH_RADIUS_M) | ~found[df["point"].to_numpy()]]
    df = df.sort_values(["point", "dist_m"], kind="stable")
    df["dist_rank"] = df.groupby("point").cumcount()
    return df[df["dist_rank"] < MAX_CANDIDATES].reset_index(drop=True)
