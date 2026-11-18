"""Per-candidate features and training labels."""
import numpy as np

FEATURES = [
    "dist_m",          # GPS point to road segment distance (m)
    "dist_rank",       # 0 = nearest candidate
    "dist_gap_m",      # extra distance compared with the nearest candidate
    "heading_diff",    # angle between vehicle direction and arc direction (deg)
    "speed_mps",       # vehicle speed around the point
    "arc_len_m",       # segment length
    "proj_frac",       # where the point falls along the segment (0..1)
    "n_candidates",    # how crowded the area is
    "time_gap_s",      # time since the previous GPS point
]


def motion(trip):
    """Heading (deg, 0 = north) and speed (m/s) using neighbouring points."""
    xy, t = trip["track_xy"], trip["track"]["t"].to_numpy()
    prev_i = np.r_[0, np.arange(len(xy) - 1)]
    next_i = np.r_[np.arange(1, len(xy)), len(xy) - 1]
    d = xy[next_i] - xy[prev_i]
    dt = np.maximum(t[next_i] - t[prev_i], 1e-6)
    heading = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    speed = np.linalg.norm(d, axis=1) / dt
    time_gap = np.r_[0.0, np.diff(t)]
    return heading, speed, time_gap


def add_features(cands, trip):
    heading, speed, time_gap = motion(trip)
    pt = cands["point"].to_numpy()
    diff = np.abs(heading[pt] - cands["arc_bearing"].to_numpy()) % 360
    cands["heading_diff"] = np.minimum(diff, 360 - diff)
    cands["speed_mps"] = speed[pt]
    cands["time_gap_s"] = time_gap[pt]
    cands["dist_gap_m"] = cands["dist_m"] - cands.groupby("point")["dist_m"].transform("min")
    cands["n_candidates"] = cands.groupby("point")["arc"].transform("size")
    return cands


def add_labels(cands, route):
    """Label = 1 for the nearest candidate that lies on the reference route.

    The route lists arcs in driving order, not one arc per GPS point, so the
    per-point label is derived: among candidates that are on the route, the
    one closest to the point (direction-consistent first) is positive.
    """
    on_route = cands["arc"].isin(set(route.tolist())).to_numpy()
    score = cands["dist_m"].to_numpy() + np.where(cands["heading_diff"].to_numpy() > 90, 25.0, 0.0)
    score = np.where(on_route, score, np.inf)
    cands["_score"] = score
    best = cands.groupby("point")["_score"].transform("min")
    cands["label"] = (on_route & (cands["_score"].to_numpy() == best.to_numpy())).astype(int)
    # break exact ties: keep only the first positive per point
    dup = cands["label"].eq(1) & cands.duplicated(["point", "label"])
    cands.loc[dup, "label"] = 0
    return cands.drop(columns="_score")
