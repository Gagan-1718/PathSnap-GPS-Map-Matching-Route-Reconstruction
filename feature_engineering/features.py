"""Per-candidate features and training labels."""
import numpy as np
from scipy.spatial import cKDTree

from feature_engineering.candidates import arc_geometry, point_segment

FEATURES = [
    "dist_m",            # GPS point to road segment distance (m)
    "dist_rank",         # 0 = nearest candidate
    "dist_gap_m",        # extra distance compared with the nearest candidate
    "heading_diff",      # angle between vehicle direction and arc direction (deg)
    "heading_reliable",  # 1 if the vehicle moved enough to trust its heading
    "speed_mps",         # vehicle speed around the point
    "arc_len_m",         # segment length
    "proj_frac",         # where the point falls along the segment (0..1)
    "n_candidates",      # how crowded the area is
    "time_gap_s",        # time since the previous GPS point
]

HEADING_WINDOW = 2       # use points i-2 .. i+2 to estimate direction
MIN_MOVE_M = 5.0         # below this movement the heading is unreliable
LABEL_RADIUS_M = 30.0    # a point is labelled only if a route arc is this close
WRONG_WAY_PENALTY_M = 25.0


def motion(trip):
    """Heading (deg, 0 = north), heading reliability and speed (m/s)."""
    xy, t = trip["track_xy"], trip["track"]["t"].to_numpy()
    n, w = len(xy), HEADING_WINDOW
    prev_i = np.maximum(np.arange(n) - w, 0)
    next_i = np.minimum(np.arange(n) + w, n - 1)
    d = xy[next_i] - xy[prev_i]
    moved = np.linalg.norm(d, axis=1)
    dt = np.maximum(t[next_i] - t[prev_i], 1e-6)
    heading = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    reliable = moved >= MIN_MOVE_M
    time_gap = np.r_[0.0, np.diff(t)]
    return heading, reliable, moved / dt, time_gap


def add_features(cands, trip):
    heading, reliable, speed, time_gap = motion(trip)
    pt = cands["point"].to_numpy()
    diff = np.abs(heading[pt] - cands["arc_bearing"].to_numpy()) % 360
    diff = np.minimum(diff, 360 - diff)
    # when the car is (almost) stopped the heading is noise: use a neutral 90 deg
    cands["heading_diff"] = np.where(reliable[pt], diff, 90.0)
    cands["heading_reliable"] = reliable[pt].astype(int)
    cands["speed_mps"] = speed[pt]
    cands["time_gap_s"] = time_gap[pt]
    cands["dist_gap_m"] = cands["dist_m"] - cands.groupby("point")["dist_m"].transform("min")
    cands["n_candidates"] = cands.groupby("point")["arc"].transform("size")
    return cands


def reference_arcs(trip):
    """Reference arc for each GPS point (-1 when no route arc is close enough).

    The .route file lists arcs in driving order, not one arc per GPS point,
    and it does not always cover the whole GPS track. So for each point we
    take the closest route arc within LABEL_RADIUS_M, preferring arcs whose
    direction matches the vehicle's direction. Points with no route arc
    nearby are left unlabelled and are excluded from training and scoring.
    """
    pts = trip["track_xy"]
    route = np.unique(trip["route"])
    p, q, length, bearing = arc_geometry(trip)
    heading, reliable, _, _ = motion(trip)

    n = np.maximum(np.ceil(length[route] / 5.0).astype(int), 1) + 1
    owner = np.repeat(route, n)
    s = np.concatenate([np.linspace(0, 1, k) for k in n])
    tree = cKDTree(p[owner] + s[:, None] * (q[owner] - p[owner]))
    hits = tree.query_ball_point(pts, r=LABEL_RADIUS_M + 2.5)

    pi = np.repeat(np.arange(len(pts)), [len(h) for h in hits])
    ai = owner[np.concatenate(hits).astype(int)] if len(pi) else np.array([], int)
    key = np.unique(pi.astype(np.int64) * len(p) + ai)
    pi, ai = key // len(p), key % len(p)

    dist, _ = point_segment(pts[pi], p[ai], q[ai])
    diff = np.abs(heading[pi] - bearing[ai]) % 360
    wrong_way = (np.minimum(diff, 360 - diff) > 90) & reliable[pi]
    score = np.where(dist <= LABEL_RADIUS_M, dist + WRONG_WAY_PENALTY_M * wrong_way, np.inf)

    ref = np.full(len(pts), -1)
    best = np.full(len(pts), np.inf)
    order = np.argsort(score)[::-1]          # write best last so it wins
    best[pi[order]] = score[order]
    ref[pi[order]] = ai[order]
    ref[~np.isfinite(best)] = -1
    return ref


def add_labels(cands, trip):
    """label = 1 for the candidate that is the point's reference arc.

    ref_arc = -1 marks unlabelled points (route does not cover them).
    """
    ref = reference_arcs(trip)
    cands["ref_arc"] = ref[cands["point"].to_numpy()]
    cands["label"] = (cands["arc"].to_numpy() == cands["ref_arc"].to_numpy()).astype(int)
    return cands
