"""Dataset inventory and data-quality checks for every trip.

Output: results/eda/trip_summary.csv and results/eda/split_summary.csv
Run:    python -m data_processing.explore_data
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from data_processing.load_data import DATA_DIR, ROOT, list_trips, load_trip, LocalProjection

OUT = ROOT / "results" / "eda"


def trip_stats(split, trip_id):
    trip = load_trip(split, trip_id)
    tr, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    files = sorted(p.suffix for p in (DATA_DIR / split / trip_id).iterdir())

    dt = np.diff(tr["t"].to_numpy())
    proj = LocalProjection(tr["lon"].mean(), tr["lat"].mean())
    xy = proj.to_xy(tr["lon"].to_numpy(), tr["lat"].to_numpy())
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)

    n = len(nodes)
    g = coo_matrix((np.ones(len(arcs)), (arcs["u"], arcs["v"])), shape=(n, n))
    n_comp, labels = connected_components(g, directed=True, connection="weak")
    degree = np.bincount(np.r_[arcs["u"], arcs["v"]], minlength=n)

    route_ok = route is not None and len(route) > 0
    route_connected = np.nan
    if route_ok:
        a = arcs.to_numpy()
        route_connected = float(np.mean(a[route[:-1], 1] == a[route[1:], 0])) if len(route) > 1 else 1.0

    return {
        "split": split,
        "trip_id": trip_id,
        "files": " ".join(files),
        "gps_points": len(tr),
        "missing_values": int(tr.isna().sum().sum()),
        "duplicate_times": int(tr["t"].duplicated().sum()),
        "time_not_sorted": int((dt < 0).sum()),
        "duration_s": float(tr["t"].iloc[-1] - tr["t"].iloc[0]),
        "pct_dt_1s": float(np.mean(dt == 1) * 100),
        "max_gap_s": float(dt.max()),
        "gaps_over_10s": int((dt > 10).sum()),
        "max_step_m": float(step.max()),
        "length_km": float(step.sum() / 1000),
        "lon_min": tr["lon"].min(), "lon_max": tr["lon"].max(),
        "lat_min": tr["lat"].min(), "lat_max": tr["lat"].max(),
        "nodes": n,
        "arcs": len(arcs),
        "isolated_nodes": int((degree == 0).sum()),
        "components": int(n_comp),
        "largest_component_pct": float(np.bincount(labels).max() / n * 100),
        "route_arcs": len(route) if route_ok else 0,
        "route_consecutive_connected_pct": route_connected * 100,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [trip_stats(s, t) for s in ("train", "student_test") for t in list_trips(s)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trip_summary.csv", index=False)

    summary = df.groupby("split").agg(
        trips=("trip_id", "count"),
        gps_points=("gps_points", "sum"),
        mean_points=("gps_points", "mean"),
        missing_values=("missing_values", "sum"),
        mean_pct_dt_1s=("pct_dt_1s", "mean"),
        trips_with_gap_over_10s=("gaps_over_10s", lambda s: int((s > 0).sum())),
        mean_nodes=("nodes", "mean"),
        mean_arcs=("arcs", "mean"),
        mean_route_arcs=("route_arcs", "mean"),
        mean_route_connected_pct=("route_consecutive_connected_pct", "mean"),
    )
    summary.to_csv(OUT / "split_summary.csv")
    print(summary.T.round(2).to_string())


if __name__ == "__main__":
    main()
