"""Build the candidate feature table for every trip in a split.

Output: cache/<split>_features.pkl  (not committed, rebuilt from raw data)
        results/features/<split>_candidate_stats.csv
Run:    python -m feature_engineering.build_dataset
"""
import time

import pandas as pd

from data_processing.load_data import ROOT, list_trips, prepare_trip
from feature_engineering.candidates import generate_candidates
from feature_engineering.features import add_features, add_labels

CACHE = ROOT / "cache"
OUT = ROOT / "results" / "features"


def build_trip(split, trip_id):
    trip = prepare_trip(split, trip_id)
    df = add_labels(add_features(generate_candidates(trip), trip), trip["route"])
    df.insert(0, "trip_id", trip_id)
    return df, len(trip["track"])


def build_split(split):
    frames, stats = [], []
    for tid in list_trips(split):
        t0 = time.time()
        df, n_pts = build_trip(split, tid)
        has_pos = df.groupby("point")["label"].max()
        stats.append({"trip_id": tid, "gps_points": n_pts, "candidate_rows": len(df),
                      "points_with_candidates": has_pos.size,
                      "candidate_miss_rate": 1 - has_pos.sum() / n_pts,
                      "seconds": round(time.time() - t0, 2)})
        frames.append(df)
        print(f"{split} {tid}: {len(df):>7} rows  miss={stats[-1]['candidate_miss_rate']:.3f}")
    data = pd.concat(frames, ignore_index=True)
    data.to_pickle(CACHE / f"{split}_features.pkl")
    pd.DataFrame(stats).to_csv(OUT / f"{split}_candidate_stats.csv", index=False)
    return data


def main():
    CACHE.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "student_test"):
        build_split(split)


if __name__ == "__main__":
    main()
