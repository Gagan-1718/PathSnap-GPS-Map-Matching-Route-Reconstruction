"""Tune HMM decoder parameters using training data only.

15 training trips are held out; the classifier is refit on the other 65
training trips, and a small grid of decoder parameters is scored on the
held-out trips by observation-level top-1 accuracy.

Output: saved_models/hmm_params.json, results/hmm/decoder_grid.csv
Run:    python -m hmm_viterbi.tune_decoder [model_name]
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

from data_processing.load_data import ROOT, prepare_trip
from feature_engineering.features import FEATURES
from hmm_viterbi.hmm import DEFAULT_PARAMS, RoadGraph, decode_trip
from model_training.train_models import MODELS, SEED, load_split

GRID = {
    "hop_penalty": [0.5, 1.0, 2.0],
    "uturn_penalty": [1.0, 3.0],
    "disconnect_penalty": [6.0, 12.0, 20.0],
}
N_HOLDOUT = 15


def main(model_name="decision_tree"):
    out = ROOT / "results" / "hmm"
    out.mkdir(parents=True, exist_ok=True)
    df = load_split("train", labelled_only=False)
    trips = np.array(sorted(df["trip_id"].unique()))
    holdout = np.random.default_rng(SEED).choice(trips, N_HOLDOUT, replace=False)

    fit_df = df[~df["trip_id"].isin(holdout) & (df["ref_arc"] >= 0)]
    spec = MODELS[model_name]
    model = spec["build"](spec["params"]).fit(fit_df[FEATURES].to_numpy(), fit_df["label"].to_numpy())

    cache = []
    for tid in holdout:
        cands = df[df["trip_id"] == tid]
        trip = prepare_trip("train", tid)
        prob = model.predict_proba(cands[FEATURES].to_numpy())[:, 1]
        ref = cands.groupby("point")["ref_arc"].first()
        cache.append((cands, prob, trip, RoadGraph(trip["arcs"]), ref))

    rows = []
    for values in itertools.product(*GRID.values()):
        params = {**DEFAULT_PARAMS, **dict(zip(GRID, values))}
        correct = total = 0
        for cands, prob, trip, graph, ref in cache:
            pts, pred = decode_trip(cands, prob, trip["arcs"], trip["track"]["t"].to_numpy(), params, graph)
            r = ref.reindex(pts).to_numpy()
            lab = r >= 0
            correct += int((pred[lab] == r[lab]).sum())
            total += int(lab.sum())
        rows.append({**dict(zip(GRID, values)), "holdout_top1_accuracy": correct / total})
        print(rows[-1])

    grid = pd.DataFrame(rows).sort_values("holdout_top1_accuracy", ascending=False)
    grid.to_csv(out / "decoder_grid.csv", index=False)
    best = {**DEFAULT_PARAMS, **grid.iloc[0][list(GRID)].to_dict()}
    best["tuned_with_model"] = model_name
    best["holdout_trips"] = holdout.tolist()
    (ROOT / "saved_models" / "hmm_params.json").write_text(json.dumps(best, indent=2))
    print("best:", best)


if __name__ == "__main__":
    main(*sys.argv[1:])
