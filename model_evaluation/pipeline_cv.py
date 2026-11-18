"""Model selection on the full pipeline: classifier + HMM, cross-validated on training trips.

For each model, 5-fold GroupKFold by trip on the 80 training trips (same folds
as model training). The model is fitted on 4 folds, predicts every GPS point
of the held-out trips, the HMM decodes each trip, and the route is scored.
This uses training data only and covers 80 trips instead of 10.

A paired bootstrap over trips (2000 resamples) then estimates how often each
model beats the others, so small differences are not mistaken for real ones.

Output: results/model_selection/pipeline_cv_per_trip.csv
        results/model_selection/pipeline_cv_summary.csv
        results/model_selection/bootstrap_pairwise.csv
Run:    python -m model_evaluation.pipeline_cv decision_tree random_forest lightgbm lightgbm_tuned
"""
import itertools
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data_processing.load_data import ROOT, prepare_trip
from feature_engineering.features import FEATURES
from hmm_viterbi.evaluate_hmm import classifier_only, load_params, score_trip
from hmm_viterbi.hmm import RoadGraph, decode_trip
from model_training.train_models import MODELS, N_FOLDS, SEED, load_split

OUT = ROOT / "results" / "model_selection"
N_BOOT = 2000


def spec_for(name):
    base = name.removesuffix("_tuned")
    spec, params = MODELS[base], MODELS[base]["params"]
    if name.endswith("_tuned"):
        params = json.loads((ROOT / "saved_models" / f"{name}_config.json").read_text())["params"]
    return spec, params


def trip_folds(labelled):
    """Same trip -> fold assignment as model_training (GroupKFold on labelled rows)."""
    g = labelled["trip_id"].to_numpy()
    fold_of = {}
    for k, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(g, g, g)):
        for t in np.unique(g[va]):
            fold_of[t] = k
    return fold_of


def run_model(name, df_all, labelled, fold_of, trips, graphs, params_hmm):
    spec, params = spec_for(name)
    rows = []
    for k in range(N_FOLDS):
        held = [t for t, f in fold_of.items() if f == k]
        fit = labelled[~labelled["trip_id"].isin(held)]
        model = spec["build"](params).fit(fit[FEATURES].to_numpy(), fit["label"].to_numpy())
        for tid in held:
            cands = df_all[df_all["trip_id"] == tid]
            trip = trips[tid]
            prob = model.predict_proba(cands[FEATURES].to_numpy())[:, 1]
            pts, clf = classifier_only(cands, prob)
            rows.append({"model": name, "fold": k, "trip_id": tid, "method": "classifier_only",
                         **score_trip(pts, clf, cands, trip)})
            t0 = time.time()
            pts, hmm = decode_trip(cands, prob, trip["arcs"], trip["track"]["t"].to_numpy(), params_hmm, graphs[tid])
            rows.append({"model": name, "fold": k, "trip_id": tid, "method": "classifier_plus_hmm",
                         **score_trip(pts, hmm, cands, trip), "decode_s": time.time() - t0})
        print(f"{name} fold {k} done", flush=True)
    return rows


def weighted_top1(d):
    return (d["top1_accuracy"] * d["labelled_points"]).sum() / d["labelled_points"].sum()


def summarise(per_trip):
    out = []
    for (m, meth), d in per_trip.groupby(["model", "method"]):
        d = d[d["labelled_points"] > 0]
        fold = d.groupby("fold").apply(weighted_top1, include_groups=False)
        out.append({"model": m, "method": meth, "trips": d["trip_id"].nunique(),
                    "top1_accuracy": weighted_top1(d), "top1_fold_std": fold.std(),
                    "route_continuity": d["route_continuity"].mean(), "route_f1": d["route_f1"].mean()})
    return pd.DataFrame(out).sort_values(["method", "top1_accuracy"], ascending=[True, False])


def bootstrap(per_trip):
    """Paired bootstrap over trips on HMM top-1: how often does model A beat model B?"""
    d = per_trip[(per_trip["method"] == "classifier_plus_hmm") & (per_trip["labelled_points"] > 0)]
    correct = d.assign(c=d["top1_accuracy"] * d["labelled_points"]).pivot(index="trip_id", columns="model", values="c")
    n = d.pivot(index="trip_id", columns="model", values="labelled_points").iloc[:, 0]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(n), size=(N_BOOT, len(n)))
    rows = []
    for a, b in itertools.permutations(correct.columns, 2):
        diff = (correct[a].to_numpy()[idx].sum(1) - correct[b].to_numpy()[idx].sum(1)) / n.to_numpy()[idx].sum(1)
        rows.append({"model_a": a, "model_b": b, "mean_diff_pts": diff.mean() * 100,
                     "ci95_low_pts": np.percentile(diff, 2.5) * 100, "ci95_high_pts": np.percentile(diff, 97.5) * 100,
                     "p_a_beats_b": float((diff > 0).mean())})
    return pd.DataFrame(rows)


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    df_all = load_split("train", labelled_only=False)
    labelled = df_all[df_all["ref_arc"] >= 0]
    fold_of = trip_folds(labelled)
    trips = {t: prepare_trip("train", t) for t in fold_of}
    graphs = {t: RoadGraph(trips[t]["arcs"]) for t in trips}
    params_hmm = load_params()

    per_file = OUT / "pipeline_cv_per_trip.csv"
    old = pd.read_csv(per_file, dtype={"trip_id": str}) if per_file.exists() else None
    rows = []
    for name in names:
        rows += run_model(name, df_all, labelled, fold_of, trips, graphs, params_hmm)
    per_trip = pd.DataFrame(rows)
    if old is not None:
        per_trip = pd.concat([old[~old["model"].isin(names)], per_trip], ignore_index=True)
    per_trip.to_csv(per_file, index=False)

    s = summarise(per_trip)
    s.to_csv(OUT / "pipeline_cv_summary.csv", index=False)
    b = bootstrap(per_trip)
    b.to_csv(OUT / "bootstrap_pairwise.csv", index=False)
    print(s.round(4).to_string(index=False))
    print(b[b["mean_diff_pts"] > 0].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["decision_tree", "random_forest", "lightgbm"])
