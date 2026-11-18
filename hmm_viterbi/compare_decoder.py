"""Compare two HMM decoder settings fairly, on trips the decoder was NOT tuned on.

Uses out-of-fold probabilities of the final model (5-fold GroupKFold on the
80 training trips), then decodes each trip with both settings. The 15 trips
used to tune the decoder are excluded, so the comparison is on 65 unseen
training trips. The 10 student-test trips are reported as a second check.

Output: results/hmm/decoder_comparison.csv
Run:    python -m hmm_viterbi.compare_decoder lightgbm_tuned old.json new.json
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

from data_processing.load_data import ROOT, prepare_trip
from feature_engineering.features import FEATURES
from hmm_viterbi.evaluate_hmm import score_trip
from hmm_viterbi.hmm import RoadGraph, decode_trip
from model_evaluation.pipeline_cv import spec_for, trip_folds
from model_training.train_models import N_FOLDS, load_split


def decode_all(split, df, probs, settings):
    rows = []
    for tid, cands in df.groupby("trip_id"):
        trip = prepare_trip(split, tid)
        graph = RoadGraph(trip["arcs"])
        for label, params in settings.items():
            pts, pred = decode_trip(cands, probs[cands.index], trip["arcs"], trip["track"]["t"].to_numpy(), params, graph)
            rows.append({"split": split, "trip_id": tid, "setting": label, **score_trip(pts, pred, cands, trip)})
    return rows


def main(model="lightgbm_tuned", old_file=None, new_file=None):
    settings = {"old": json.loads(open(old_file).read()), "new": json.loads(open(new_file).read())}
    excluded = set(settings["new"].get("holdout_trips", [])) | set(settings["old"].get("holdout_trips", []))

    # out-of-fold probabilities on the training trips
    train = load_split("train", labelled_only=False)
    labelled = train[train["ref_arc"] >= 0]
    fold_of = trip_folds(labelled)
    spec, params = spec_for(model)
    probs = pd.Series(np.nan, index=train.index)
    for k in range(N_FOLDS):
        held = [t for t, f in fold_of.items() if f == k]
        fit = labelled[~labelled["trip_id"].isin(held)]
        m = spec["build"](params).fit(fit[FEATURES].to_numpy(), fit["label"].to_numpy())
        mask = train["trip_id"].isin(held)
        probs[mask] = m.predict_proba(train.loc[mask, FEATURES].to_numpy())[:, 1]
    unseen = train[~train["trip_id"].isin(excluded)]
    rows = decode_all("train", unseen, probs, settings)

    # student-test trips with the saved final model
    test = load_split("student_test", labelled_only=False)
    final = joblib.load(ROOT / "saved_models" / f"{model}.joblib")
    tprobs = pd.Series(final.predict_proba(test[FEATURES].to_numpy())[:, 1], index=test.index)
    rows += decode_all("student_test", test, tprobs, settings)

    per = pd.DataFrame(rows)
    per = per[per["labelled_points"] > 0]
    out = []
    for (split, label), d in per.groupby(["split", "setting"]):
        out.append({"split": split, "setting": label, "trips": d["trip_id"].nunique(),
                    "top1_accuracy": (d["top1_accuracy"] * d["labelled_points"]).sum() / d["labelled_points"].sum(),
                    "route_continuity": d["route_continuity"].mean(), "route_f1": d["route_f1"].mean()})
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "results" / "hmm" / "decoder_comparison.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
