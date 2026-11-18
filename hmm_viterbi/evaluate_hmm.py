"""Compare classifier-only predictions with classifier + HMM/Viterbi decoding.

Runs on the 10 student-testing trips for every saved model.
Output: results/hmm/classifier_vs_hmm.csv, results/hmm/classifier_vs_hmm_per_trip.csv
Run:    python -m hmm_viterbi.evaluate_hmm
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from data_processing.load_data import ROOT, prepare_trip
from feature_engineering.features import FEATURES
from hmm_viterbi.hmm import DEFAULT_PARAMS, decode_trip
from model_evaluation.metrics import route_continuity
from model_training.train_models import MODELS, load_split

OUT = ROOT / "results" / "hmm"
PARAMS_FILE = ROOT / "saved_models" / "hmm_params.json"


def load_params():
    return json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else dict(DEFAULT_PARAMS)


def route_scores(pred_arcs, route):
    """Sequence-level: overlap between predicted arc set and reference route arcs."""
    p, r = set(np.asarray(pred_arcs).tolist()), set(np.asarray(route).tolist())
    tp = len(p & r)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(r) if r else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def score_trip(points, pred, cands, trip):
    ref = cands.groupby("point")["ref_arc"].first().reindex(points).to_numpy()
    lab = ref >= 0
    prec, rec, f1 = route_scores(pred, trip["route"])
    return {"top1_accuracy": float(np.mean(pred[lab] == ref[lab])) if lab.any() else np.nan,
            "labelled_points": int(lab.sum()),
            "route_continuity": route_continuity(pred, trip["arcs"]),
            "route_precision": prec, "route_recall": rec, "route_f1": f1}


def classifier_only(cands, prob):
    d = cands[["point", "arc"]].assign(prob=prob)
    top = d.loc[d.groupby("point")["prob"].idxmax()].sort_values("point")
    return top["point"].to_numpy(), top["arc"].to_numpy()


def evaluate(split="student_test", params=None, model_names=None):
    params = params or load_params()
    df = load_split(split, labelled_only=False)
    trips = {t: prepare_trip(split, t) for t in df["trip_id"].unique()}
    rows = []
    names = model_names or (list(MODELS) + [f"{m}_tuned" for m in MODELS])
    for name in names:
        path = ROOT / "saved_models" / f"{name}.joblib"
        if not path.exists():
            continue
        model = joblib.load(path)
        for tid, cands in df.groupby("trip_id"):
            trip = trips[tid]
            t0 = time.time()
            prob = model.predict_proba(cands[FEATURES].to_numpy())[:, 1]
            t_model = time.time() - t0
            pts, pred = classifier_only(cands, prob)
            rows.append({"model": name, "method": "classifier_only", "trip_id": tid,
                         **score_trip(pts, pred, cands, trip), "time_s": t_model})
            t0 = time.time()
            pts, pred = decode_trip(cands, prob, trip["arcs"], trip["track"]["t"].to_numpy(), params)
            rows.append({"model": name, "method": "classifier_plus_hmm", "trip_id": tid,
                         **score_trip(pts, pred, cands, trip), "time_s": t_model + time.time() - t0})
    return pd.DataFrame(rows)


def summarise(per_trip):
    w = per_trip.assign(correct=per_trip["top1_accuracy"] * per_trip["labelled_points"])
    g = w.groupby(["model", "method"])
    s = pd.DataFrame({
        "top1_accuracy": g["correct"].sum() / g["labelled_points"].sum(),
        "route_continuity": g["route_continuity"].mean(),
        "route_precision": g["route_precision"].mean(),
        "route_recall": g["route_recall"].mean(),
        "route_f1": g["route_f1"].mean(),
        "time_per_trip_s": g["time_s"].mean(),
    })
    return s.reset_index()


def main(names=None):
    OUT.mkdir(parents=True, exist_ok=True)
    per_trip = evaluate(model_names=names)
    per_trip.to_csv(OUT / "classifier_vs_hmm_per_trip.csv", index=False)
    s = summarise(per_trip)
    s.to_csv(OUT / "classifier_vs_hmm.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
