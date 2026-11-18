"""Prediction pipeline and server-side scoring used by the API.

Prediction never reads the reference route. Scoring reads it separately,
after prediction, and only returns summary metrics (never the route itself).
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from data_processing.load_data import ROOT, list_trips, load_trip, prepare_trip
from feature_engineering.candidates import generate_candidates
from feature_engineering.features import FEATURES, add_features, reference_arcs
from hmm_viterbi.evaluate_hmm import classifier_only, load_params, route_scores
from hmm_viterbi.hmm import decode_trip
from model_evaluation.metrics import route_continuity

CONFIG = json.loads((ROOT / "deployment" / "config.json").read_text(encoding="utf-8-sig"))
MODEL_DIR = ROOT / "saved_models"


def load_models(names=None):
    """Load saved models. names = which models to expose (from config visible_models)."""
    paths = sorted(MODEL_DIR.glob("*.joblib"))
    if names:
        paths = [p for p in paths if p.stem in names]
    return {p.stem: joblib.load(p) for p in paths}


def available_trips(split):
    return list_trips(split)


def predict_trip(split, trip_id, model, params):
    """Full pipeline: clean -> candidates -> features -> classifier -> HMM/Viterbi."""
    t0 = time.time()
    trip = prepare_trip(split, trip_id, with_route=False)
    cands = add_features(generate_candidates(trip), trip)
    prob = model.predict_proba(cands[FEATURES].to_numpy())[:, 1]
    _, clf_arcs = classifier_only(cands, prob)
    points, arcs = decode_trip(cands, prob, trip["arcs"], trip["track"]["t"].to_numpy(), params)
    return trip, points, arcs, clf_arcs, time.time() - t0


def score_prediction(split, trip_id, trip, points, arcs, clf_arcs):
    """Server-side evaluation against the reference route (kept on this machine)."""
    route = load_trip(split, trip_id)["route"]
    if route is None:
        return {"evaluated": False}
    ref = reference_arcs({**trip, "route": route})[points]
    lab = ref >= 0
    prec, rec, f1 = route_scores(arcs, route)
    return {
        "evaluated": True,
        "top1_accuracy": float(np.mean(arcs[lab] == ref[lab])) if lab.any() else None,
        "classifier_only_top1_accuracy": float(np.mean(clf_arcs[lab] == ref[lab])) if lab.any() else None,
        "labelled_points": int(lab.sum()),
        "route_continuity": route_continuity(arcs, trip["arcs"]),
        "classifier_only_route_continuity": route_continuity(clf_arcs, trip["arcs"]),
        "route_precision": prec,
        "route_recall": rec,
        "route_f1": f1,
    }


def route_geometry(trip, arc_seq):
    """Predicted route as [lat, lon] polylines, split where arcs are not connected."""
    nodes = trip["nodes"][["lat", "lon"]].to_numpy()
    a = trip["arcs"].to_numpy()
    seq = np.asarray(arc_seq)
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    lines, cur = [], []
    for i, arc in enumerate(seq):
        if cur and a[seq[i - 1], 1] != a[arc, 0]:
            lines.append(cur)
            cur = []
        if not cur:
            cur.append(nodes[a[arc, 0]].round(6).tolist())
        cur.append(nodes[a[arc, 1]].round(6).tolist())
    if cur:
        lines.append(cur)
    return lines, int(len(seq))


def gps_points(trip, limit):
    tr = trip["track"][["lat", "lon"]].to_numpy()
    step = max(1, int(np.ceil(len(tr) / limit)))
    return tr[::step].round(6).tolist()
