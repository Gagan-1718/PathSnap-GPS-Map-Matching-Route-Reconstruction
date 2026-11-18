"""Evaluate saved models on the 10 student-testing trips.

Output: results/evaluation/student_test_metrics.csv
        results/evaluation/student_test_per_trip.csv
Run:    python -m model_evaluation.evaluate_models
"""
import time

import joblib
import pandas as pd

from data_processing.load_data import ROOT, load_trip
from feature_engineering.features import FEATURES
from model_evaluation.metrics import all_metrics, route_continuity, top1_predictions
from model_training.train_models import MODELS, load_split

OUT = ROOT / "results" / "evaluation"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_split("student_test")
    arcs = {t: load_trip("student_test", t, with_route=False)["arcs"] for t in df["trip_id"].unique()}

    summary, per_trip = [], []
    for name in MODELS:
        model = joblib.load(ROOT / "saved_models" / f"{name}.joblib")
        t0 = time.time()
        prob = model.predict_proba(df[FEATURES].to_numpy())[:, 1]
        infer_s = time.time() - t0

        top = top1_predictions(df, prob)
        cont = {t: route_continuity(g["arc"].to_numpy(), arcs[t]) for t, g in top.groupby("trip_id")}
        m = all_metrics(df, prob)
        summary.append({"model": name, **m, "route_continuity": sum(cont.values()) / len(cont),
                        "inference_time_s": infer_s, "inference_ms_per_point": infer_s / len(top) * 1000})
        for t, g in top.groupby("trip_id"):
            per_trip.append({"model": name, "trip_id": t, "top1_accuracy": g["label"].mean(),
                             "route_continuity": cont[t], "points": len(g)})

    s = pd.DataFrame(summary)
    s.to_csv(OUT / "student_test_metrics.csv", index=False)
    pd.DataFrame(per_trip).to_csv(OUT / "student_test_per_trip.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
