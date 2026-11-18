"""Aggregate experiment results (from results/*.csv) for the UI.

Only summary metrics are returned. No reference routes or labels.
"""
import json
import math

import pandas as pd

from data_processing.load_data import ROOT

RES = ROOT / "results"


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _read(path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else None


def model_rows(names):
    cv = _read(RES / "training" / "cv_summary.csv")
    test = _read(RES / "evaluation" / "student_test_metrics.csv")
    hmm = _read(RES / "hmm" / "classifier_vs_hmm.csv")
    rows = []
    for name in names:
        base = name.removesuffix("_tuned")
        cfg_path = ROOT / "saved_models" / f"{name}_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        r = {"model": name, "family": cfg.get("family"), "params": cfg.get("params"), "tuned": name.endswith("_tuned")}
        if r["tuned"]:
            hpo = _read(RES / "hpo" / f"{base}_untuned_vs_tuned.csv")
            if hpo is not None:
                t = hpo[hpo["version"] == "tuned"].iloc[0]
                r.update(cv_macro_f1=t["cv_macro_f1"], cv_log_loss=t["cv_log_loss"],
                         test_macro_f1=t["test_macro_f1"], test_log_loss=t["test_log_loss"])
        else:
            if cv is not None and name in set(cv["model"]):
                c = cv[cv["model"] == name].iloc[0]
                r.update(cv_macro_f1=c["macro_f1_mean"], cv_macro_f1_std=c["macro_f1_std"],
                         cv_log_loss=c["log_loss_mean"], train_time_s=c["train_time_s_mean"])
            if test is not None and name in set(test["model"]):
                t = test[test["model"] == name].iloc[0]
                r.update(test_macro_f1=t["macro_f1"], test_log_loss=t["log_loss"],
                         test_precision=t["precision"], test_recall=t["recall"])
        if hmm is not None:
            for method, tag in [("classifier_only", "clf"), ("classifier_plus_hmm", "hmm")]:
                h = hmm[(hmm["model"] == name) & (hmm["method"] == method)]
                if len(h):
                    h = h.iloc[0]
                    r.update({f"{tag}_top1": h["top1_accuracy"], f"{tag}_continuity": h["route_continuity"],
                              f"{tag}_route_f1": h["route_f1"], f"{tag}_time_per_trip_s": h["time_per_trip_s"]})
        rows.append({k: _clean(v.item() if hasattr(v, "item") else v) for k, v in r.items()})
    return rows


def dataset_summary():
    split = _read(RES / "eda" / "split_summary.csv")
    out = {}
    if split is not None:
        for _, s in split.iterrows():
            out[s["split"]] = {"trips": int(s["trips"]), "gps_points": int(s["gps_points"]),
                               "mean_pct_dt_1s": float(s["mean_pct_dt_1s"]),
                               "mean_arcs": float(s["mean_arcs"])}
    for name in ["train", "student_test"]:
        st = _read(RES / "features" / f"{name}_candidate_stats.csv")
        if st is not None and name in out:
            out[name]["label_coverage"] = float(st["labelled_points"].sum() / st["gps_points"].sum())
            out[name]["candidate_miss_rate"] = float(
                1 - ((1 - st["candidate_miss_rate"]) * st["labelled_points"]).sum() / st["labelled_points"].sum())
    return out
