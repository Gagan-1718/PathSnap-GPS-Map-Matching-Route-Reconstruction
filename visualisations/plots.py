"""EDA and model-comparison plots, all generated from files in results/.

Output: results/plots/*.png
Run:    python -m visualisations.plots
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from data_processing.load_data import ROOT, prepare_trip
from feature_engineering.features import FEATURES
from model_training.train_models import load_split

RES = ROOT / "results"
OUT = RES / "plots"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def eda_plots():
    eda = pd.read_csv(RES / "eda" / "trip_summary.csv", dtype={"trip_id": str})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for split, g in eda.groupby("split"):
        ax[0].hist(g["gps_points"], bins=20, alpha=0.6, label=split)
        ax[1].hist(g["length_km"], bins=20, alpha=0.6, label=split)
        ax[2].hist(g["pct_dt_1s"], bins=20, alpha=0.6, label=split)
    for a, t in zip(ax, ["GPS points per trip", "Trip length (km)", "% of steps exactly 1 s apart"]):
        a.set_title(t)
        a.legend()
    save(fig, "eda_trip_distributions.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    g = eda[eda["split"] == "train"]
    ax.scatter(g["nodes"], g["arcs"], s=15)
    ax.set_xlabel("road nodes")
    ax.set_ylabel("road arcs")
    ax.set_title("Road-network size per trip (train)")
    save(fig, "eda_network_size.png")

    cov = pd.read_csv(RES / "features" / "train_candidate_stats.csv", dtype={"trip_id": str})
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(cov["trip_id"].str[-2:], cov["label_coverage"] * 100)
    ax.set_ylabel("% GPS points within 30 m of route")
    ax.set_title("Reference-route coverage per training trip (low = route covers only part of the track)")
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    save(fig, "eda_route_coverage.png")

    trip = prepare_trip("train", "00000005")
    xy, nxy = trip["track_xy"], trip["nodes_xy"]
    a = trip["arcs"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(len(a)):
        ax.plot(nxy[a[i], 0], nxy[a[i], 1], color="#cccccc", lw=0.6)
    for r in trip["route"]:
        ax.plot(nxy[a[r], 0], nxy[a[r], 1], color="#2ca02c", lw=2.5)
    ax.scatter(xy[:, 0], xy[:, 1], s=2, color="#1f77b4", label="GPS")
    ax.set_aspect("equal")
    ax.set_title("Train trip 00000005: road network, route (green), GPS (blue)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    save(fig, "eda_trip_00000005.png")


def bar(ax, labels, values, title, err=None, fmt="{:.3f}"):
    bars = ax.bar(labels, values, yerr=err, capsize=4, color="#4c72b0")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)


def model_plots():
    cv = pd.read_csv(RES / "training" / "cv_summary.csv")
    te = pd.read_csv(RES / "evaluation" / "student_test_metrics.csv")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    bar(ax[0], cv["model"], cv["macro_f1_mean"], "CV Macro F1 (5-fold GroupKFold)", cv["macro_f1_std"])
    bar(ax[1], te["model"], te["top1_accuracy"], "Student-test Top-1 accuracy")
    bar(ax[2], te["model"], te["macro_f1"], "Student-test Macro F1")
    save(fig, "model_comparison_f1_top1.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    bar(ax[0], cv["model"], cv["log_loss_mean"], "CV Log Loss (lower is better)", cv["log_loss_std"])
    bar(ax[1], te["model"], te["log_loss"], "Student-test Log Loss (lower is better)")
    save(fig, "model_comparison_logloss.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cv))
    ax.bar(x - 0.2, cv["train_macro_f1_mean"], 0.4, label="train folds")
    ax.bar(x + 0.2, cv["macro_f1_mean"], 0.4, label="validation folds")
    ax.set_xticks(x, cv["model"], rotation=25)
    ax.set_title("Train vs validation Macro F1 (over/underfitting check)")
    ax.legend()
    save(fig, "model_train_vs_validation.png")

    hmm_file = RES / "hmm" / "classifier_vs_hmm.csv"
    if hmm_file.exists():
        h = pd.read_csv(hmm_file)
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
        for a, col, title in zip(ax, ["top1_accuracy", "route_continuity", "route_f1"],
                                 ["Top-1 accuracy", "Route continuity", "Route F1 (arc overlap)"]):
            p = h.pivot(index="model", columns="method", values=col)
            p.plot.bar(ax=a, rot=25)
            a.set_title(f"Classifier only vs + HMM: {title}")
            a.set_ylim(0, 1)
        save(fig, "classifier_vs_hmm.png")

    df = load_split("student_test")
    models = te["model"].tolist()
    fig, ax = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    for a, name in zip(np.atleast_1d(ax), models):
        model = joblib.load(ROOT / "saved_models" / f"{name}.joblib")
        pred = (model.predict_proba(df[FEATURES].to_numpy())[:, 1] >= 0.5).astype(int)
        ConfusionMatrixDisplay.from_predictions(df["label"], pred, ax=a, colorbar=False,
                                                normalize="true", values_format=".2f")
        a.set_title(name, fontsize=9)
    save(fig, "confusion_matrices_student_test.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eda_plots()
    model_plots()
    print("plots saved to", OUT)


if __name__ == "__main__":
    main()
