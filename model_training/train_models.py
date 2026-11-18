"""Train and cross-validate candidate-scoring models on the 80 training trips.

Cross-validation: 5-fold GroupKFold grouped by trip, so no trip appears in
both the training and validation folds.

Output: saved_models/<model>.joblib, saved_models/<model>_config.json
        results/training/cv_folds.csv, results/training/cv_summary.csv
Run:    python -m model_training.train_models
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_processing.load_data import ROOT
from feature_engineering.features import FEATURES
from model_evaluation.metrics import all_metrics

SEED = 42
N_FOLDS = 5
MODEL_DIR = ROOT / "saved_models"
OUT = ROOT / "results" / "training"

MODELS = {
    "logistic_regression": {
        "family": "Linear / probabilistic",
        "params": {"C": 1.0, "max_iter": 1000},
        "build": lambda p: make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED, **p)),
    },
    "decision_tree": {
        "family": "Tree-based",
        "params": {"max_depth": 8, "min_samples_leaf": 50},
        "build": lambda p: DecisionTreeClassifier(random_state=SEED, **p),
    },
    "gaussian_naive_bayes": {
        "family": "Probabilistic",
        "params": {"var_smoothing": 1e-9},
        "build": lambda p: GaussianNB(**p),
    },
    "knn": {
        "family": "Instance-based",
        "params": {"n_neighbors": 25, "weights": "distance"},
        "build": lambda p: make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1, **p)),
    },
    "random_forest": {
        "family": "Ensemble (bagging)",
        "params": {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 20},
        "build": lambda p: RandomForestClassifier(random_state=SEED, n_jobs=-1, **p),
    },
    "lightgbm": {
        "family": "Boosting",
        "params": {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 50},
        "build": lambda p: LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **p),
    },
}


def load_split(split, labelled_only=True):
    """Candidate table for a split. Unlabelled points (no route arc nearby) are dropped."""
    df = pd.read_pickle(ROOT / "cache" / f"{split}_features.pkl")
    if labelled_only:
        df = df[df["ref_arc"] >= 0].reset_index(drop=True)
    return df


def cross_validate(name, spec, df):
    X, y, groups = df[FEATURES].to_numpy(), df["label"].to_numpy(), df["trip_id"].to_numpy()
    rows = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, groups)):
        model = spec["build"](spec["params"])
        t0 = time.time()
        model.fit(X[tr], y[tr])
        train_s = time.time() - t0
        t0 = time.time()
        prob = model.predict_proba(X[va])[:, 1]
        infer_s = time.time() - t0
        m = all_metrics(df.iloc[va], prob)
        # training-fold score (for over/underfitting checks) on 10 random training trips
        rng = np.random.default_rng(SEED + fold)
        sample_trips = rng.choice(np.unique(groups[tr]), size=10, replace=False)
        ts = tr[np.isin(groups[tr], sample_trips)]
        train_m = all_metrics(df.iloc[ts], model.predict_proba(X[ts])[:, 1])
        rows.append({"model": name, "fold": fold, **m,
                     "train_macro_f1": train_m["macro_f1"], "train_top1_accuracy": train_m["top1_accuracy"],
                     "train_time_s": train_s, "inference_time_s": infer_s})
        print(f"{name} fold {fold}: macroF1={m['macro_f1']:.4f} top1={m['top1_accuracy']:.4f} logloss={m['log_loss']:.4f}")
    return rows


def fit_final(name, spec, df):
    model = spec["build"](spec["params"])
    t0 = time.time()
    model.fit(df[FEATURES].to_numpy(), df["label"].to_numpy())
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    # real epoch-wise history, only for models that produce one (MLP)
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "loss_curve_"):
        hist = pd.DataFrame({"epoch": range(1, len(est.loss_curve_) + 1), "train_loss": est.loss_curve_})
        if getattr(est, "validation_scores_", None) is not None:
            hist["validation_accuracy"] = est.validation_scores_
        hist.to_csv(OUT / f"{name}_training_history.csv", index=False)
    config = {"model": name, "family": spec["family"], "params": spec["params"], "seed": SEED,
              "features": FEATURES, "cv": f"GroupKFold({N_FOLDS}) by trip",
              "train_trips": int(df["trip_id"].nunique()), "train_rows": len(df),
              "fit_time_s": round(time.time() - t0, 3)}
    (MODEL_DIR / f"{name}_config.json").write_text(json.dumps(config, indent=2))


def main(names=None):
    MODEL_DIR.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_split("train")
    print(f"train rows={len(df)} trips={df['trip_id'].nunique()} positive_rate={df['label'].mean():.4f}")

    rows = []
    for name, spec in MODELS.items():
        if names and name not in names:
            continue
        rows += cross_validate(name, spec, df)
        fit_final(name, spec, df)

    folds = pd.DataFrame(rows)
    # keep earlier results for models not retrained in this run
    old_file = OUT / "cv_folds.csv"
    if names and old_file.exists():
        old = pd.read_csv(old_file)
        folds = pd.concat([old[~old["model"].isin(folds["model"])], folds], ignore_index=True)
    folds.to_csv(OUT / "cv_folds.csv", index=False)
    summary = folds.drop(columns="fold").groupby("model").agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(OUT / "cv_summary.csv")
    print(summary[["macro_f1_mean", "macro_f1_std", "top1_accuracy_mean", "log_loss_mean"]].round(4))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
