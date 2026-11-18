"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model.

Search objective : mean Macro F1 over 3-fold GroupKFold (grouped by trip) on
                   the 80 training trips. 3 folds keep each trial affordable.
Final comparison : untuned vs tuned, both re-scored with the same 5-fold
                   GroupKFold and on the 10 student-testing trips.

Output: results/hpo/<model>_trials.csv, results/hpo/<model>_search_space.json,
        results/hpo/<model>_untuned_vs_tuned.csv, results/plots/hpo_<model>_history.png,
        saved_models/<model>_tuned.joblib, saved_models/<model>_tuned_config.json
Run:    python -m hyperparameter_optimisation.bayes_opt random_forest 25
"""
import json
import sys
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from data_processing.load_data import ROOT
from feature_engineering.features import FEATURES
from model_evaluation.metrics import all_metrics
from model_training.train_models import MODELS, N_FOLDS, SEED, cross_validate, load_split

OUT = ROOT / "results" / "hpo"
SEARCH_FOLDS = 3

SEARCH_SPACES = {
    "random_forest": {
        "n_estimators": ("int", 50, 200),
        "max_depth": ("int", 6, 24),
        "min_samples_leaf": ("int", 5, 200, "log"),
        "max_features": ("float", 0.3, 1.0),
    },
    "decision_tree": {
        "max_depth": ("int", 3, 25),
        "min_samples_leaf": ("int", 5, 500, "log"),
        "criterion": ("cat", ["gini", "entropy"]),
    },
    "logistic_regression": {
        "C": ("float", 1e-3, 100.0, "log"),
    },
}


def suggest(trial, space):
    params = {}
    for k, spec in space.items():
        kind = spec[0]
        if kind == "int":
            params[k] = trial.suggest_int(k, spec[1], spec[2], log=len(spec) > 3)
        elif kind == "float":
            params[k] = trial.suggest_float(k, spec[1], spec[2], log=len(spec) > 3)
        else:
            params[k] = trial.suggest_categorical(k, spec[1])
    return params


def test_scores(model, df):
    t0 = time.time()
    prob = model.predict_proba(df[FEATURES].to_numpy())[:, 1]
    infer = time.time() - t0
    m = all_metrics(df, prob)
    return {f"test_{k}": v for k, v in m.items()} | {"test_inference_time_s": infer}


def main(name="random_forest", n_trials="25"):
    OUT.mkdir(parents=True, exist_ok=True)
    spec = MODELS[name]
    space = SEARCH_SPACES[name]
    train = load_split("train")
    X, y, groups = train[FEATURES].to_numpy(), train["label"].to_numpy(), train["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=SEARCH_FOLDS).split(X, y, groups))

    def objective(trial):
        params = {**spec["params"], **suggest(trial, space)}
        scores = []
        for tr, va in folds:
            model = spec["build"](params).fit(X[tr], y[tr])
            scores.append(all_metrics(train.iloc[va], model.predict_proba(X[va])[:, 1])["macro_f1"])
        trial.set_user_attr("fold_macro_f1", scores)
        return float(np.mean(scores))

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name=f"{name}_tpe")
    study.enqueue_trial({k: spec["params"][k] for k in space if k in spec["params"]})  # untuned baseline
    study.optimize(objective, n_trials=int(n_trials))
    search_s = time.time() - t0

    trials = study.trials_dataframe()
    trials.to_csv(OUT / f"{name}_trials.csv", index=False)
    (OUT / f"{name}_search_space.json").write_text(json.dumps(
        {"sampler": "TPE (Bayesian)", "objective": f"mean Macro F1, {SEARCH_FOLDS}-fold GroupKFold by trip",
         "n_trials": int(n_trials), "seed": SEED, "space": space, "search_time_s": round(search_s, 1)}, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4))
    vals = trials["value"].to_numpy()
    ax.plot(vals, "o", alpha=0.6, label="trial")
    ax.plot(np.maximum.accumulate(vals), "-", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel(f"CV Macro F1 ({SEARCH_FOLDS}-fold)")
    ax.set_title(f"Bayesian optimisation history: {name}")
    ax.legend()
    fig.tight_layout()
    (ROOT / "results" / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "plots" / f"hpo_{name}_history.png", dpi=130)

    best = {**spec["params"], **study.best_params}
    test = load_split("student_test")
    rows = []
    for label, params in [("untuned", spec["params"]), ("tuned", best)]:
        cv = pd.DataFrame(cross_validate(f"{name}_{label}", {**spec, "params": params}, train))
        model = spec["build"](params)
        t1 = time.time()
        model.fit(X, y)
        fit_s = time.time() - t1
        rows.append({"model": name, "version": label, "params": json.dumps(params),
                     "cv_macro_f1": cv["macro_f1"].mean(), "cv_macro_f1_std": cv["macro_f1"].std(),
                     "cv_top1_accuracy": cv["top1_accuracy"].mean(), "cv_log_loss": cv["log_loss"].mean(),
                     "fit_time_s": fit_s, **test_scores(model, test)})
        if label == "tuned":
            joblib.dump(model, ROOT / "saved_models" / f"{name}_tuned.joblib")
            (ROOT / "saved_models" / f"{name}_tuned_config.json").write_text(json.dumps(
                {"model": f"{name}_tuned", "family": spec["family"], "params": best, "seed": SEED,
                 "features": FEATURES, "tuned_by": "Optuna TPE", "cv": f"GroupKFold({N_FOLDS}) by trip"}, indent=2))
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / f"{name}_untuned_vs_tuned.csv", index=False)
    print(comp.drop(columns="params").round(4).T.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
