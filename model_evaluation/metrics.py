"""Evaluation metrics for candidate scoring and route quality."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y_true, prob, threshold=0.5):
    """Row-level (candidate) classification metrics."""
    y_pred = (prob >= threshold).astype(int)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "log_loss": log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def top1_predictions(df, prob):
    """Pick the highest-scoring candidate arc for each GPS point."""
    d = df[["trip_id", "point", "arc", "label"]].copy()
    d["prob"] = prob
    idx = d.groupby(["trip_id", "point"])["prob"].idxmax()
    return d.loc[idx].sort_values(["trip_id", "point"])


def top1_accuracy(df, prob):
    """Share of GPS points whose top-scoring candidate is the labelled arc.

    Points whose labelled arc is not among the candidates count as wrong.
    """
    top = top1_predictions(df, prob)
    return float(top["label"].mean())


def route_continuity(pred_arcs, arcs):
    """Share of consecutive (changed) predicted arcs that are connected.

    Two arcs are connected when the first ends at the node where the second starts.
    """
    seq = np.asarray(pred_arcs)
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]  # collapse repeats
    if len(seq) < 2:
        return 1.0
    a = arcs.to_numpy()
    return float(np.mean(a[seq[:-1], 1] == a[seq[1:], 0]))


def all_metrics(df, prob):
    m = candidate_metrics(df["label"].to_numpy(), prob)
    m["top1_accuracy"] = top1_accuracy(df, prob)
    return m


def summarise(rows):
    return pd.DataFrame(rows)
