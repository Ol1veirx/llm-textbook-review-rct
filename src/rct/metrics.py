"""Confusion counts, performance metrics and per-participant tables."""
import numpy as np
import pandas as pd

from .data import ARMS

METRIC_COLUMNS = ["Responses", "Correct", "Precision", "Recall", "F1", "Accuracy", "Kappa"]


def confusion_counts(flagged, gold, universe):
    """TP, FP, FN, TN for a set of flagged sentences inside a universe of sentences."""
    flagged = set(flagged) & universe
    gold = gold & universe
    tp = len(flagged & gold)
    fp = len(flagged - gold)
    fn = len(gold) - tp
    tn = len(universe) - tp - fp - fn
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    """Precision, recall, F1, accuracy and Cohen's kappa. Undefined ratios are set to 0."""
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "Responses": tp + fp, "Correct": tp,
            "Precision": precision, "Recall": recall, "F1": f1, "Accuracy": accuracy, "Kappa": kappa}


def participant_metrics(study):
    """One row per participant with at least one record, computed inside their exposure window."""
    rows = []
    for r in study.exposure.itertuples():
        flagged = study.positives.get((r.arm, r.participant), set())
        window = study.universe_upto(r.last_page)
        counts = confusion_counts(flagged, study.gold, window)
        rows.append({"Arm": r.arm, "Participant": r.participant, "LastPage": r.last_page,
                     **metrics_from_counts(*counts)})
    return pd.DataFrame(rows).sort_values(["Arm", "Participant"]).reset_index(drop=True)


def to_itt(per_protocol, n_randomized):
    """Add the randomized participants who registered nothing.

    They enter with zero in every metric (accuracy included), so that each
    group keeps its randomized size (40 control, 38 experimental).
    """
    rows = []
    for arm in ARMS:
        missing = n_randomized[arm] - int((per_protocol["Arm"] == arm).sum())
        for i in range(missing):
            rows.append({"Arm": arm, "Participant": f"NO_RECORD_{arm[0]}{i + 1}", "LastPage": np.nan,
                         **{c: 0.0 for c in METRIC_COLUMNS}})
    return pd.concat([per_protocol, pd.DataFrame(rows)], ignore_index=True)


def tukey_outliers(values, k=1.5):
    """True for values outside Q1 - k*IQR and Q3 + k*IQR."""
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (values < q1 - k * iqr) | (values > q3 + k * iqr)


def drop_outliers(per_participant):
    """Remove outliers in total responses, found separately inside each group."""
    flag = per_participant.groupby("Arm", group_keys=False)["Responses"].apply(tukey_outliers)
    return per_participant[~flag].reset_index(drop=True), per_participant[flag]
