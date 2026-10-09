from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    out = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) == 2 else float("nan"),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return out


def recall_at_fixed_fpr(y_true: np.ndarray, y_proba: np.ndarray, target_fpr: float) -> Dict[str, float]:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        idx = 0
    else:
        idx = valid[np.argmax(tpr[valid])]
    return {
        "target_fpr": float(target_fpr),
        "threshold": float(thresholds[idx]),
        "actual_fpr": float(fpr[idx]),
        "recall_sensitivity": float(tpr[idx]),
    }
