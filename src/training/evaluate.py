from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, precision_recall_curve, roc_curve

from src.utils.metrics import binary_metrics, recall_at_fixed_fpr


def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    metrics = binary_metrics(y_true, y_proba, threshold=threshold)
    for fpr in [0.05, 0.10, 0.15, 0.20]:
        fixed = recall_at_fixed_fpr(y_true, y_proba, target_fpr=fpr)
        metrics[f"recall_at_fpr_{int(fpr * 100)}"] = fixed["recall_sensitivity"]
        metrics[f"threshold_at_fpr_{int(fpr * 100)}"] = fixed["threshold"]
    return metrics


def save_curves(y_true: np.ndarray, y_proba: np.ndarray, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {prefix}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_roc.png", dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {prefix}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_pr.png", dpi=200)
    plt.close()
