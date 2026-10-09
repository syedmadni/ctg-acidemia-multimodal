"""Clinically-motivated threshold selection + calibration diagnostics.

Threshold selection: none of the variants already in train_cv.py
(F1-optimal, Youden, target-FPR) are chosen on clinical grounds -- they're
statistical optima. For fetal acidemia screening, a missed positive (false
negative) risks permanent injury, while a false positive triggers closer
monitoring/expedited delivery -- a real but much lower-severity cost. That
asymmetry is the standard clinical justification for prioritizing sensitivity
over specificity in a screening/triage tool, so the threshold implemented
here is chosen to hit a target sensitivity (default 90%), not to optimize any
single blended statistic.

Held-out discipline: the CV harness's saved artifacts (oof_predictions.csv)
only contain outer-test-fold predictions -- validation-fold predictions used
for per-fold checkpoint selection were never persisted (see
train_deep_fold_by_val_pr_auc in train_cv.py, which discards val_proba after
selecting the best checkpoint). Retraining every cohort solely to persist
val_proba would cost hours of redundant GPU time for artifacts that already
exist. Instead, threshold calibration here uses a single stratified
calibration/evaluation split of the already-out-of-fold averaged OOF
predictions themselves (80% calibration / 20% evaluation, seed 42) -- the
threshold is chosen on the calibration slice only and applied to the
evaluation slice, so it is never chosen on the same data it's evaluated on.
This is a documented deviation from the project's per-fold nested calibration
standard (see CLAUDE.md's Validation Standard section) -- flag it as such
in any writeup, not as equivalent to a fresh nested-CV calibration.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from src.training.evaluate import evaluate_predictions


def calibration_evaluation_split(
    y: np.ndarray, seed: int = 42, calib_fraction: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    calib_idx, eval_idx = train_test_split(
        idx, train_size=calib_fraction, random_state=seed, stratify=y,
    )
    return calib_idx, eval_idx


def threshold_for_target_sensitivity(
    y_true: np.ndarray, y_proba: np.ndarray, target_sensitivity: float = 0.90
) -> float:
    """Highest threshold (most specific) that still achieves >= target_sensitivity
    on the given (calibration) data. Falls back to 0.0 (flag everything) if
    the target can't be reached at any threshold with this data."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    candidates = np.unique(np.concatenate([y_proba, np.linspace(0.0, 1.0, 101)]))
    best_threshold = 0.0
    best_specificity = -1.0
    for t in candidates:
        metrics = evaluate_predictions(y_true, y_proba, threshold=float(t))
        if metrics["recall_sensitivity"] >= target_sensitivity:
            if metrics["specificity"] > best_specificity:
                best_specificity = metrics["specificity"]
                best_threshold = float(t)
    return best_threshold


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(brier_score_loss(np.asarray(y_true, dtype=int), np.asarray(y_proba, dtype=float)))


def reliability_bins(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 5) -> Dict[str, np.ndarray]:
    """Quantile-binned reliability diagram data (fewer, quantile-based bins
    given the small n here -- fixed-width bins would leave many bins with 0-1
    records at n~300-425)."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return {"mean_predicted": prob_pred, "observed_frequency": prob_true}
