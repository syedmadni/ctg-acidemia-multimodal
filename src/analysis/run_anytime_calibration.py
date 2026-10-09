"""Clinical-threshold (target sensitivity) + calibration analysis for the
any-elapsed-time cohort (primary_705_anytime_vr060), across all 6 models
and all 6 elapsed-time checkpoints -- the anytime counterpart to
run_calibration_analysis.py. See src/analysis/calibration.py's module
docstring for why this uses a calibration/evaluation split of the averaged
OOF rather than per-fold nested calibration.

Usage: python -m src.analysis.run_anytime_calibration
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.analysis.calibration import (
    brier_score,
    calibration_evaluation_split,
    reliability_bins,
    threshold_for_target_sensitivity,
)
from src.training.evaluate import evaluate_predictions
from src.training.train_cv import aggregate_repeated_oof_anytime

COHORT_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060")
TARGET_SENSITIVITY = 0.90
SEED = 42
N_BINS = 5

MODELS = ["ctg_only", "fusion", "gated_fusion", "maternal_only", "logistic_regression", "random_forest"]
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]


def main() -> None:
    oof = pd.read_csv(COHORT_DIR / "oof_predictions.csv")
    averaged = aggregate_repeated_oof_anytime(oof)

    metric_rows: List[Dict[str, object]] = []
    bin_rows: List[Dict[str, object]] = []

    for checkpoint_minutes in CHECKPOINTS:
        for model in MODELS:
            print(f"checkpoint={checkpoint_minutes}min | {model}")
            group = averaged[(averaged["checkpoint_minutes"] == checkpoint_minutes) & (averaged["model"] == model)]
            y = group["y_true"].astype(int).to_numpy()
            p = group["y_proba"].astype(float).to_numpy()

            calib_idx, eval_idx = calibration_evaluation_split(y, seed=SEED, calib_fraction=0.8)
            threshold = threshold_for_target_sensitivity(y[calib_idx], p[calib_idx], TARGET_SENSITIVITY)
            eval_metrics = evaluate_predictions(y[eval_idx], p[eval_idx], threshold=threshold)

            metric_rows.append({
                "checkpoint_minutes": checkpoint_minutes,
                "model": model,
                "target_sensitivity": TARGET_SENSITIVITY,
                "clinical_threshold": threshold,
                "n_calib": len(calib_idx),
                "n_pos_calib": int(y[calib_idx].sum()),
                "n_eval": len(eval_idx),
                "n_pos_eval": int(y[eval_idx].sum()),
                "eval_sensitivity": eval_metrics["recall_sensitivity"],
                "eval_specificity": eval_metrics["specificity"],
                "eval_ppv_precision": eval_metrics["precision"],
                "eval_f1": eval_metrics["f1"],
                "eval_tp": eval_metrics["tp"],
                "eval_fp": eval_metrics["fp"],
                "eval_fn": eval_metrics["fn"],
                "eval_tn": eval_metrics["tn"],
                "brier_score_full_oof": brier_score(y, p),
                "n_full_oof": len(y),
            })

            bins = reliability_bins(y, p, n_bins=N_BINS)
            for mean_pred, obs_freq in zip(bins["mean_predicted"], bins["observed_frequency"]):
                bin_rows.append({
                    "checkpoint_minutes": checkpoint_minutes,
                    "model": model,
                    "mean_predicted_probability": mean_pred,
                    "observed_frequency": obs_freq,
                })

    metrics_df = pd.DataFrame(metric_rows).sort_values(["checkpoint_minutes", "model"])
    metrics_path = COHORT_DIR / "anytime_clinical_threshold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    bins_df = pd.DataFrame(bin_rows)
    bins_path = COHORT_DIR / "anytime_calibration_curve_bins.csv"
    bins_df.to_csv(bins_path, index=False)

    print(f"\nWrote {len(metrics_df)} rows to {metrics_path}")
    print(f"Wrote {len(bins_df)} reliability-bin rows to {bins_path}")
    print("\n=== Clinical-threshold (target sensitivity 90%) summary ===")
    print(metrics_df[[
        "checkpoint_minutes", "model", "clinical_threshold",
        "eval_sensitivity", "eval_specificity", "eval_ppv_precision", "eval_f1", "brier_score_full_oof",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
