"""Generate ROC/PR curve PNGs for a quality-cohort CV run (run_quality_cohorts
in train_cv.py writes oof_predictions.csv but never renders figures).

Uses the same repeat-averaged OOF predictions the reported ROC-AUC/PR-AUC
point estimates and bootstrap CIs are computed from, so the curves match the
numbers in cv_summary.json exactly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.training.evaluate import save_curves
from src.training.train_cv import aggregate_repeated_oof


def main() -> None:
    parser = argparse.ArgumentParser(description="Save ROC/PR curve figures for one or more quality-cohort runs.")
    parser.add_argument("--cohort-dir", action="append", required=True, help="e.g. outputs/quality_cohorts/primary_705_vr060 (repeatable)")
    args = parser.parse_args()

    for cohort_dir_str in args.cohort_dir:
        cohort_dir = Path(cohort_dir_str)
        oof_path = cohort_dir / "oof_predictions.csv"
        if not oof_path.exists():
            raise FileNotFoundError(f"Missing {oof_path}")

        oof = pd.read_csv(oof_path)
        averaged = aggregate_repeated_oof(oof)
        figures_dir = cohort_dir / "figures"

        for model, group in averaged.groupby("model"):
            save_curves(
                group["y_true"].astype(int).to_numpy(),
                group["y_proba"].astype(float).to_numpy(),
                figures_dir,
                prefix=str(model),
            )
        print(f"Saved ROC/PR curves for {sorted(averaged['model'].unique())} under {figures_dir}")


if __name__ == "__main__":
    main()
