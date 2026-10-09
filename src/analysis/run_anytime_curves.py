"""ROC/PR curve figures for the any-elapsed-time cohort (primary_705_anytime_vr060).

run_anytime_cohort() writes oof_predictions.csv but never renders figures (same
gap save_quality_cohort_curves.py fills for the fixed-window cohorts). This is
the anytime counterpart, using the same repeat-averaged OOF predictions the
reported ROC-AUC point estimates come from.

Two figure groupings (more informative than one curve per (checkpoint, model)):
  1. figures/by_model/<model>_roc.png / _pr.png -- one figure per model, six
     curves (one per checkpoint) overlaid, showing how that model's curve
     shifts as more elapsed time becomes available.
  2. figures/by_checkpoint/<checkpoint>min_roc.png / _pr.png -- one figure per
     checkpoint, six curves (one per model) overlaid, showing model comparison
     at a fixed elapsed time.

Usage: python -m src.analysis.run_anytime_curves
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.training.train_cv import aggregate_repeated_oof_anytime

COHORT_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060")
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]
MODELS = ["ctg_only", "fusion", "gated_fusion", "maternal_only", "logistic_regression", "random_forest"]


def _plot_roc(ax, y_true, y_proba, label):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})")


def _plot_pr(ax, y_true, y_proba, label):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax.plot(recall, precision, label=f"{label} (AUC={auc(recall, precision):.3f})")


def main() -> None:
    oof = pd.read_csv(COHORT_DIR / "oof_predictions.csv")
    averaged = aggregate_repeated_oof_anytime(oof)

    by_model_dir = COHORT_DIR / "figures" / "by_model"
    by_checkpoint_dir = COHORT_DIR / "figures" / "by_checkpoint"
    by_model_dir.mkdir(parents=True, exist_ok=True)
    by_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. One figure per model, curves across checkpoints.
    for model in MODELS:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
        fig_pr, ax_pr = plt.subplots(figsize=(6, 6))
        for checkpoint in CHECKPOINTS:
            group = averaged[(averaged["model"] == model) & (averaged["checkpoint_minutes"] == checkpoint)]
            y = group["y_true"].astype(int).to_numpy()
            p = group["y_proba"].astype(float).to_numpy()
            _plot_roc(ax_roc, y, p, f"{checkpoint:g} min")
            _plot_pr(ax_pr, y, p, f"{checkpoint:g} min")

        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"ROC by checkpoint - {model}")
        ax_roc.legend(fontsize=8)
        fig_roc.tight_layout()
        fig_roc.savefig(by_model_dir / f"{model}_roc.png", dpi=200)
        plt.close(fig_roc)

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"PR by checkpoint - {model}")
        ax_pr.legend(fontsize=8)
        fig_pr.tight_layout()
        fig_pr.savefig(by_model_dir / f"{model}_pr.png", dpi=200)
        plt.close(fig_pr)

    # 2. One figure per checkpoint, curves across models.
    for checkpoint in CHECKPOINTS:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
        fig_pr, ax_pr = plt.subplots(figsize=(6, 6))
        for model in MODELS:
            group = averaged[(averaged["model"] == model) & (averaged["checkpoint_minutes"] == checkpoint)]
            y = group["y_true"].astype(int).to_numpy()
            p = group["y_proba"].astype(float).to_numpy()
            _plot_roc(ax_roc, y, p, model)
            _plot_pr(ax_pr, y, p, model)

        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"ROC across models - {checkpoint:g} min")
        ax_roc.legend(fontsize=8)
        fig_roc.tight_layout()
        fig_roc.savefig(by_checkpoint_dir / f"{checkpoint:g}min_roc.png", dpi=200)
        plt.close(fig_roc)

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"PR across models - {checkpoint:g} min")
        ax_pr.legend(fontsize=8)
        fig_pr.tight_layout()
        fig_pr.savefig(by_checkpoint_dir / f"{checkpoint:g}min_pr.png", dpi=200)
        plt.close(fig_pr)

    print(f"Saved {len(MODELS)*2} by-model figures to {by_model_dir}")
    print(f"Saved {len(CHECKPOINTS)*2} by-checkpoint figures to {by_checkpoint_dir}")


if __name__ == "__main__":
    main()
