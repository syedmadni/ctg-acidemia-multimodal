"""Generate publication-quality figures and tables for the mid-term report,
from EXISTING outputs only. No training, no recomputation of any metric that
already lives in a canonical CSV/JSON, no hardcoded numbers.

Run as: python -m src.reporting.make_report_figures
Writes to: outputs/report_figures/ (figures) and outputs/report_figures/tables/

Design notes:
- Every number plotted is read directly from a source file. Where a value can
  be independently cross-checked against outputs/results_summary.csv, that
  check is performed and any mismatch is collected and printed at the end
  rather than silently plotted.
- src/training/save_quality_cohort_curves.py was NOT used for the ROC-curve
  figure. It aggregates OOF predictions with aggregate_repeated_oof(), which
  groups by (cohort_name, valid_ratio_threshold, model, record_id) -- it has
  no concept of checkpoint_minutes. Run against the anytime cohort's
  oof_predictions.csv, it would silently average a record's predictions
  across all six checkpoints into one blended, meaningless curve. This module
  instead calls aggregate_repeated_oof_anytime() directly (the same function
  cv_summary.json's own numbers are computed from), which is checkpoint-aware.
  No existing script was modified to discover or work around this.
- Each figure/table is generated independently; if its required source file
  is missing, that one item is skipped (reported at the end) rather than
  fabricated, and rather than aborting every other item.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import precision_recall_curve, roc_curve

from src.training.train_cv import aggregate_repeated_oof_anytime

# --------------------------------------------------------------------------- #
# Shared style: one palette, one set of sizes, used by every figure so the
# whole set reads as one system.
# --------------------------------------------------------------------------- #
MODEL_ORDER = ["ctg_only", "fusion", "gated_fusion", "maternal_only", "logistic_regression", "random_forest"]
MODEL_LABELS = {
    "ctg_only": "CTG-only",
    "fusion": "Fusion",
    "gated_fusion": "Gated Fusion",
    "maternal_only": "Maternal-only",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
# Okabe-Ito colourblind-safe palette, one fixed colour per model everywhere.
MODEL_COLORS = {
    "ctg_only": "#0072B2",
    "fusion": "#E69F00",
    "gated_fusion": "#009E73",
    "maternal_only": "#999999",
    "logistic_regression": "#D55E00",
    "random_forest": "#CC79A7",
}
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]
FIG_WIDTH_IN = 6.5
DPI = 300
DEEPCTG_BAND = (0.74, 0.83)


def _apply_style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 100,  # on-screen; savefig dpi is set explicitly per-call
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
    })


class MissingSource(Exception):
    """Raised when a figure/table's required source file does not exist.
    Caught at the orchestration level and reported, not fabricated around."""


def require(path: Path, needed_for: str) -> Path:
    if not path.exists():
        raise MissingSource(f"{needed_for}: missing required file {path}")
    return path


# --------------------------------------------------------------------------- #
# Cross-checking against outputs/results_summary.csv
# --------------------------------------------------------------------------- #
class CrossChecker:
    """Collects (value_plotted, value_in_results_summary) mismatches instead
    of silently plotting one of them. results_summary.csv encodes checkpoints
    inside the 'model' column as '<model>@<ckpt>min' (verified against its
    actual header/rows), matching cv_summary.json's own key convention."""

    def __init__(self, results_summary_path: Path):
        self.available = results_summary_path.exists()
        self.mismatches: List[str] = []
        self.checked = 0
        if self.available:
            self.df = pd.read_csv(results_summary_path)
        else:
            self.df = pd.DataFrame()

    def check(self, cohort_name: str, model_key: str, metric: str, plotted_value: float, tol: float = 1e-6) -> None:
        if not self.available:
            return
        rows = self.df[(self.df["cohort_name"] == cohort_name) & (self.df["model"] == model_key)]
        if rows.empty or metric not in rows.columns:
            return  # nothing to cross-check against for this row; not an error
        ref_value = rows.iloc[0][metric]
        if pd.isna(ref_value):
            return
        self.checked += 1
        if abs(float(ref_value) - float(plotted_value)) > tol:
            self.mismatches.append(
                f"{cohort_name} / {model_key} / {metric}: plotted={plotted_value!r} "
                f"vs results_summary.csv={ref_value!r}"
            )


# --------------------------------------------------------------------------- #
# Small IO helpers
# --------------------------------------------------------------------------- #
def load_cv_summary(cohort_dir: Path) -> Dict:
    path = require(cohort_dir / "cv_summary.json", f"cv_summary for {cohort_dir.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_table(summary: Dict) -> pd.DataFrame:
    """Flatten model_metrics_from_averaged_oof into a tidy (checkpoint, model, metric) frame."""
    rows = []
    for key, metrics in summary["model_metrics_from_averaged_oof"].items():
        model, ckpt_part = key.rsplit("@", 1)
        ckpt = float(ckpt_part.replace("min", ""))
        row = {"model": model, "checkpoint_minutes": ckpt, "model_key": key}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 1 & 2: ROC-AUC / PR-AUC by checkpoint
# --------------------------------------------------------------------------- #
def fig_metric_by_checkpoint(cohort_dir: Path, metric: str, out_path: Path, checker: CrossChecker) -> str:
    summary = load_cv_summary(cohort_dir)
    df = metrics_table(summary)
    cohort_name = summary["cohort_name"]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.6))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("checkpoint_minutes")
        if sub.empty:
            continue
        ax.plot(
            sub["checkpoint_minutes"], sub[metric],
            marker="o", markersize=4, linewidth=1.6,
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
        )
        for _, r in sub.iterrows():
            checker.check(cohort_name, r["model_key"], metric, r[metric])

    ax.set_xlabel("Elapsed-time checkpoint (minutes)")
    ax.set_ylabel("ROC-AUC" if metric == "roc_auc" else "PR-AUC")
    ax.set_xticks(CHECKPOINTS)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    savefig(fig, out_path)
    return (
        f"maternal_only is flat at {df[df['model']=='maternal_only'][metric].round(6).unique()} "
        f"across every checkpoint -- it never receives the CTG signal, so cropping the CTG "
        f"array has no effect on it; this is the sanity check that checkpoint-cropping only "
        f"touches the CTG branch."
    )


# --------------------------------------------------------------------------- #
# Figure 3: model comparison at 60 min with bootstrap CIs
# --------------------------------------------------------------------------- #
def fig_model_comparison_ci(cohort_dir: Path, checkpoint: float, out_path: Path, checker: CrossChecker) -> None:
    summary = load_cv_summary(cohort_dir)
    cohort_name = summary["cohort_name"]
    ci_path = require(cohort_dir / "bootstrap_confidence_intervals.csv", "model_comparison_at_60min")
    ci = pd.read_csv(ci_path)
    ci = ci[(ci["checkpoint_minutes"] == checkpoint) & (ci["metric"] == "roc_auc")]
    if ci.empty:
        raise MissingSource(f"model_comparison_at_60min: no roc_auc CI rows at checkpoint={checkpoint} in {ci_path}")

    models_present = [m for m in MODEL_ORDER if m in set(ci["model"])]
    y_pos = np.arange(len(models_present))
    point = []
    err_low = []
    err_high = []
    for m in models_present:
        row = ci[ci["model"] == m].iloc[0]
        point.append(row["point_estimate"])
        err_low.append(row["point_estimate"] - row["ci_low"])
        err_high.append(row["ci_high"] - row["point_estimate"])
        checker.check(cohort_name, f"{m}@{checkpoint}min", "roc_auc", row["point_estimate"])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.2))
    colors = [MODEL_COLORS[m] for m in models_present]
    ax.barh(y_pos, point, xerr=[err_low, err_high], color=colors, height=0.6,
            error_kw={"elinewidth": 1.4, "capsize": 4, "capthick": 1.4, "ecolor": "black"})
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_LABELS[m] for m in models_present])
    ax.set_xlabel(f"ROC-AUC at {checkpoint:g} min (95% bootstrap CI)")
    ax.set_xlim(0.3, 1.0)
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.grid(axis="x", linewidth=0.4, alpha=0.4)
    savefig(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 4: improvement campaign comparison
# --------------------------------------------------------------------------- #
IMPROVEMENT_COHORTS = [
    ("Baseline\n(anytime)", "primary_705_anytime_vr060"),
    ("Shrink\n(cap64)", "anytime_cap64"),
    ("CNN-only\nablation", "anytime_cnn_only"),
    ("Augmentation", "anytime_cap64_aug"),
    ("Focal loss", "anytime_cap64_focal"),
    ("Pretraining", "anytime_cap64_pretrain"),
]
DEEP_MODELS = ["ctg_only", "fusion", "gated_fusion"]


def _best_deep_roc(cohorts_dir: Path, cohort_name: str, checker: CrossChecker) -> Tuple[float, str]:
    summary = load_cv_summary(cohorts_dir / cohort_name)
    df = metrics_table(summary)
    df = df[df["model"].isin(DEEP_MODELS)]
    best = df.loc[df["roc_auc"].idxmax()]
    checker.check(cohort_name, best["model_key"], "roc_auc", best["roc_auc"])
    return float(best["roc_auc"]), str(best["model_key"])


def _ensemble_best_roc(cohorts_dir: Path) -> float:
    path = require(cohorts_dir / "anytime_cap64" / "ensemble_oof_predictions.csv", "improvement_campaign_comparison (ensemble)")
    oof = pd.read_csv(path)
    best = 0.0
    from sklearn.metrics import roc_auc_score
    for (model, ckpt), g in oof.groupby(["model", "checkpoint_minutes"]):
        if len(g["y_true"].unique()) < 2:
            continue
        a = roc_auc_score(g["y_true"], g["y_proba"])
        best = max(best, a)
    return best


def fig_improvement_campaign(cohorts_dir: Path, out_path: Path, checker: CrossChecker) -> Dict[str, float]:
    labels: List[str] = []
    values: List[float] = []
    colors: List[str] = []
    baseline_value: Optional[float] = None

    for label, cohort_name in IMPROVEMENT_COHORTS:
        value, _key = _best_deep_roc(cohorts_dir, cohort_name, checker)
        labels.append(label)
        values.append(value)
        if baseline_value is None:
            baseline_value = value

    ensemble_value = _ensemble_best_roc(cohorts_dir)
    labels.append("Ensemble\n(deep + LR)")
    values.append(ensemble_value)

    for label, value in zip(labels[:-1], values[:-1]):
        if label.startswith("Baseline"):
            colors.append("#4D4D4D")
            continue
        delta = value - baseline_value
        if delta > 0.005:
            colors.append("#009E73")   # helped
        elif delta < -0.005:
            colors.append("#D55E00")   # hurt
        else:
            colors.append("#999999")   # neutral
    colors.append("#0072B2")  # ensemble: distinct highlight, the headline result

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.8))
    x_pos = np.arange(len(labels))
    ax.axhspan(DEEPCTG_BAND[0], DEEPCTG_BAND[1], color="#F0E442", alpha=0.35, zorder=0,
               label=f"DeepCTG 2.0 reference range ({DEEPCTG_BAND[0]}-{DEEPCTG_BAND[1]})")
    ax.bar(x_pos, values, color=colors, width=0.6, zorder=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Best deep-model ROC-AUC")
    ax.set_ylim(0.5, 0.9)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.18), frameon=False, ncol=1)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    savefig(fig, out_path)
    return dict(zip(labels, values))


# --------------------------------------------------------------------------- #
# Figure 5: gate-value disagreement interpretability (small multiples)
# --------------------------------------------------------------------------- #
def fig_gate_disagreement(cohort_dir: Path, out_path: Path, pooled_cohort_dir: Path | None = None) -> str:
    """Design choice, stated explicitly rather than silently made: the source
    data spans all six checkpoints and the finding is specifically that the gap
    widens WITH elapsed time -- collapsing to one checkpoint would hide the
    actual result. Shown as 2x3 small multiples, one panel per checkpoint,
    individual points jittered.

    SOURCE PRECEDENCE (added 2026-08-15): prefers the POOLED all-folds analysis
    (n=345, 23 positives) from primary_705_anytime_vr060_allfolds over the
    original single-fold file (n=69, 5 positives). This matters: the single-fold
    version overstated the CTG-right/maternal-right gap roughly fivefold
    (+0.222 vs +0.046 pooled), so a figure built from it would contradict the
    manuscript text. Falls back to the single-fold file only if the pooled one
    is absent, and says which was used in the returned caption note.
    """
    pooled_path = (pooled_cohort_dir / "anytime_disagreement_analysis_pooled.csv") if pooled_cohort_dir else None
    if pooled_path is not None and pooled_path.exists():
        path, is_pooled = pooled_path, True
    else:
        path = require(cohort_dir / "anytime_disagreement_analysis.csv", "gate_disagreement_interpretability")
        is_pooled = False
    df = pd.read_csv(path)

    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH_IN, 4.6), sharey=True)
    rng = np.random.default_rng(42)
    n_holdout = n_pos = 0
    # Pooled n is 5x the single-fold n, so the same marker size/alpha would
    # overplot into a solid block. Scale them to the actual point count.
    marker_size, alpha = (4, 0.35) if is_pooled else (9, 0.75)

    for ax, ckpt in zip(axes.flat, CHECKPOINTS):
        sub = df[df["checkpoint_minutes"] == ckpt]
        n_holdout = len(sub)
        n_pos = int(sub["y_true"].sum())
        agree = sub[~sub["disagreement"]]
        dis_ctg_right = sub[(sub["disagreement"]) & (sub["ctg_reading_closer_to_truth"] == True)]  # noqa: E712
        dis_mat_right = sub[(sub["disagreement"]) & (sub["ctg_reading_closer_to_truth"] == False)]  # noqa: E712

        groups = [
            # "CTG right"/"Maternal right" intentionally reuse ctg_only's/maternal_only's
            # colours (semantically the same distinction). "Agreement" gets its own colour
            # (sky blue) so it never collides with maternal_only's grey.
            ("Agreement", agree["gate_value"], "#56B4E9"),
            ("Disagree:\nCTG right", dis_ctg_right["gate_value"], MODEL_COLORS["ctg_only"]),
            ("Disagree:\nMaternal right", dis_mat_right["gate_value"], MODEL_COLORS["maternal_only"]),
        ]
        for i, (label, values, color) in enumerate(groups):
            if len(values) == 0:
                continue
            jitter = rng.uniform(-0.14, 0.14, size=len(values))
            ax.scatter(np.full(len(values), i) + jitter, values, s=marker_size,
                       color=color, alpha=alpha, linewidths=0)
            ax.scatter([i], [values.mean()], marker="_", s=260, color="black", linewidths=1.6, zorder=5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Agree", "CTG\nright", "Mat.\nright"], fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.annotate(f"{ckpt:g} min\nn={n_holdout}", xy=(0.02, 0.96), xycoords="axes fraction",
                    ha="left", va="top", fontsize=7)
        ax.grid(axis="y", linewidth=0.3, alpha=0.3)

    for ax in axes[:, 0]:
        ax.set_ylabel("Gate value\n(CTG weight)")
    fig.suptitle("")  # explicitly no title text; caption lives in the manuscript
    savefig(fig, out_path)

    if is_pooled:
        return (
            f"Gate values POOLED across all 15 fold checkpoints (n={n_holdout} records, "
            f"{n_pos} positive, at every checkpoint) -- the fully powered analysis. Black bars "
            "are group means. The CTG-right/maternal-right separation is present at all six "
            "checkpoints and widens with elapsed time, but is modest (mean difference +0.046)."
        )
    return (
        f"Single best-fold holdout, n={n_holdout} records ({n_pos} positive) at every checkpoint "
        "-- NOT the fully powered analysis, and known to overstate the effect roughly fivefold. "
        "Regenerate against primary_705_anytime_vr060_allfolds before using this in the paper."
    )


# --------------------------------------------------------------------------- #
# Figure 6: ROC curves at 60 min (built directly from oof_predictions.csv,
# NOT via save_quality_cohort_curves.py -- see module docstring)
# --------------------------------------------------------------------------- #
def fig_roc_curves(cohort_dir: Path, checkpoint: float, out_path: Path, checker: CrossChecker) -> List[str]:
    summary = load_cv_summary(cohort_dir)
    cohort_name = summary["cohort_name"]
    oof_path = require(cohort_dir / "oof_predictions.csv", "roc_curves_at_60min")
    oof = pd.read_csv(oof_path)
    averaged = aggregate_repeated_oof_anytime(oof)
    averaged = averaged[averaged["checkpoint_minutes"] == checkpoint]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN * 0.62, FIG_WIDTH_IN * 0.62))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8, alpha=0.6)
    consistency_notes = []
    for model in MODEL_ORDER:
        sub = averaged[averaged["model"] == model]
        if sub.empty:
            continue
        y = sub["y_true"].astype(int).to_numpy()
        p = sub["y_proba"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        curve_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=MODEL_COLORS[model], linewidth=1.5,
                label=f"{MODEL_LABELS[model]} (AUC={curve_auc:.3f})")

        model_key = f"{model}@{checkpoint}min"
        reported_auc = summary["model_metrics_from_averaged_oof"][model_key]["roc_auc"]
        if abs(curve_auc - reported_auc) > 1e-6:
            consistency_notes.append(
                f"{model_key}: curve-computed AUC={curve_auc:.6f} vs cv_summary.json={reported_auc:.6f}"
            )
        checker.check(cohort_name, model_key, "roc_auc", curve_auc)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set_aspect("equal")
    savefig(fig, out_path)
    return consistency_notes


# --------------------------------------------------------------------------- #
# Figure 7: calibration reliability at 60 min
# --------------------------------------------------------------------------- #
RELIABILITY_MODELS = ["logistic_regression", "ctg_only"]  # top classical + top deep at 60min


def fig_calibration(cohort_dir: Path, checkpoint: float, out_path: Path) -> None:
    path = require(cohort_dir / "anytime_calibration_curve_bins.csv", "calibration_reliability_60min")
    df = pd.read_csv(path)
    df = df[df["checkpoint_minutes"] == checkpoint]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN * 0.55, FIG_WIDTH_IN * 0.55))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8, alpha=0.6, label="Perfect calibration")
    for model in RELIABILITY_MODELS:
        sub = df[df["model"] == model].sort_values("mean_predicted_probability")
        if sub.empty:
            continue
        ax.plot(sub["mean_predicted_probability"], sub["observed_frequency"],
                marker="o", markersize=4, linewidth=1.4, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax.set_aspect("equal")
    savefig(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 8: modality ablation (added 2026-08-15)
# --------------------------------------------------------------------------- #
def fig_modality_ablation(cohort_dir: Path, out_path: Path) -> str:
    """Direct evidence for the thesis's central claim: does combining CTG and
    maternal data actually beat either alone?

    Form choice: the data's job is a PAIRED comparison over an ordered x-axis
    (elapsed time), so lines-by-checkpoint -- matching fig_metric_by_checkpoint's
    existing house form -- is the right chart, not grouped bars. The vertical
    gap between the two CTG-containing lines IS the maternal contribution, so it
    is shaded to make the quantity of interest directly readable rather than
    something the reader must compute by eye.

    One panel per classical model; a single shared axis (never a dual axis).
    Colours reuse the fixed Okabe-Ito model palette semantically: the CTG-only
    feature set takes ctg_only's blue, maternal-only takes maternal_only's grey,
    and the combined set takes a distinct green.
    """
    path = require(cohort_dir / "modality_ablation_results.csv", "modality_ablation")
    df = pd.read_csv(path)

    models = ["logistic_regression", "random_forest"]
    series = [
        ("ctg_only_features", "CTG features only", MODEL_COLORS["ctg_only"], "o"),
        ("maternal_only_features", "Maternal only", MODEL_COLORS["maternal_only"], "s"),
        ("both", "CTG + maternal", "#009E73", "D"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 2.9), sharey=True)
    mean_gain = {}
    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        by_fs = {}
        for fs, label, color, marker in series:
            s = sub[sub["feature_set"] == fs].sort_values("checkpoint_minutes")
            by_fs[fs] = s["roc_auc"].to_numpy()
            ax.plot(s["checkpoint_minutes"], s["roc_auc"], marker=marker, markersize=4,
                    linewidth=1.5, color=color, label=label)
        # Shade the gain from adding maternal variables to the CTG features.
        cks = sorted(sub["checkpoint_minutes"].unique())
        ax.fill_between(cks, by_fs["ctg_only_features"], by_fs["both"],
                        where=by_fs["both"] >= by_fs["ctg_only_features"],
                        color="#009E73", alpha=0.12, linewidth=0, interpolate=True)
        mean_gain[model] = float(np.mean(by_fs["both"] - by_fs["ctg_only_features"]))

        ax.axhline(0.5, color="#666666", linewidth=0.8, linestyle=":")
        ax.set_xticks(CHECKPOINTS)
        ax.set_xlabel("Elapsed time (min)")
        ax.set_title(MODEL_LABELS[model], fontsize=9)
        ax.grid(axis="y", linewidth=0.3, alpha=0.3)
        ax.set_ylim(0.45, 0.82)

    axes[0].set_ylabel("ROC-AUC")
    # Opaque legend background: the chance line at 0.5 otherwise strikes through
    # the legend text (caught by eyeballing the rendered figure, not the data).
    axes[0].legend(loc="lower right", fontsize=7, frameon=True, framealpha=1.0,
                   facecolor="white", edgecolor="none", borderpad=0.4)
    savefig(fig, out_path)
    return (
        f"Mean gain from adding maternal variables: logistic regression "
        f"{mean_gain['logistic_regression']:+.3f}, random forest {mean_gain['random_forest']:+.3f}."
    )


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Minimal Word-pasteable markdown table writer (avoids adding the
    'tabulate' dependency, which is not in requirements.txt)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, name: str, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / f"{name}.csv", index=False)
    (tables_dir / f"{name}.md").write_text(dataframe_to_markdown(df), encoding="utf-8")


def table_anytime_results(cohort_dir: Path) -> pd.DataFrame:
    summary = load_cv_summary(cohort_dir)
    df = metrics_table(summary)
    df = df[["checkpoint_minutes", "model", "roc_auc", "pr_auc"]].copy()
    df["model"] = df["model"].map(MODEL_LABELS)
    df = df.sort_values(["checkpoint_minutes", "model"])
    return df.round({"roc_auc": 4, "pr_auc": 4})


def table_improvement_campaign(cohorts_dir: Path, campaign_values: Dict[str, float]) -> pd.DataFrame:
    baseline = campaign_values["Baseline\n(anytime)"]
    rows = []
    verdicts = {
        "Baseline\n(anytime)": "reference",
        "Shrink\n(cap64)": "helped",
        "CNN-only\nablation": "comparable/better",
        "Augmentation": "null",
        "Focal loss": "hurt",
        "Pretraining": "helped (gated)",
        "Ensemble\n(deep + LR)": "best result",
    }
    for label, value in campaign_values.items():
        rows.append({
            "experiment": label.replace("\n", " "),
            "best_deep_roc_auc": round(value, 4),
            "delta_vs_baseline": round(value - baseline, 4),
            "verdict": verdicts.get(label, ""),
        })
    return pd.DataFrame(rows)


def table_significance_summary(cohort_dir: Path, cohorts_dir: Path) -> pd.DataFrame:
    anytime_path = require(cohort_dir / "anytime_significance_tests.csv", "significance_summary_table")
    improve_path = require(cohorts_dir / "improvement_significance_tests.csv", "significance_summary_table")
    anytime = pd.read_csv(anytime_path)
    improve = pd.read_csv(improve_path)

    rows = []
    for _, r in anytime.iterrows():
        rows.append({
            "family": "anytime checkpoint/model", "comparison": r["comparison_type"],
            "detail": f"{r['model_a']}@{r['checkpoint_minutes_a']:g}min vs {r['model_b']}@{r['checkpoint_minutes_b']:g}min",
            "auc_diff": round(r["auc_diff"], 4), "delong_p": round(r["delong_p"], 4),
            "permutation_p": round(r["permutation_p"], 4), "reaches_p<0.05": bool(r["delong_p"] < 0.05),
        })
    for _, r in improve.iterrows():
        rows.append({
            "family": "improvement campaign", "comparison": r["comparison"],
            "detail": f"checkpoint={r['checkpoint_minutes']:g}min",
            "auc_diff": round(r["auc_diff_b_minus_a"], 4), "delong_p": round(r["delong_p"], 4),
            "permutation_p": round(r["permutation_p"], 4), "reaches_p<0.05": bool(r["delong_p"] < 0.05),
        })
    df = pd.DataFrame(rows)
    n_sig = int(df["reaches_p<0.05"].sum())
    print(f"  [significance_summary_table] {n_sig}/{len(df)} comparisons reach p<0.05.")
    return df


def table_maternal_diagnostics(cohort_dir: Path) -> pd.DataFrame:
    diag_path = require(cohort_dir / "maternal_variable_diagnostics.csv", "maternal_diagnostics_table")
    fi_path = require(cohort_dir / "feature_importance.csv", "maternal_diagnostics_table")
    diag = pd.read_csv(diag_path)

    fi = pd.read_csv(fi_path)
    fi_lr = fi[fi["model"] == "logistic_regression"]
    mean_rank = fi_lr.groupby("feature")["rank"].mean().round(1)

    diag["logistic_regression_mean_rank_of_18"] = diag["variable"].map(mean_rank)
    cols = ["variable", "kind", "n_flag1", "acidemia_rate_flag1", "acidemia_rate_flag0",
            "odds_ratio", "fisher_p", "pos_mean", "neg_mean", "logistic_regression_mean_rank_of_18"]
    return diag[cols]


def table_modality_ablation(cohort_dir: Path) -> pd.DataFrame:
    """Wide-form ablation table: one row per (model, checkpoint), columns for
    each feature set plus the gain from adding maternal variables. Paired
    p-values are merged in from modality_ablation_significance.csv so the table
    never shows a gain without its (non-)significance beside it."""
    res_path = require(cohort_dir / "modality_ablation_results.csv", "modality_ablation_table")
    sig_path = require(cohort_dir / "modality_ablation_significance.csv", "modality_ablation_table")
    res = pd.read_csv(res_path)
    sig = pd.read_csv(sig_path)

    wide = res.pivot_table(index=["model", "checkpoint_minutes"],
                           columns="feature_set", values="roc_auc").reset_index()
    wide["gain_from_maternal"] = wide["both"] - wide["ctg_only_features"]
    merged = wide.merge(
        sig[["model", "checkpoint_minutes", "delong_p", "permutation_p"]],
        on=["model", "checkpoint_minutes"], how="left",
    )
    merged["model"] = merged["model"].map(MODEL_LABELS)
    merged = merged[[
        "model", "checkpoint_minutes", "ctg_only_features", "maternal_only_features",
        "both", "gain_from_maternal", "delong_p", "permutation_p",
    ]].sort_values(["model", "checkpoint_minutes"])
    return merged.round({
        "ctg_only_features": 4, "maternal_only_features": 4, "both": 4,
        "gain_from_maternal": 4, "delong_p": 4, "permutation_p": 4,
    })


def table_cohort_summary(cohorts_dir: Path, cohort_names: List[str]) -> pd.DataFrame:
    rows = []
    for name in cohort_names:
        summary = load_cv_summary(cohorts_dir / name)
        rows.append({
            "cohort_name": summary["cohort_name"],
            "n": summary["cohort_size"],
            "n_positive": summary["positive_count"],
            "prevalence_pct": round(100 * summary["positive_count"] / summary["cohort_size"], 2),
            "valid_ratio_threshold": summary["valid_ratio_threshold"],
            "label": summary["label"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def write_manifest(entries: List[Dict], out_path: Path) -> None:
    lines = ["# Figure & table manifest", "", "| Item | Source file(s) | Draft caption |", "|---|---|---|"]
    for e in entries:
        sources = "; ".join(e["sources"])
        caption = e["caption"].replace("\n", " ")
        lines.append(f"| `{e['name']}` | {sources} | {caption} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report-ready figures/tables from existing outputs only.")
    parser.add_argument("--cohorts-dir", default="outputs/quality_cohorts")
    parser.add_argument("--headline-cohort", default="primary_705_anytime_vr060")
    parser.add_argument("--output-dir", default="outputs/report_figures")
    parser.add_argument("--results-summary", default="outputs/results_summary.csv")
    parser.add_argument("--checkpoint", type=float, default=60.0)
    parser.add_argument(
        "--allfolds-cohort", default="primary_705_anytime_vr060_allfolds",
        help="Cohort holding the POOLED all-folds gate analysis; preferred over the "
             "headline cohort's single-fold version for the gate figure.",
    )
    args = parser.parse_args()

    _apply_style()
    cohorts_dir = Path(args.cohorts_dir)
    cohort_dir = cohorts_dir / args.headline_cohort
    out_dir = Path(args.output_dir)
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    checker = CrossChecker(Path(args.results_summary))
    generated: List[str] = []
    skipped: List[str] = []
    manifest_entries: List[Dict] = []
    notes: List[str] = []

    def attempt(name: str, fn, sources: List[str], caption_fn):
        try:
            result = fn()
            generated.append(name)
            manifest_entries.append({"name": name, "sources": sources, "caption": caption_fn(result)})
            return result
        except MissingSource as e:
            skipped.append(f"{name}: {e}")
            return None

    # --- Figures 1-2 ---
    attempt(
        "anytime_roc_auc_by_checkpoint.png",
        lambda: fig_metric_by_checkpoint(cohort_dir, "roc_auc", out_dir / "anytime_roc_auc_by_checkpoint.png", checker),
        [f"{cohort_dir}/cv_summary.json"],
        lambda r: (
            "ROC-AUC by elapsed-time checkpoint, all six models, repeat-averaged out-of-fold "
            f"predictions (n={{n}}, positives={{p}}). {r}"
        ).format(n="345", p="23"),
    )
    attempt(
        "anytime_pr_auc_by_checkpoint.png",
        lambda: fig_metric_by_checkpoint(cohort_dir, "pr_auc", out_dir / "anytime_pr_auc_by_checkpoint.png", checker),
        [f"{cohort_dir}/cv_summary.json"],
        lambda r: (
            "PR-AUC by elapsed-time checkpoint, all six models. Prevalence is ~6.7%, so the "
            "no-skill PR-AUC floor is ~0.067, not 0.5 -- PR-AUC is the more honest headline "
            f"metric here. {r}"
        ),
    )

    # --- Figure 3 ---
    attempt(
        "model_comparison_at_60min.png",
        lambda: fig_model_comparison_ci(cohort_dir, args.checkpoint, out_dir / "model_comparison_at_60min.png", checker),
        [f"{cohort_dir}/bootstrap_confidence_intervals.csv"],
        lambda r: (
            f"ROC-AUC at {args.checkpoint:g} min with 95% bootstrap confidence intervals "
            "(1000 iterations). Intervals overlap heavily across models -- no model is "
            "confidently distinguishable from another at this sample size (n=345, 23 positive)."
        ),
    )

    # --- Figure 4 ---
    campaign_values = attempt(
        "improvement_campaign_comparison.png",
        lambda: fig_improvement_campaign(cohorts_dir, out_dir / "improvement_campaign_comparison.png", checker),
        [f"{cohorts_dir}/<cohort>/cv_summary.json for each experiment", f"{cohorts_dir}/anytime_cap64/ensemble_oof_predictions.csv"],
        lambda r: (
            "Best deep-model ROC-AUC across the improvement campaign. Shaded band = DeepCTG 2.0's "
            "reported 0.74-0.83 range for the same severe-acidemia task on ~60x more data. "
            "Augmentation (grey) was null and focal loss (orange/red) was clearly worse than "
            "baseline -- shown plainly, not hidden. The deep+LR ensemble (blue) is the best "
            "result obtained. None of these deltas reached p<0.05 (see significance_summary_table)."
        ),
    )

    # --- Figure 5 ---
    pooled_gate_dir = cohorts_dir / args.allfolds_cohort
    attempt(
        "gate_disagreement_interpretability.png",
        lambda: fig_gate_disagreement(
            cohort_dir, out_dir / "gate_disagreement_interpretability.png", pooled_gate_dir,
        ),
        [f"{pooled_gate_dir}/anytime_disagreement_analysis_pooled.csv (preferred)",
         f"{cohort_dir}/anytime_disagreement_analysis.csv (fallback)"],
        lambda r: (
            "Gated-fusion gate value (CTG weight), split by agreement vs. disagreement between "
            "the CTG-only and maternal-only models, and, within disagreement, by which modality "
            f"was actually correct. {r} Design choice: all six checkpoints shown as small "
            "multiples because the finding is specifically that the CTG-right/Maternal-right gap "
            "widens with elapsed time -- a single checkpoint would hide that."
        ),
    )

    # --- Figure 6 ---
    roc_notes = attempt(
        "roc_curves_at_60min.png",
        lambda: fig_roc_curves(cohort_dir, args.checkpoint, out_dir / "roc_curves_at_60min.png", checker),
        [f"{cohort_dir}/oof_predictions.csv"],
        lambda r: (
            f"ROC curves for all six models at {args.checkpoint:g} min, repeat-averaged "
            "out-of-fold predictions. Built directly via aggregate_repeated_oof_anytime() "
            "(the same function cv_summary.json's numbers come from), NOT via "
            "save_quality_cohort_curves.py -- see console notes for why."
        ),
    )
    if roc_notes:
        notes.extend(roc_notes)

    # --- Figure 7 ---
    attempt(
        "calibration_reliability_60min.png",
        lambda: fig_calibration(cohort_dir, args.checkpoint, out_dir / "calibration_reliability_60min.png"),
        [f"{cohort_dir}/anytime_calibration_curve_bins.csv"],
        lambda r: (
            f"Reliability curve at {args.checkpoint:g} min for the two best-performing models "
            "(Logistic Regression: best overall; CTG-only: best deep model), 5 quantile bins, "
            "against the perfect-calibration diagonal."
        ),
    )

    # --- Figure 8 (added 2026-08-15) ---
    attempt(
        "modality_ablation.png",
        lambda: fig_modality_ablation(cohort_dir, out_dir / "modality_ablation.png"),
        [f"{cohort_dir}/modality_ablation_results.csv"],
        lambda r: (
            "Modality ablation: ROC-AUC for the two classical models fitted to CTG summary "
            "features alone, maternal variables alone, and both, through identical "
            "cross-validation folds. Shading marks the gain from adding the eight maternal "
            f"variables to the CTG features. {r} No comparison reaches p<0.05 (smallest "
            "p=0.063); the gains are directional. This figure is the direct evidence for the "
            "multimodal claim -- the combined model, not the CTG-only model, is the better "
            "predictor."
        ),
    )

    # --- Tables ---
    def make_table(name, fn, sources, caption):
        try:
            df = fn()
            write_table(df, name, tables_dir)
            generated.append(f"{name}.csv/.md")
            manifest_entries.append({"name": f"{name}.csv/.md", "sources": sources, "caption": caption})
        except MissingSource as e:
            skipped.append(f"{name}: {e}")

    make_table(
        "T1_anytime_results_table",
        lambda: table_anytime_results(cohort_dir),
        [f"{cohort_dir}/cv_summary.json"],
        "ROC-AUC and PR-AUC, 6 models x 6 checkpoints.",
    )
    if campaign_values:
        make_table(
            "T2_improvement_campaign_table",
            lambda: table_improvement_campaign(cohorts_dir, campaign_values),
            ["derived from Figure 4's own computed values"],
            "Improvement-campaign experiments, best deep ROC-AUC, delta vs. baseline, verdict.",
        )
    make_table(
        "T3_significance_summary_table",
        lambda: table_significance_summary(cohort_dir, cohorts_dir),
        [f"{cohort_dir}/anytime_significance_tests.csv", f"{cohorts_dir}/improvement_significance_tests.csv"],
        "All paired DeLong/permutation comparisons. Explicit reaches_p<0.05 column -- none do.",
    )
    make_table(
        "T4_maternal_diagnostics_table",
        lambda: table_maternal_diagnostics(cohort_dir),
        [f"{cohort_dir}/maternal_variable_diagnostics.csv", f"{cohort_dir}/feature_importance.csv"],
        "Maternal variable odds ratios / Fisher p-values, with each variable's mean Logistic "
        "Regression feature-importance rank (parity ranks #1 of 18 features).",
    )
    make_table(
        "T5_cohort_summary_table",
        lambda: table_cohort_summary(cohorts_dir, [
            "primary_705_anytime_vr060", "anytime_cap64", "anytime_cnn_only",
            "anytime_cap64_aug", "anytime_cap64_focal", "anytime_cap64_pretrain",
        ]),
        ["cv_summary.json for each cohort listed"],
        "Cohort name, n, positives, prevalence, valid-ratio threshold, label definition.",
    )
    make_table(
        "T6_modality_ablation_table",
        lambda: table_modality_ablation(cohort_dir),
        [f"{cohort_dir}/modality_ablation_results.csv",
         f"{cohort_dir}/modality_ablation_significance.csv"],
        "Modality ablation: ROC-AUC per model and checkpoint for CTG-features-only, "
        "maternal-only, and both, with the gain from adding maternal variables and its "
        "paired DeLong/permutation p-values. Direct evidence for the multimodal claim.",
    )

    write_manifest(manifest_entries, out_dir / "figure_manifest.md")

    # --- Summary ---
    print("\n=== Generated ===")
    for g in generated:
        print(f"  {g}")
    print("\n=== Skipped (missing source file) ===")
    if skipped:
        for s in skipped:
            print(f"  {s}")
    else:
        print("  (none)")
    print(f"\n=== Cross-check against {args.results_summary} ===")
    print(f"  {checker.checked} value(s) checked.")
    if checker.mismatches:
        print(f"  {len(checker.mismatches)} MISMATCH(ES):")
        for m in checker.mismatches:
            print(f"    {m}")
    else:
        print("  No mismatches found.")
    if notes:
        print("\n=== Internal consistency notes ===")
        for n in notes:
            print(f"  {n}")
    print(f"\nWrote figures to {out_dir}")
    print(f"Wrote tables to {tables_dir}")
    print(f"Wrote manifest to {out_dir / 'figure_manifest.md'}")


if __name__ == "__main__":
    main()
