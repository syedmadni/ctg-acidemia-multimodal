from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from src.models.ctg_cnn import CTGOnlyModel
from src.models.fusion_model import FusionModel, GatedFusionModel, MaternalOnlyModel
from src.training.evaluate import evaluate_predictions
from src.training.train_deep import (
    AnyTimeCTGDataset,
    CTGMaternalDataset,
    FocalLossWithLogits,
    fit_preprocessors,
    predict_model,
    pretrain_ctg_encoder,
    train_one_model,
    transform_arrays,
)
from src.utils.seed import set_global_seed


def label_short(label_col: str) -> str:
    return label_col.rsplit("_", maxsplit=1)[-1]


def ratio_short(valid_ratio: float) -> str:
    return f"vr{int(round(valid_ratio * 100)):03d}"


def make_suffix(label_col: str, valid_ratio: float) -> str:
    return f"_{label_short(label_col)}_{ratio_short(valid_ratio)}"


def load_variant(processed_dir: Path, suffix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    X_ctg = np.load(processed_dir / f"X_ctg{suffix}.npy")
    X_maternal = np.load(processed_dir / f"X_maternal{suffix}.npy")
    y = np.load(processed_dir / f"y{suffix}.npy").astype(int)
    record_ids = pd.read_csv(processed_dir / f"record_ids{suffix}.csv")
    return X_ctg, X_maternal, y, record_ids


def ctg_summary_features(X_ctg: np.ndarray) -> np.ndarray:
    stats = [
        np.nanmean(X_ctg, axis=1),
        np.nanstd(X_ctg, axis=1),
        np.nanmin(X_ctg, axis=1),
        np.nanmax(X_ctg, axis=1),
        np.nanmedian(X_ctg, axis=1),
    ]
    return np.concatenate(stats, axis=1).astype(np.float32)


def make_tabular_features(
    X_ctg: np.ndarray,
    X_maternal: np.ndarray,
    include_clinical: bool = False,
    target_sample_rate_hz: int = 1,
) -> np.ndarray:
    """Tabular feature block for the classical models. Default is the original
    10 CTG summary stats + 8 maternal features (unchanged, so existing cohorts
    reproduce exactly). When include_clinical=True, the Phase-4 FIGO/NICE-style
    clinical CTG features are APPENDED after the maternal block (append-only, so
    prior feature indices never shift)."""
    blocks = [ctg_summary_features(X_ctg), X_maternal]
    if include_clinical:
        from src.data.clinical_ctg_features import clinical_ctg_features
        blocks.append(clinical_ctg_features(X_ctg, target_sample_rate_hz))
    return np.concatenate(blocks, axis=1)


def make_sklearn_model(name: str, seed: int) -> Pipeline:
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
    if name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])
    raise ValueError(f"Unknown sklearn model: {name}")


def choose_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def safe_ranking_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def threshold_grid(y_proba: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(y_proba, dtype=float))
    return np.unique(np.concatenate([values, np.linspace(0.0, 1.0, 101)]))


def select_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    selected: Dict[str, float] = {}
    rows = []
    for threshold in threshold_grid(y_proba):
        metrics = evaluate_predictions(y_true, y_proba, threshold=float(threshold))
        rows.append({
            "threshold": float(threshold),
            "f1": metrics["f1"],
            "youden": metrics["recall_sensitivity"] + metrics["specificity"] - 1.0,
        })

    scores = pd.DataFrame(rows)
    selected["f1"] = float(scores.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]["threshold"])
    selected["youden"] = float(scores.sort_values(["youden", "threshold"], ascending=[False, False]).iloc[0]["threshold"])

    selected["fpr10"] = threshold_for_target_fpr(y_true, y_proba, target_fpr=0.10)
    selected["fpr20"] = threshold_for_target_fpr(y_true, y_proba, target_fpr=0.20)
    return selected


def threshold_for_target_fpr(y_true: np.ndarray, y_proba: np.ndarray, target_fpr: float) -> float:
    best_threshold = 1.0
    best_recall = -1.0
    best_fpr = -1.0
    for threshold in threshold_grid(y_proba):
        metrics = evaluate_predictions(y_true, y_proba, threshold=float(threshold))
        fp = metrics["fp"]
        tn = metrics["tn"]
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        recall = metrics["recall_sensitivity"]
        if fpr <= target_fpr and (recall > best_recall or (recall == best_recall and fpr > best_fpr)):
            best_threshold = float(threshold)
            best_recall = float(recall)
            best_fpr = float(fpr)
    return best_threshold


def calibrated_test_metrics(
    y_val: np.ndarray,
    val_proba: np.ndarray,
    y_test: np.ndarray,
    test_proba: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, threshold in select_thresholds(y_val, val_proba).items():
        metrics = evaluate_predictions(y_test, test_proba, threshold=threshold)
        for metric_name, value in metrics.items():
            out[f"cal_{name}_{metric_name}"] = value
    return out


def run_sklearn_fold(
    model_name: str,
    X_tabular: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    threshold: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    fit_idx, calib_idx = train_test_split(
        train_idx,
        test_size=0.20,
        random_state=seed,
        stratify=y[train_idx],
    )
    model = make_sklearn_model(model_name, seed)
    model.fit(X_tabular[fit_idx], y[fit_idx])
    val_proba = model.predict_proba(X_tabular[calib_idx])[:, 1]
    test_proba = model.predict_proba(X_tabular[test_idx])[:, 1]
    metrics = evaluate_predictions(y[test_idx], test_proba, threshold=threshold)
    metrics.update(safe_ranking_metrics(y[test_idx], test_proba))
    metrics.update(calibrated_test_metrics(y[calib_idx], val_proba, y[test_idx], test_proba))
    return metrics, test_proba


def deep_model_for_name(
    model_name: str,
    X_ctg: np.ndarray,
    X_maternal: np.ndarray,
    encoder_kwargs: Dict[str, object] | None = None,
) -> nn.Module:
    """encoder_kwargs (e.g. {d_model, transformer_layers, n_heads, dropout}) are
    forwarded to CTGEncoder for the Phase-3 capacity sweep. Absent/None -> the
    original CTGEncoder defaults, so every existing cohort reproduces exactly.
    maternal_only has no CTG encoder, so it ignores these."""
    encoder_kwargs = dict(encoder_kwargs or {})
    if model_name == "ctg_only":
        return CTGOnlyModel(input_channels=X_ctg.shape[-1], **encoder_kwargs)
    if model_name == "maternal_only":
        return MaternalOnlyModel(num_features=X_maternal.shape[1])
    if model_name == "fusion":
        return FusionModel(num_maternal_features=X_maternal.shape[1], input_channels=X_ctg.shape[-1], **encoder_kwargs)
    if model_name == "gated_fusion":
        return GatedFusionModel(num_maternal_features=X_maternal.shape[1], input_channels=X_ctg.shape[-1], **encoder_kwargs)
    raise ValueError(f"Unknown deep model: {model_name}")


def run_deep_fold(
    model_name: str,
    X_ctg: np.ndarray,
    X_maternal: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Dict,
    seed: int,
    threshold: float,
    epochs: int,
    out_dir: Path,
    fold_name: str,
) -> Tuple[Dict[str, float], np.ndarray]:
    inner_train_idx, val_idx = train_test_split(
        train_idx,
        test_size=float(cfg["training"]["val_size"]),
        random_state=seed,
        stratify=y[train_idx],
    )

    _, _, imputer, scaler, ctg_mean, ctg_std = fit_preprocessors(X_ctg[inner_train_idx], X_maternal[inner_train_idx])
    X_ctg_norm, X_mat_norm = transform_arrays(X_ctg, X_maternal, imputer, scaler, ctg_mean, ctg_std)

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(
        CTGMaternalDataset(X_ctg_norm[inner_train_idx], X_mat_norm[inner_train_idx], y[inner_train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        CTGMaternalDataset(X_ctg_norm[val_idx], X_mat_norm[val_idx], y[val_idx]),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        CTGMaternalDataset(X_ctg_norm[test_idx], X_mat_norm[test_idx], y[test_idx]),
        batch_size=batch_size,
        shuffle=False,
    )

    pos = int(y[inner_train_idx].sum())
    neg = int((y[inner_train_idx] == 0).sum())
    pos_weight = float(neg / max(pos, 1))
    device = choose_device(str(cfg["training"]["device"]))
    model_dir = out_dir / "models" / model_name / fold_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model = deep_model_for_name(model_name, X_ctg, X_maternal)
    trained = train_one_model(
        name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        pos_weight=pos_weight,
        epochs=epochs,
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        patience=int(cfg["training"]["early_stopping_patience"]),
        models_dir=model_dir,
    )
    val_proba = predict_model(model_name, trained, val_loader, device)
    test_proba = predict_model(model_name, trained, test_loader, device)
    metrics = evaluate_predictions(y[test_idx], test_proba, threshold=threshold)
    metrics.update(safe_ranking_metrics(y[test_idx], test_proba))
    metrics.update(calibrated_test_metrics(y[val_idx], val_proba, y[test_idx], test_proba))
    return metrics, test_proba


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in fold_metrics.columns
        if c not in {"label", "valid_ratio", "suffix", "model", "repeat", "fold", "train_size", "test_size", "train_positives", "test_positives"}
    ]
    rows = []
    for (label, valid_ratio, suffix, model), group in fold_metrics.groupby(["label", "valid_ratio", "suffix", "model"]):
        row = {
            "label": label,
            "valid_ratio": valid_ratio,
            "suffix": suffix,
            "model": model,
            "folds": int(len(group)),
            "mean_test_positives": float(group["test_positives"].mean()),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_label(row: pd.Series) -> str:
    model = str(row["model"]).replace("_", " ")
    label = str(row["label"]).replace("y_acidemia_", "pH<7.")
    valid_ratio = int(round(float(row["valid_ratio"]) * 100))
    return f"{model}\n{label}, VR {valid_ratio}%"


def save_bar_plot(
    summary: pd.DataFrame,
    out_path: Path,
    value_col: str,
    err_col: str | None, 
    title: str,
    ylabel: str,
) -> None:
    plot_df = summary.copy()
    plot_df = plot_df.sort_values(value_col, ascending=False).reset_index(drop=True)
    labels = [plot_label(row) for _, row in plot_df.iterrows()]
    values = plot_df[value_col].astype(float).to_numpy()
    errors = plot_df[err_col].astype(float).to_numpy() if err_col and err_col in plot_df.columns else None

    width = max(9.0, 0.65 * len(plot_df))
    plt.figure(figsize=(width, 5.5))
    plt.bar(np.arange(len(plot_df)), values, yerr=errors, capsize=4, color="#4C78A8")
    plt.xticks(np.arange(len(plot_df)), labels, rotation=45, ha="right")
    plt.ylim(0.0, min(1.05, max(1.0, float(np.nanmax(values + (errors if errors is not None else 0))) + 0.08)))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close() 


def save_grouped_metric_plot(summary: pd.DataFrame, out_path: Path, metric_cols: List[str], title: str) -> None:
    plot_df = summary.copy().reset_index(drop=True)
    labels = [plot_label(row) for _, row in plot_df.iterrows()]
    x = np.arange(len(plot_df))
    width = 0.8 / max(len(metric_cols), 1)

    plt.figure(figsize=(max(10.0, 0.75 * len(plot_df)), 5.8))
    for i, col in enumerate(metric_cols):
        offset = (i - (len(metric_cols) - 1) / 2) * width
        plt.bar(x + offset, plot_df[col].astype(float), width=width, label=col.replace("_mean", "").replace("cal_fpr20_", "").replace("_", " "))
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_auc_comparison_plot(summary: pd.DataFrame, out_path: Path) -> None:
    required = {"roc_auc_mean", "pr_auc_mean"}
    if not required.issubset(summary.columns):
        return

    plot_df = summary.copy()
    plot_df = plot_df.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    labels = [plot_label(row) for _, row in plot_df.iterrows()]
    x = np.arange(len(plot_df))
    width = 0.38

    plt.figure(figsize=(max(10.0, 0.75 * len(plot_df)), 5.8))
    plt.bar(x - width / 2, plot_df["roc_auc_mean"].astype(float), width=width, label="ROC AUC", color="#4C78A8")
    plt.bar(x + width / 2, plot_df["pr_auc_mean"].astype(float), width=width, label="PR AUC", color="#F58518")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean CV Score")
    plt.title("Cross-Validated ROC AUC and PR AUC")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def curve_label(group_key: Tuple[str, float, str]) -> str:
    label, valid_ratio, model = group_key
    label_text = str(label).replace("y_acidemia_", "pH<7.")
    valid_ratio_text = f"VR {int(round(float(valid_ratio) * 100))}%"
    model_text = str(model).replace("_", " ")
    return f"{model_text} - {label_text}, {valid_ratio_text}"


def save_pooled_roc_curves(predictions: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7.0, 6.2))
    for key, group in predictions.groupby(["label", "valid_ratio", "model"]):
        y_true = group["y_true"].astype(int).to_numpy()
        y_proba = group["y_proba"].astype(float).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_value = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, linewidth=1.6, label=f"{curve_label(key)} (area = {auc_value:.2f})")
    plt.plot([0, 1], [0, 1], color="0.35", linewidth=1.0)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_pooled_pr_curves(predictions: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7.0, 6.2))
    for key, group in predictions.groupby(["label", "valid_ratio", "model"]):
        y_true = group["y_true"].astype(int).to_numpy()
        y_proba = group["y_proba"].astype(float).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap_value = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, linewidth=1.6, label=f"{curve_label(key)} (area = {ap_value:.2f})")
    prevalence = float(predictions["y_true"].mean()) if len(predictions) else 0.0
    plt.axhline(prevalence, color="0.35", linewidth=1.0)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_prediction_curve_plots(predictions: pd.DataFrame, out_dir: Path) -> None:
    if predictions.empty:
        return
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_pooled_roc_curves(
        predictions,
        figures_dir / "pooled_cv_roc_curves.png",
        "Pooled Cross-Validated ROC Curves",
    )
    save_pooled_pr_curves(
        predictions,
        figures_dir / "pooled_cv_pr_curves.png",
        "Pooled Cross-Validated Precision-Recall Curves",
    )

    focused = predictions[
        (predictions["label"] == "y_acidemia_710")
        & np.isclose(predictions["valid_ratio"].astype(float), 0.70)
    ].copy()
    if not focused.empty:
        save_pooled_roc_curves(
            focused,
            figures_dir / "focused_710_vr070_roc_curves.png",
            "ROC Curves: pH < 7.10, Valid Ratio 0.70",
        )
        save_pooled_pr_curves(
            focused,
            figures_dir / "focused_710_vr070_pr_curves.png",
            "Precision-Recall Curves: pH < 7.10, Valid Ratio 0.70",
        )


def save_cv_summary_plots(summary: pd.DataFrame, out_dir: Path) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_auc_comparison_plot(summary, figures_dir / "roc_pr_auc_comparison.png")

    if "roc_auc_mean" in summary.columns:
        save_bar_plot(
            summary,
            figures_dir / "roc_auc_mean.png",
            "roc_auc_mean",
            "roc_auc_std",
            "Cross-Validated ROC AUC",
            "Mean ROC AUC",
        )
    if "pr_auc_mean" in summary.columns:
        save_bar_plot(
            summary,
            figures_dir / "pr_auc_mean.png",
            "pr_auc_mean",
            "pr_auc_std",
            "Cross-Validated PR AUC",
            "Mean PR AUC",
        )

    default_cols = [c for c in ["recall_sensitivity_mean", "specificity_mean", "precision_mean", "f1_mean"] if c in summary.columns]
    if default_cols:
        save_grouped_metric_plot(
            summary,
            figures_dir / "default_threshold_metrics.png",
            default_cols,
            "Default Threshold Metrics",
        )

    calibrated_cols = [
        c for c in [
            "cal_fpr20_recall_sensitivity_mean",
            "cal_fpr20_specificity_mean",
            "cal_fpr20_precision_mean",
            "cal_fpr20_f1_mean",
        ]
        if c in summary.columns
    ]
    if calibrated_cols:
        save_grouped_metric_plot(
            summary,
            figures_dir / "calibrated_fpr20_metrics.png",
            calibrated_cols,
            "Validation-Calibrated Metrics at Target 20% FPR",
        )

    focused = summary[
        (summary["label"] == "y_acidemia_710")
        & np.isclose(summary["valid_ratio"].astype(float), 0.70)
    ].copy()
    if len(focused) > 0:
        save_grouped_metric_plot(
            focused,
            figures_dir / "focused_710_vr070_ranking.png",
            ["roc_auc_mean", "pr_auc_mean"],
            "Focused Comparison: pH < 7.10, Valid Ratio 0.70",
        )


def requested_models(default_models: Iterable[str], cli_models: str | None) -> List[str]:
    if not cli_models:
        return list(default_models)
    return [m.strip() for m in cli_models.split(",") if m.strip()]


def requested_labels(default_labels: Iterable[str], cli_labels: str | None) -> List[str]:
    if not cli_labels:
        return list(default_labels)
    return [label.strip() for label in cli_labels.split(",") if label.strip()]


def requested_valid_ratios(default_ratios: Iterable[float], cli_ratios: str | None) -> List[float]:
    if not cli_ratios:
        return [float(ratio) for ratio in default_ratios]
    return [float(ratio.strip()) for ratio in cli_ratios.split(",") if ratio.strip()]


def default_run_name(models: List[str], epochs: int, max_folds: int | None) -> str:
    kind = "deep" if any(m in {"maternal_only", "ctg_only", "fusion"} for m in models) else "tabular"
    suffix = f"epochs{epochs}"
    if max_folds is not None:
        suffix = f"{suffix}_maxfolds{max_folds}"
    return f"{kind}_{'-'.join(models)}_{suffix}"


def plot_existing_run(run_dir: Path) -> None:
    summary_path = run_dir / "summary_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    summary = pd.read_csv(summary_path)
    save_cv_summary_plots(summary, run_dir)
    predictions_path = run_dir / "fold_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        save_prediction_curve_plots(predictions, run_dir)
    print(f"Saved figures under: {run_dir / 'figures'}")


def ensure_absent_or_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite only if intentional.")
        shutil.rmtree(path)


def require_cohort_suffix(suffix: str) -> None:
    if not suffix:
        raise ValueError("Quality-cohort runs require non-empty suffixed processed arrays.")


def load_cohort_arrays(processed_dir: Path, suffix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    require_cohort_suffix(suffix)
    paths = {
        "X_ctg": processed_dir / f"X_ctg{suffix}.npy",
        "X_maternal": processed_dir / f"X_maternal{suffix}.npy",
        "y": processed_dir / f"y{suffix}.npy",
        "record_ids": processed_dir / f"record_ids{suffix}.csv",
        "cohort_manifest": processed_dir / f"cohort_manifest{suffix}.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing cohort input files: " + "; ".join(missing))
    X_ctg = np.load(paths["X_ctg"])
    X_maternal = np.load(paths["X_maternal"])
    y = np.load(paths["y"]).astype(int)
    record_ids = pd.read_csv(paths["record_ids"])
    manifest = pd.read_csv(paths["cohort_manifest"])
    if not (len(y) == X_ctg.shape[0] == X_maternal.shape[0] == len(record_ids)):
        raise ValueError(f"Cohort shape mismatch for suffix {suffix}.")
    included_manifest = manifest[manifest["inclusion_status"] == "included"].copy()
    if len(included_manifest) != len(y):
        raise ValueError(f"Included manifest row count does not match arrays for suffix {suffix}.")
    return X_ctg, X_maternal, y, record_ids, manifest


def precompute_quality_splits(
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
    val_size: float,
) -> List[Dict[str, np.ndarray | int | str]]:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits: List[Dict[str, np.ndarray | int | str]] = []
    idx = np.arange(len(y))
    for split_number, (outer_train_idx, outer_test_idx) in enumerate(cv.split(idx, y), start=1):
        repeat = ((split_number - 1) // n_splits) + 1
        fold = ((split_number - 1) % n_splits) + 1
        inner_train_idx, val_idx = train_test_split(
            outer_train_idx,
            test_size=val_size,
            random_state=seed + split_number,
            stratify=y[outer_train_idx],
        )
        splits.append({
            "repeat": repeat,
            "fold": fold,
            "fold_name": f"repeat{repeat:02d}_fold{fold:02d}",
            "outer_train_idx": outer_train_idx,
            "outer_test_idx": outer_test_idx,
            "inner_train_idx": inner_train_idx,
            "val_idx": val_idx,
        })
    return splits


def make_split_manifest(
    splits: List[Dict[str, np.ndarray | int | str]],
    record_ids: pd.DataFrame,
    y: np.ndarray,
    cohort_name: str,
    valid_ratio_threshold: float,
) -> pd.DataFrame:
    rows = []
    for split in splits:
        repeat = int(split["repeat"])
        fold = int(split["fold"])
        outer_test = set(np.asarray(split["outer_test_idx"], dtype=int).tolist())
        inner_train = set(np.asarray(split["inner_train_idx"], dtype=int).tolist())
        val = set(np.asarray(split["val_idx"], dtype=int).tolist())
        for row_index in range(len(y)):
            if row_index in outer_test:
                outer_split = "test"
                inner_split = "outer_test"
            elif row_index in val:
                outer_split = "train"
                inner_split = "validation"
            elif row_index in inner_train:
                outer_split = "train"
                inner_split = "inner_train"
            else:
                raise RuntimeError(f"Row {row_index} missing from split repeat={repeat} fold={fold}")
            rows.append({
                "cohort_name": cohort_name,
                "valid_ratio_threshold": valid_ratio_threshold,
                "repeat": repeat,
                "fold": fold,
                "row_index": row_index,
                "record_id": record_ids.iloc[row_index]["record_id"],
                "y": int(y[row_index]),
                "outer_split": outer_split,
                "inner_split": inner_split,
            })
    return pd.DataFrame(rows)


def model_logits(model_name: str, model: nn.Module, x_ctg: torch.Tensor, x_mat: torch.Tensor) -> torch.Tensor:
    if model_name == "ctg_only":
        return model(x_ctg)
    if model_name == "maternal_only":
        return model(x_mat)
    return model(x_ctg, x_mat)


def train_deep_fold_by_val_pr_auc(
    model_name: str,
    X_ctg: np.ndarray,
    X_maternal: np.ndarray,
    y: np.ndarray,
    split: Dict[str, np.ndarray | int | str],
    cfg: Dict,
    seed: int,
    fold_artifact_dir: Path,
) -> Tuple[nn.Module, Dict[str, object], np.ndarray, np.ndarray]:
    set_global_seed(seed + int(split["repeat"]) * 100 + int(split["fold"]))
    inner_train_idx = np.asarray(split["inner_train_idx"], dtype=int)
    val_idx = np.asarray(split["val_idx"], dtype=int)
    outer_test_idx = np.asarray(split["outer_test_idx"], dtype=int)

    _, _, imputer, scaler, ctg_mean, ctg_std = fit_preprocessors(X_ctg[inner_train_idx], X_maternal[inner_train_idx])
    X_ctg_norm, X_mat_norm = transform_arrays(X_ctg, X_maternal, imputer, scaler, ctg_mean, ctg_std)

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(
        CTGMaternalDataset(X_ctg_norm[inner_train_idx], X_mat_norm[inner_train_idx], y[inner_train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        CTGMaternalDataset(X_ctg_norm[val_idx], X_mat_norm[val_idx], y[val_idx]),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        CTGMaternalDataset(X_ctg_norm[outer_test_idx], X_mat_norm[outer_test_idx], y[outer_test_idx]),
        batch_size=batch_size,
        shuffle=False,
    )

    pos = int(y[inner_train_idx].sum())
    neg = int((y[inner_train_idx] == 0).sum())
    pos_weight = float(neg / max(pos, 1))
    device = choose_device(str(cfg["training"]["device"]))
    model = deep_model_for_name(model_name, X_ctg, X_maternal).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    fold_artifact_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_artifact_dir / f"{model_name}.pt"
    preprocessor_path = fold_artifact_dir / "preprocessors.joblib"
    best_val_pr_auc = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    patience = int(cfg["training"]["early_stopping_patience"])

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        for x_ctg, x_mat, batch_y in train_loader:
            x_ctg, x_mat, batch_y = x_ctg.to(device), x_mat.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model_logits(model_name, model, x_ctg, x_mat), batch_y)
            loss.backward()
            optimizer.step()

        val_proba = predict_model(model_name, model, val_loader, device)
        val_pr_auc = (
            float(average_precision_score(y[val_idx], val_proba))
            if len(np.unique(y[val_idx])) == 2
            else float("nan")
        )
        score = val_pr_auc if np.isfinite(val_pr_auc) else -float("inf")
        if best_epoch == 0 or score > best_val_pr_auc:
            best_val_pr_auc = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            joblib.dump(
                {"maternal_imputer": imputer, "maternal_scaler": scaler, "ctg_mean": ctg_mean, "ctg_std": ctg_std},
                preprocessor_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_proba = predict_model(model_name, model, test_loader, device)
    val_proba = predict_model(model_name, model, val_loader, device)
    info = {
        "best_val_pr_auc": float(best_val_pr_auc),
        "best_epoch": int(best_epoch),
        "pos_weight": pos_weight,
        "artifact_dir": str(fold_artifact_dir),
        "checkpoint_path": str(best_path),
        "preprocessor_path": str(preprocessor_path),
    }
    return model, info, test_proba, val_proba


def train_sklearn_quality_fold(
    model_name: str,
    X_tabular: np.ndarray,
    y: np.ndarray,
    split: Dict[str, np.ndarray | int | str],
    seed: int,
    fold_artifact_dir: Path,
) -> Tuple[Pipeline, Dict[str, object], np.ndarray, np.ndarray]:
    """Classical (LR/RF) counterpart to train_deep_fold_by_val_pr_auc, fit on
    the identical inner_train/val/outer_test row indices so folds are
    comparable to the deep models one-for-one."""
    inner_train_idx = np.asarray(split["inner_train_idx"], dtype=int)
    val_idx = np.asarray(split["val_idx"], dtype=int)
    outer_test_idx = np.asarray(split["outer_test_idx"], dtype=int)

    model = make_sklearn_model(model_name, seed)
    model.fit(X_tabular[inner_train_idx], y[inner_train_idx])

    val_proba = model.predict_proba(X_tabular[val_idx])[:, 1]
    test_proba = model.predict_proba(X_tabular[outer_test_idx])[:, 1]
    val_pr_auc = (
        float(average_precision_score(y[val_idx], val_proba))
        if len(np.unique(y[val_idx])) == 2
        else float("nan")
    )

    fold_artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_artifact_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    info = {
        "best_val_pr_auc": val_pr_auc,
        "best_epoch": 0,
        "pos_weight": None,
        "artifact_dir": str(fold_artifact_dir),
        "checkpoint_path": str(model_path),
        "preprocessor_path": None,
    }
    return model, info, test_proba, val_proba


def crop_ctg_array(X_ctg_norm: np.ndarray, checkpoint_minutes: float, target_sample_rate_hz: int) -> np.ndarray:
    """Zero out everything past checkpoint_minutes of elapsed time on an
    already-normalized [N, T, C] array -- the fixed-checkpoint counterpart to
    AnyTimeCTGDataset's random crop, used for outer-test evaluation so
    reported numbers come from deliberately chosen, reproducible checkpoints
    rather than a random draw."""
    crop_samples = int(round(checkpoint_minutes * 60 * target_sample_rate_hz))
    cropped = X_ctg_norm.copy()
    if crop_samples < cropped.shape[1]:
        cropped[:, crop_samples:, :] = 0.0
    return cropped


def train_deep_fold_anytime(
    model_name: str,
    X_ctg: np.ndarray,
    X_maternal: np.ndarray,
    y: np.ndarray,
    split: Dict[str, np.ndarray | int | str],
    cfg: Dict,
    seed: int,
    fold_artifact_dir: Path,
    target_sample_rate_hz: int,
    min_crop_minutes: float,
    max_crop_minutes: float,
    deep_overrides: Dict[str, object] | None = None,
    duration_minutes: np.ndarray | None = None,
) -> Tuple[nn.Module, Dict[str, object], SimpleImputer, StandardScaler, np.ndarray, np.ndarray]:
    """Anytime counterpart to train_deep_fold_by_val_pr_auc: trains ONE model
    per (model_name, fold) using randomly-cropped elapsed-time lengths for
    both train and validation data every epoch (so checkpoint selection by
    val PR-AUC isn't biased toward one fixed length) -- instead of one model
    per fixed window. Returns the fitted preprocessors too, so the caller can
    transform+crop the outer-test arrays at each reporting checkpoint using
    this exact fold's fitted preprocessing.

    deep_overrides (all optional; absent -> original behavior, bit-identical):
      - "encoder": kwargs forwarded to CTGEncoder (Phase 3 capacity sweep).
      - "loss": {"type": "bce"|"focal", "focal_alpha", "focal_gamma"} (Phase 6b).
      - "augmentation": {"noise_std", "scale_range"} for the TRAIN loader only
        (Phase 6a)."""
    deep_overrides = dict(deep_overrides or {})
    encoder_kwargs = dict(deep_overrides.get("encoder", {}) or {})
    loss_cfg = dict(deep_overrides.get("loss", {}) or {})
    augment = deep_overrides.get("augmentation", None)

    fold_seed = seed + int(split["repeat"]) * 100 + int(split["fold"])
    set_global_seed(fold_seed)
    inner_train_idx = np.asarray(split["inner_train_idx"], dtype=int)
    val_idx = np.asarray(split["val_idx"], dtype=int)

    _, _, imputer, scaler, ctg_mean, ctg_std = fit_preprocessors(X_ctg[inner_train_idx], X_maternal[inner_train_idx])
    X_ctg_norm, X_mat_norm = transform_arrays(X_ctg, X_maternal, imputer, scaler, ctg_mean, ctg_std)

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(
        AnyTimeCTGDataset(
            X_ctg_norm[inner_train_idx], X_mat_norm[inner_train_idx], y[inner_train_idx],
            target_sample_rate_hz, min_crop_minutes, max_crop_minutes, seed=fold_seed,
            augment=augment,  # train loader only
            duration_minutes=(None if duration_minutes is None
                              else np.asarray(duration_minutes)[inner_train_idx]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        AnyTimeCTGDataset(
            X_ctg_norm[val_idx], X_mat_norm[val_idx], y[val_idx],
            target_sample_rate_hz, min_crop_minutes, max_crop_minutes, seed=fold_seed + 1,
            augment=None,  # never augment validation
            duration_minutes=(None if duration_minutes is None
                              else np.asarray(duration_minutes)[val_idx]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    pos = int(y[inner_train_idx].sum())
    neg = int((y[inner_train_idx] == 0).sum())
    pos_weight = float(neg / max(pos, 1))
    device = choose_device(str(cfg["training"]["device"]))
    model = deep_model_for_name(model_name, X_ctg, X_maternal, encoder_kwargs).to(device)

    # Phase 6d: optional auxiliary pretraining of the CTG encoder on clinical
    # CTG features before the acidemia fine-tune. Skipped for maternal_only
    # (no CTG encoder). Targets come from the raw signal, standardized on
    # inner-train rows only.
    pretrain_cfg = deep_overrides.get("pretrain", None)
    pretrain_info: Dict[str, float] = {}
    if pretrain_cfg and model_name != "maternal_only":
        from src.data.clinical_ctg_features import clinical_ctg_features

        encoder_attr = "encoder" if model_name == "ctg_only" else "ctg_encoder"
        encoder = getattr(model, encoder_attr)
        targets_all = clinical_ctg_features(X_ctg, target_sample_rate_hz)
        t_mean = targets_all[inner_train_idx].mean(axis=0)
        t_std = targets_all[inner_train_idx].std(axis=0)
        t_std = np.where(t_std < 1e-6, 1.0, t_std)
        targets_norm = ((targets_all - t_mean) / t_std).astype(np.float32)
        pretrain_info = pretrain_ctg_encoder(
            encoder,
            X_ctg_norm[inner_train_idx],
            targets_norm[inner_train_idx],
            device=device,
            epochs=int(pretrain_cfg.get("epochs", 20)),
            learning_rate=float(pretrain_cfg.get("learning_rate", 1e-3)),
            batch_size=batch_size,
            seed=fold_seed,
        )

    if str(loss_cfg.get("type", "bce")).lower() == "focal":
        criterion: nn.Module = FocalLossWithLogits(
            alpha=float(loss_cfg.get("focal_alpha", 0.25)),
            gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    fold_artifact_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_artifact_dir / f"{model_name}.pt"
    preprocessor_path = fold_artifact_dir / "preprocessors.joblib"
    best_val_pr_auc = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    patience = int(cfg["training"]["early_stopping_patience"])

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        for x_ctg, x_mat, batch_y in train_loader:
            x_ctg, x_mat, batch_y = x_ctg.to(device), x_mat.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model_logits(model_name, model, x_ctg, x_mat), batch_y)
            loss.backward()
            optimizer.step()

        val_proba = predict_model(model_name, model, val_loader, device)
        val_pr_auc = (
            float(average_precision_score(y[val_idx], val_proba))
            if len(np.unique(y[val_idx])) == 2
            else float("nan")
        )
        score = val_pr_auc if np.isfinite(val_pr_auc) else -float("inf")
        if best_epoch == 0 or score > best_val_pr_auc:
            best_val_pr_auc = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            joblib.dump(
                {"maternal_imputer": imputer, "maternal_scaler": scaler, "ctg_mean": ctg_mean, "ctg_std": ctg_std},
                preprocessor_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    info = {
        "best_val_pr_auc": float(best_val_pr_auc),
        "best_epoch": int(best_epoch),
        "pos_weight": pos_weight,
        "artifact_dir": str(fold_artifact_dir),
        "checkpoint_path": str(best_path),
        "preprocessor_path": str(preprocessor_path),
    }
    info.update(pretrain_info)
    return model, info, imputer, scaler, ctg_mean, ctg_std


def aggregate_repeated_oof(oof: pd.DataFrame) -> pd.DataFrame:
    return (
        oof.groupby(["cohort_name", "valid_ratio_threshold", "model", "record_id"], as_index=False)
        .agg(y_true=("y_true", "first"), y_proba=("y_proba", "mean"))
    )


def bootstrap_metric_ci(
    averaged_oof: pd.DataFrame,
    threshold: float,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    metric_names = ["roc_auc", "pr_auc", "recall_sensitivity", "specificity", "f1"]
    for (cohort_name, valid_ratio_threshold, model), group in averaged_oof.groupby(["cohort_name", "valid_ratio_threshold", "model"]):
        y_true = group["y_true"].astype(int).to_numpy()
        y_proba = group["y_proba"].astype(float).to_numpy()
        point = evaluate_predictions(y_true, y_proba, threshold=threshold)
        samples = {name: [] for name in metric_names}
        for _ in range(iterations):
            sample_idx = rng.integers(0, len(group), size=len(group))
            y_s = y_true[sample_idx]
            p_s = y_proba[sample_idx]
            if len(np.unique(y_s)) < 2:
                continue
            metrics = evaluate_predictions(y_s, p_s, threshold=threshold)
            for name in metric_names:
                samples[name].append(metrics[name])
        for name in metric_names:
            values = np.asarray(samples[name], dtype=float)
            rows.append({
                "cohort_name": cohort_name,
                "valid_ratio_threshold": valid_ratio_threshold,
                "model": model,
                "metric": name,
                "point_estimate": float(point[name]),
                "ci_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                "bootstrap_iterations_requested": int(iterations),
                "bootstrap_iterations_used": int(len(values)),
            })
    return pd.DataFrame(rows)


def summarize_quality_oof(averaged_oof: pd.DataFrame, threshold: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for model, group in averaged_oof.groupby("model"):
        metrics = evaluate_predictions(
            group["y_true"].astype(int).to_numpy(),
            group["y_proba"].astype(float).to_numpy(),
            threshold=threshold,
        )
        out[str(model)] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in metrics.items()}
    return out


def aggregate_repeated_oof_anytime(oof: pd.DataFrame) -> pd.DataFrame:
    """Anytime counterpart to aggregate_repeated_oof: groups by
    checkpoint_minutes in addition to model/record_id, since one cohort now
    yields per-checkpoint OOF predictions from the same trained model rather
    than one set of OOF predictions per fixed-window cohort."""
    return (
        oof.groupby(["cohort_name", "checkpoint_minutes", "model", "record_id"], as_index=False)
        .agg(y_true=("y_true", "first"), y_proba=("y_proba", "mean"))
    )


def bootstrap_metric_ci_anytime(
    averaged_oof: pd.DataFrame,
    threshold: float,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    metric_names = ["roc_auc", "pr_auc", "recall_sensitivity", "specificity", "f1"]
    for (cohort_name, checkpoint_minutes, model), group in averaged_oof.groupby(
        ["cohort_name", "checkpoint_minutes", "model"]
    ):
        y_true = group["y_true"].astype(int).to_numpy()
        y_proba = group["y_proba"].astype(float).to_numpy()
        point = evaluate_predictions(y_true, y_proba, threshold=threshold)
        samples = {name: [] for name in metric_names}
        for _ in range(iterations):
            sample_idx = rng.integers(0, len(group), size=len(group))
            y_s = y_true[sample_idx]
            p_s = y_proba[sample_idx]
            if len(np.unique(y_s)) < 2:
                continue
            metrics = evaluate_predictions(y_s, p_s, threshold=threshold)
            for name in metric_names:
                samples[name].append(metrics[name])
        for name in metric_names:
            values = np.asarray(samples[name], dtype=float)
            rows.append({
                "cohort_name": cohort_name,
                "checkpoint_minutes": checkpoint_minutes,
                "model": model,
                "metric": name,
                "point_estimate": float(point[name]),
                "ci_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                "bootstrap_iterations_requested": int(iterations),
                "bootstrap_iterations_used": int(len(values)),
            })
    return pd.DataFrame(rows)


def summarize_oof_anytime(averaged_oof: pd.DataFrame, threshold: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for (checkpoint_minutes, model), group in averaged_oof.groupby(["checkpoint_minutes", "model"]):
        metrics = evaluate_predictions(
            group["y_true"].astype(int).to_numpy(),
            group["y_proba"].astype(float).to_numpy(),
            threshold=threshold,
        )
        out[f"{model}@{checkpoint_minutes}min"] = {
            k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in metrics.items()
        }
    return out


def run_quality_cohorts(args: argparse.Namespace) -> None:
    exp_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(exp_cfg["base_config"]).read_text(encoding="utf-8"))
    processed_dir = Path(base_cfg["paths"]["processed_dir"])
    output_base = Path(exp_cfg["output"]["base_dir"])
    models = list(exp_cfg["models"])
    known_models = {
        "ctg_only", "maternal_only", "fusion", "gated_fusion",
        "logistic_regression", "random_forest",
    }
    if not set(models).issubset(known_models):
        raise ValueError(
            f"--quality-cohorts model names must be a subset of {sorted(known_models)}."
        )

    cv_cfg = exp_cfg["cross_validation"]
    n_splits = int(cv_cfg["n_splits"])
    n_repeats = int(cv_cfg["n_repeats"])
    seed = int(cv_cfg["seed"])
    bootstrap_iterations = int(cv_cfg["bootstrap_iterations"])
    threshold = float(base_cfg["training"]["threshold"])
    val_size = float(base_cfg["training"]["val_size"])

    configured_cohorts = list(exp_cfg["cohorts"])
    if args.cohort:
        cohort_names = {str(cohort["name"]) for cohort in configured_cohorts}
        if args.cohort not in cohort_names:
            raise ValueError(f"Unknown cohort '{args.cohort}'. Available cohorts: {sorted(cohort_names)}")
        configured_cohorts = [cohort for cohort in configured_cohorts if str(cohort["name"]) == args.cohort]

    for cohort in configured_cohorts:
        cohort_name = str(cohort["name"])
        suffix = str(cohort["suffix"])
        valid_ratio_threshold = float(cohort["min_valid_ratio"])
        require_cohort_suffix(suffix)
        cohort_dir = output_base / cohort_name
        ensure_absent_or_overwrite(cohort_dir, args.overwrite)
        cohort_dir.mkdir(parents=True, exist_ok=False)

        X_ctg, X_maternal, y, record_ids, manifest = load_cohort_arrays(processed_dir, suffix)
        X_tabular = make_tabular_features(X_ctg, X_maternal)
        manifest.to_csv(cohort_dir / "cohort_manifest.csv", index=False)
        snapshot = {
            "base_config": base_cfg,
            "quality_cohort_config": exp_cfg,
            "cohort": cohort,
            "quality_filtering_label_dependent": bool(exp_cfg["quality_filtering"]["label_dependent"]),
            "audit_source": exp_cfg["quality_filtering"]["audit_source"],
            "interpretation_note": exp_cfg["quality_filtering"]["interpretation_note"],
        }
        (cohort_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

        splits = precompute_quality_splits(y, n_splits, n_repeats, seed, val_size)
        split_manifest = make_split_manifest(splits, record_ids, y, cohort_name, valid_ratio_threshold)
        split_manifest.to_csv(cohort_dir / "split_manifest.csv", index=False)

        fold_metric_rows = []
        oof_rows = []
        checkpoint_candidates: Dict[str, List[Dict[str, object]]] = {model: [] for model in models}
        fold_root = cohort_dir / ("models" if args.save_all_fold_models else "_fold_checkpoints")

        for split in splits:
            repeat = int(split["repeat"])
            fold = int(split["fold"])
            fold_name = str(split["fold_name"])
            outer_train_idx = np.asarray(split["outer_train_idx"], dtype=int)
            outer_test_idx = np.asarray(split["outer_test_idx"], dtype=int)
            inner_train_idx = np.asarray(split["inner_train_idx"], dtype=int)
            val_idx = np.asarray(split["val_idx"], dtype=int)
            for model_name in models:
                print(f"{cohort_name} | {fold_name} | {model_name}")
                artifact_dir = fold_root / model_name / fold_name
                if model_name in {"logistic_regression", "random_forest"}:
                    _, train_info, test_proba, _ = train_sklearn_quality_fold(
                        model_name,
                        X_tabular,
                        y,
                        split,
                        seed,
                        artifact_dir,
                    )
                else:
                    _, train_info, test_proba, _ = train_deep_fold_by_val_pr_auc(
                        model_name,
                        X_ctg,
                        X_maternal,
                        y,
                        split,
                        base_cfg,
                        seed,
                        artifact_dir,
                    )
                metrics = evaluate_predictions(y[outer_test_idx], test_proba, threshold=threshold)
                metrics.update(safe_ranking_metrics(y[outer_test_idx], test_proba))
                row = {
                    "cohort_name": cohort_name,
                    "valid_ratio_threshold": valid_ratio_threshold,
                    "label": exp_cfg["label"]["name"],
                    "suffix": suffix,
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "train_size": int(len(outer_train_idx)),
                    "inner_train_size": int(len(inner_train_idx)),
                    "val_size": int(len(val_idx)),
                    "test_size": int(len(outer_test_idx)),
                    "train_positives": int(y[outer_train_idx].sum()),
                    "inner_train_positives": int(y[inner_train_idx].sum()),
                    "val_positives": int(y[val_idx].sum()),
                    "test_positives": int(y[outer_test_idx].sum()),
                    "best_val_pr_auc": train_info["best_val_pr_auc"],
                    "best_epoch": train_info["best_epoch"],
                    "threshold": threshold,
                    "quality_filtering_label_dependent": bool(exp_cfg["quality_filtering"]["label_dependent"]),
                }
                row.update(metrics)
                fold_metric_rows.append(row)
                checkpoint_candidates[model_name].append(train_info)

                for local_i, row_idx in enumerate(outer_test_idx):
                    oof_rows.append({
                        "cohort_name": cohort_name,
                        "valid_ratio_threshold": valid_ratio_threshold,
                        "repeat": repeat,
                        "fold": fold,
                        "record_id": record_ids.iloc[row_idx]["record_id"],
                        "row_index": int(row_idx),
                        "y_true": int(y[row_idx]),
                        "y_proba": float(test_proba[local_i]),
                        "model": model_name,
                    })

        fold_metrics = pd.DataFrame(fold_metric_rows)
        oof_predictions = pd.DataFrame(oof_rows)
        averaged_oof = aggregate_repeated_oof(oof_predictions)
        bootstrap_ci = bootstrap_metric_ci(averaged_oof, threshold, bootstrap_iterations, seed)
        model_summaries = summarize_quality_oof(averaged_oof, threshold)

        fold_metrics.to_csv(cohort_dir / "fold_metrics.csv", index=False)
        oof_predictions.to_csv(cohort_dir / "oof_predictions.csv", index=False)
        bootstrap_ci.to_csv(cohort_dir / "bootstrap_confidence_intervals.csv", index=False)

        if not args.save_all_fold_models:
            models_dir = cohort_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            for model_name, candidates in checkpoint_candidates.items():
                best = max(candidates, key=lambda item: float(item["best_val_pr_auc"]))
                best_dir = models_dir / model_name / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_src = Path(best["checkpoint_path"])
                shutil.copy2(checkpoint_src, best_dir / checkpoint_src.name)
                if best.get("preprocessor_path"):
                    shutil.copy2(best["preprocessor_path"], best_dir / "preprocessors.joblib")
                (best_dir / "best_checkpoint_metadata.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
            shutil.rmtree(cohort_dir / "_fold_checkpoints")

        cv_summary = {
            "cohort_name": cohort_name,
            "valid_ratio_threshold": valid_ratio_threshold,
            "suffix": suffix,
            "label": exp_cfg["label"]["name"],
            "cohort_size": int(len(y)),
            "positive_count": int(y.sum()),
            "negative_count": int((y == 0).sum()),
            "models": models,
            "model_metrics_from_averaged_oof": model_summaries,
            "cv": {
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "seed": seed,
                "bootstrap_iterations": bootstrap_iterations,
            },
            "threshold": threshold,
            "config_snapshot": str(cohort_dir / "config_snapshot.json"),
            "quality_filtering_label_dependent": bool(exp_cfg["quality_filtering"]["label_dependent"]),
            "audit_source": exp_cfg["quality_filtering"]["audit_source"],
            "interpretation_note": exp_cfg["quality_filtering"]["interpretation_note"],
        }
        (cohort_dir / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")


def _json_numpy_scalar(obj: object) -> object:
    """json.dumps default= hook for the _progress/ files: numpy scalars (e.g. the
    int64 record_id pulled from the manifest) become Python scalars. Anything else
    still raises, so a genuinely unserialisable object is not silently stringified."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run_anytime_cohort(args: argparse.Namespace) -> None:
    """Trains ONE model per deep architecture (ctg_only/maternal_only/fusion/
    gated_fusion) using randomly-cropped elapsed-time lengths, then evaluates
    that same trained model at several fixed elapsed-time checkpoints on the
    held-out test folds -- replacing the earlier six-separate-fixed-window-
    cohorts design. Classical (logistic_regression/random_forest) baselines
    are retrained per checkpoint (cheap, and there's no "anytime" training
    trick that applies to a hand-crafted-feature model), on tabular features
    computed from the same checkpoint-cropped signal, through the identical
    split objects, so every model is compared fold-for-fold at every
    checkpoint."""
    exp_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(exp_cfg["base_config"]).read_text(encoding="utf-8"))
    processed_dir = Path(base_cfg["paths"]["processed_dir"])
    output_base = Path(exp_cfg["output"]["base_dir"])
    models = list(exp_cfg["models"])
    known_models = {
        "ctg_only", "maternal_only", "fusion", "gated_fusion",
        "logistic_regression", "random_forest",
    }
    if not set(models).issubset(known_models):
        raise ValueError(f"--anytime-cohort model names must be a subset of {sorted(known_models)}.")

    cv_cfg = exp_cfg["cross_validation"]
    n_splits = int(cv_cfg["n_splits"])
    n_repeats = int(cv_cfg["n_repeats"])
    seed = int(cv_cfg["seed"])
    # Step 7d seed-variance: separate the MODEL-training seed from the fold seed, so
    # repeated runs vary weight init / crop sampling while the folds stay IDENTICAL
    # (precompute_quality_splits keeps using `seed`). Absent key -> equals `seed` ->
    # bit-identical to every previous run.
    training_seed = int(cv_cfg.get("training_seed", seed))
    bootstrap_iterations = int(cv_cfg["bootstrap_iterations"])
    threshold = float(base_cfg["training"]["threshold"])
    val_size = float(base_cfg["training"]["val_size"])
    target_sample_rate_hz = int(base_cfg["signals"]["target_sample_rate_hz"])
    checkpoints_minutes = [float(m) for m in exp_cfg["checkpoints_minutes"]]
    min_crop_minutes = float(exp_cfg["anytime_training"]["min_crop_minutes"])
    max_crop_minutes = float(exp_cfg["anytime_training"]["max_crop_minutes"])
    # Phase 4: opt-in clinical CTG features for the tabular (LR/RF) models. Absent
    # key -> False -> identical to every existing cohort. Deep models are unaffected
    # (they read the raw signal, not tabular features).
    include_clinical = bool(exp_cfg.get("clinical_features", False))
    # Phase 3/6a/6b: optional deep-model overrides (encoder size, focal loss,
    # augmentation). Absent key -> None -> original deep behavior, unchanged.
    deep_overrides = exp_cfg.get("deep_model", None)
    # Step 7d: opt-in duration-aware training cutoffs. Absent key -> False -> every
    # prior anytime cohort reproduces exactly. When True, each training/validation
    # sample's crop upper bound is min(max_crop_minutes, that record's true duration),
    # so the model is never shown interpolated tail past the real end of a recording.
    # Spec: .../maternal_value/STEP7D_SPEC.md
    duration_aware_cutoffs = bool(exp_cfg.get("duration_aware_cutoffs", False))

    configured_cohorts = list(exp_cfg["cohorts"])
    if args.cohort:
        cohort_names = {str(cohort["name"]) for cohort in configured_cohorts}
        if args.cohort not in cohort_names:
            raise ValueError(f"Unknown cohort '{args.cohort}'. Available cohorts: {sorted(cohort_names)}")
        configured_cohorts = [cohort for cohort in configured_cohorts if str(cohort["name"]) == args.cohort]

    for cohort in configured_cohorts:
        cohort_name = str(cohort["name"])
        suffix = str(cohort["suffix"])
        valid_ratio_threshold = float(cohort["min_valid_ratio"])
        require_cohort_suffix(suffix)
        cohort_dir = output_base / cohort_name
        if getattr(args, "resume", False) and cohort_dir.exists():
            print(f"[resume] reusing existing cohort directory {cohort_dir}")
        else:
            ensure_absent_or_overwrite(cohort_dir, args.overwrite)
        cohort_dir.mkdir(parents=True, exist_ok=True)

        X_ctg, X_maternal, y, record_ids, manifest = load_cohort_arrays(processed_dir, suffix)
        manifest.to_csv(cohort_dir / "cohort_manifest.csv", index=False)

        # Step 7d: true recording duration per record, for duration-aware cutoffs.
        cohort_durations = None
        if duration_aware_cutoffs:
            source_hz = int(base_cfg["signals"]["source_sample_rate_hz"])
            cohort_durations = (
                record_ids["recording_length_source_samples"].to_numpy() / (source_hz * 60.0)
            )
            print(f"[7d] duration-aware cutoffs ON: durations min={cohort_durations.min():.2f} "
                  f"median={np.median(cohort_durations):.2f} max={cohort_durations.max():.2f} min")

        snapshot = {
            "base_config": base_cfg,
            "anytime_cohort_config": exp_cfg,
            "cohort": cohort,
            "checkpoints_minutes": checkpoints_minutes,
            "anytime_training": {"min_crop_minutes": min_crop_minutes, "max_crop_minutes": max_crop_minutes},
            "duration_aware_cutoffs": duration_aware_cutoffs,
            "fold_seed": seed,
            "training_seed": training_seed,
            "clinical_features": include_clinical,
            "deep_model_overrides": deep_overrides,
            "quality_filtering_label_dependent": bool(exp_cfg["quality_filtering"]["label_dependent"]),
            "audit_source": exp_cfg["quality_filtering"]["audit_source"],
            "interpretation_note": exp_cfg["quality_filtering"]["interpretation_note"],
        }
        (cohort_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

        splits = precompute_quality_splits(y, n_splits, n_repeats, seed, val_size)
        split_manifest = make_split_manifest(splits, record_ids, y, cohort_name, valid_ratio_threshold)
        split_manifest.to_csv(cohort_dir / "split_manifest.csv", index=False)

        fold_metric_rows = []
        oof_rows = []
        checkpoint_candidates: Dict[str, List[Dict[str, object]]] = {}
        fold_root = cohort_dir / ("models" if args.save_all_fold_models else "_fold_checkpoints")
        device = choose_device(str(base_cfg["training"]["device"]))
        batch_size = int(base_cfg["training"]["batch_size"])

        # Step 7d-seeds: per-fold progress checkpointing so a killed run resumes
        # instead of restarting. Each completed fold writes its own rows to
        # _progress/<fold_name>.json; on --resume those folds are loaded and skipped.
        # Absent --resume -> behaviour unchanged.
        progress_dir = cohort_dir / "_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        completed_folds: set[str] = set()
        if getattr(args, "resume", False):
            for pf in sorted(progress_dir.glob("*.json")):
                try:
                    payload = json.loads(pf.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    print(f"[resume] ignoring unreadable progress file {pf.name}")
                    continue
                fold_metric_rows.extend(payload["fold_metric_rows"])
                oof_rows.extend(payload["oof_rows"])
                for key, info in payload.get("checkpoint_candidates", {}).items():
                    checkpoint_candidates.setdefault(key, []).extend(info)
                completed_folds.add(payload["fold_name"])
            if completed_folds:
                print(f"[resume] loaded {len(completed_folds)} completed fold(s): "
                      f"{sorted(completed_folds)}")

        for split in splits:
            repeat = int(split["repeat"])
            fold = int(split["fold"])
            fold_name = str(split["fold_name"])
            if fold_name in completed_folds:
                print(f"[resume] skipping already-completed fold {fold_name}")
                continue
            fold_metric_rows_before = len(fold_metric_rows)
            oof_rows_before = len(oof_rows)
            candidates_before = {k: len(v) for k, v in checkpoint_candidates.items()}
            outer_train_idx = np.asarray(split["outer_train_idx"], dtype=int)
            outer_test_idx = np.asarray(split["outer_test_idx"], dtype=int)
            inner_train_idx = np.asarray(split["inner_train_idx"], dtype=int)
            val_idx = np.asarray(split["val_idx"], dtype=int)

            for model_name in models:
                if model_name in {"logistic_regression", "random_forest"}:
                    for checkpoint_minutes in checkpoints_minutes:
                        print(f"{cohort_name} | {fold_name} | {model_name} | {checkpoint_minutes}min")
                        cropped_raw_ctg = crop_ctg_array(X_ctg, checkpoint_minutes, target_sample_rate_hz)
                        X_tabular_ckpt = make_tabular_features(
                            cropped_raw_ctg, X_maternal,
                            include_clinical=include_clinical,
                            target_sample_rate_hz=target_sample_rate_hz,
                        )
                        key = f"{model_name}__ckpt{checkpoint_minutes:g}"
                        artifact_dir = fold_root / key / fold_name
                        _, train_info, test_proba, _ = train_sklearn_quality_fold(
                            model_name, X_tabular_ckpt, y, split, seed, artifact_dir,
                        )
                        metrics = evaluate_predictions(y[outer_test_idx], test_proba, threshold=threshold)
                        metrics.update(safe_ranking_metrics(y[outer_test_idx], test_proba))
                        row = {
                            "cohort_name": cohort_name, "valid_ratio_threshold": valid_ratio_threshold,
                            "label": exp_cfg["label"]["name"], "suffix": suffix, "model": model_name,
                            "checkpoint_minutes": checkpoint_minutes, "repeat": repeat, "fold": fold,
                            "train_size": int(len(outer_train_idx)), "inner_train_size": int(len(inner_train_idx)),
                            "val_size": int(len(val_idx)), "test_size": int(len(outer_test_idx)),
                            "train_positives": int(y[outer_train_idx].sum()), "inner_train_positives": int(y[inner_train_idx].sum()),
                            "val_positives": int(y[val_idx].sum()), "test_positives": int(y[outer_test_idx].sum()),
                            "best_val_pr_auc": train_info["best_val_pr_auc"], "best_epoch": train_info["best_epoch"],
                            "threshold": threshold,
                        }
                        row.update(metrics)
                        fold_metric_rows.append(row)
                        checkpoint_candidates.setdefault(key, []).append(train_info)
                        for local_i, row_idx in enumerate(outer_test_idx):
                            oof_rows.append({
                                "cohort_name": cohort_name, "valid_ratio_threshold": valid_ratio_threshold,
                                "checkpoint_minutes": checkpoint_minutes, "repeat": repeat, "fold": fold,
                                "record_id": record_ids.iloc[row_idx]["record_id"], "row_index": int(row_idx),
                                "y_true": int(y[row_idx]), "y_proba": float(test_proba[local_i]), "model": model_name,
                            })
                else:
                    print(f"{cohort_name} | {fold_name} | {model_name} | anytime-train (all checkpoints)")
                    artifact_dir = fold_root / model_name / fold_name
                    model, train_info, imputer, scaler, ctg_mean, ctg_std = train_deep_fold_anytime(
                        model_name, X_ctg, X_maternal, y, split, base_cfg, training_seed, artifact_dir,
                        target_sample_rate_hz, min_crop_minutes, max_crop_minutes,
                        deep_overrides=deep_overrides,
                        duration_minutes=cohort_durations,
                    )
                    checkpoint_candidates.setdefault(model_name, []).append(train_info)
                    X_ctg_norm_all, X_mat_norm_all = transform_arrays(X_ctg, X_maternal, imputer, scaler, ctg_mean, ctg_std)
                    test_mat = X_mat_norm_all[outer_test_idx]
                    for checkpoint_minutes in checkpoints_minutes:
                        cropped_test_ctg = crop_ctg_array(X_ctg_norm_all[outer_test_idx], checkpoint_minutes, target_sample_rate_hz)
                        test_loader = DataLoader(
                            CTGMaternalDataset(cropped_test_ctg, test_mat, y[outer_test_idx]),
                            batch_size=batch_size, shuffle=False,
                        )
                        test_proba = predict_model(model_name, model, test_loader, device)
                        metrics = evaluate_predictions(y[outer_test_idx], test_proba, threshold=threshold)
                        metrics.update(safe_ranking_metrics(y[outer_test_idx], test_proba))
                        row = {
                            "cohort_name": cohort_name, "valid_ratio_threshold": valid_ratio_threshold,
                            "label": exp_cfg["label"]["name"], "suffix": suffix, "model": model_name,
                            "checkpoint_minutes": checkpoint_minutes, "repeat": repeat, "fold": fold,
                            "train_size": int(len(outer_train_idx)), "inner_train_size": int(len(inner_train_idx)),
                            "val_size": int(len(val_idx)), "test_size": int(len(outer_test_idx)),
                            "train_positives": int(y[outer_train_idx].sum()), "inner_train_positives": int(y[inner_train_idx].sum()),
                            "val_positives": int(y[val_idx].sum()), "test_positives": int(y[outer_test_idx].sum()),
                            "best_val_pr_auc": train_info["best_val_pr_auc"], "best_epoch": train_info["best_epoch"],
                            "threshold": threshold,
                        }
                        row.update({k: v for k, v in train_info.items() if k.startswith("pretrain")})
                        row.update(metrics)
                        fold_metric_rows.append(row)
                        for local_i, row_idx in enumerate(outer_test_idx):
                            oof_rows.append({
                                "cohort_name": cohort_name, "valid_ratio_threshold": valid_ratio_threshold,
                                "checkpoint_minutes": checkpoint_minutes, "repeat": repeat, "fold": fold,
                                "record_id": record_ids.iloc[row_idx]["record_id"], "row_index": int(row_idx),
                                "y_true": int(y[row_idx]), "y_proba": float(test_proba[local_i]), "model": model_name,
                            })

            # Fold finished: persist its rows so a kill mid-run loses at most this fold.
            new_candidates = {}
            for key, vals in checkpoint_candidates.items():
                start = candidates_before.get(key, 0)
                if len(vals) > start:
                    new_candidates[key] = vals[start:]
            (progress_dir / f"{fold_name}.json").write_text(json.dumps({
                "fold_name": fold_name,
                "fold_metric_rows": fold_metric_rows[fold_metric_rows_before:],
                "oof_rows": oof_rows[oof_rows_before:],
                "checkpoint_candidates": new_candidates,
            }, indent=1, default=_json_numpy_scalar), encoding="utf-8")
            print(f"[progress] fold {fold_name} complete and saved", flush=True)

        fold_metrics = pd.DataFrame(fold_metric_rows)
        oof_predictions = pd.DataFrame(oof_rows)
        averaged_oof = aggregate_repeated_oof_anytime(oof_predictions)
        bootstrap_ci = bootstrap_metric_ci_anytime(averaged_oof, threshold, bootstrap_iterations, seed)
        model_summaries = summarize_oof_anytime(averaged_oof, threshold)

        fold_metrics.to_csv(cohort_dir / "fold_metrics.csv", index=False)
        oof_predictions.to_csv(cohort_dir / "oof_predictions.csv", index=False)
        bootstrap_ci.to_csv(cohort_dir / "bootstrap_confidence_intervals.csv", index=False)

        if not args.save_all_fold_models:
            models_dir = cohort_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            for key, candidates in checkpoint_candidates.items():
                best = max(candidates, key=lambda item: float(item["best_val_pr_auc"]))
                best_dir = models_dir / key / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_src = Path(best["checkpoint_path"])
                shutil.copy2(checkpoint_src, best_dir / checkpoint_src.name)
                if best.get("preprocessor_path"):
                    shutil.copy2(best["preprocessor_path"], best_dir / "preprocessors.joblib")
                (best_dir / "best_checkpoint_metadata.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
            shutil.rmtree(cohort_dir / "_fold_checkpoints")

        cv_summary = {
            "cohort_name": cohort_name,
            "valid_ratio_threshold": valid_ratio_threshold,
            "suffix": suffix,
            "label": exp_cfg["label"]["name"],
            "cohort_size": int(len(y)),
            "positive_count": int(y.sum()),
            "negative_count": int((y == 0).sum()),
            "models": models,
            "checkpoints_minutes": checkpoints_minutes,
            "anytime_training": {"min_crop_minutes": min_crop_minutes, "max_crop_minutes": max_crop_minutes},
            "model_metrics_from_averaged_oof": model_summaries,
            "cv": {
                "n_splits": n_splits, "n_repeats": n_repeats, "seed": seed,
                "bootstrap_iterations": bootstrap_iterations,
            },
            "threshold": threshold,
            "config_snapshot": str(cohort_dir / "config_snapshot.json"),
            "quality_filtering_label_dependent": bool(exp_cfg["quality_filtering"]["label_dependent"]),
            "audit_source": exp_cfg["quality_filtering"]["audit_source"],
            "interpretation_note": exp_cfg["quality_filtering"]["interpretation_note"],
        }
        (cohort_dir / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated stratified CV experiments for CTG acidemia models.")
    parser.add_argument("--config", default="configs/cv_experiments.yaml")
    parser.add_argument("--models", default=None, help="Comma-separated model names to run.")
    parser.add_argument("--epochs", type=int, default=None, help="Override deep-model epochs.")
    parser.add_argument("--max-folds", type=int, default=None, help="Limit folds for smoke testing.")
    parser.add_argument("--run-name", default=None, help="Output folder name under outputs/cv.")
    parser.add_argument("--labels", default=None, help="Comma-separated labels to run, e.g. y_acidemia_710.")
    parser.add_argument("--valid-ratios", default=None, help="Comma-separated valid-ratio thresholds, e.g. 0.60,0.70.")
    parser.add_argument("--n-repeats", type=int, default=None, help="Override repeated-CV repeats.")
    parser.add_argument("--plot-existing-run", default=None, help="Generate figures from an existing outputs/cv run directory.")
    parser.add_argument("--quality-cohorts", action="store_true", help="Run strict predefined signal-quality cohort CV.")
    parser.add_argument("--anytime-cohort", action="store_true", help="Train one any-elapsed-time model per architecture and evaluate at several fixed checkpoints.")
    parser.add_argument("--cohort", default=None, help="With --quality-cohorts/--anytime-cohort, run only one configured cohort name.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing quality-cohort output directories.")
    parser.add_argument("--save-all-fold-models", action="store_true", help="Keep every fold checkpoint instead of only the best per model/cohort.")
    parser.add_argument("--resume", action="store_true", help="With --anytime-cohort, reuse an existing cohort directory and skip folds already recorded in its _progress/ dir.")
    args = parser.parse_args()

    if args.anytime_cohort:
        run_anytime_cohort(args)
        return

    if args.quality_cohorts:
        run_quality_cohorts(args)
        return

    exp_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(exp_cfg["base_config"]).read_text(encoding="utf-8"))
    seed = int(exp_cfg["cross_validation"]["seed"])
    set_global_seed(seed)

    processed_dir = Path(base_cfg["paths"]["processed_dir"])
    outputs_base = Path(base_cfg["paths"]["outputs_dir"]) / "cv"
    if args.plot_existing_run:
        run_dir = Path(args.plot_existing_run)
        if not run_dir.is_absolute():
            run_dir = outputs_base / run_dir
        plot_existing_run(run_dir)
        return

    labels = requested_labels(exp_cfg["experiments"]["labels"], args.labels)
    valid_ratios = requested_valid_ratios(exp_cfg["experiments"]["valid_ratio_thresholds"], args.valid_ratios)
    models = requested_models(exp_cfg["models"], args.models)
    threshold = float(base_cfg["training"]["threshold"])
    epochs = int(args.epochs if args.epochs is not None else base_cfg["training"]["epochs"])
    run_name = args.run_name or default_run_name(models, epochs, args.max_folds)
    outputs_dir = outputs_base / run_name
    outputs_dir.mkdir(parents=True, exist_ok=True)

    n_splits = int(exp_cfg["cross_validation"]["n_splits"])
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=int(args.n_repeats if args.n_repeats is not None else exp_cfg["cross_validation"]["n_repeats"]),
        random_state=seed,
    )

    all_rows = []
    prediction_rows = []
    for label_col in labels:
        for valid_ratio in valid_ratios:
            suffix = make_suffix(label_col, valid_ratio)
            variant_dir = outputs_dir / suffix.lstrip("_")
            variant_dir.mkdir(parents=True, exist_ok=True)
            X_ctg, X_maternal, y, record_ids = load_variant(processed_dir, suffix)
            X_tabular = make_tabular_features(X_ctg, X_maternal)

            fold_counter = 0
            for split_number, (train_idx, test_idx) in enumerate(cv.split(X_tabular, y), start=1):
                repeat = ((split_number - 1) // n_splits) + 1
                fold = ((split_number - 1) % n_splits) + 1
                fold_name = f"repeat{repeat:02d}_fold{fold:02d}"

                for model_name in models:
                    print(f"{suffix} | {fold_name} | {model_name}")
                    if model_name in {"logistic_regression", "random_forest"}:
                        metrics, test_proba = run_sklearn_fold(model_name, X_tabular, y, train_idx, test_idx, seed, threshold)
                    else:
                        metrics, test_proba = run_deep_fold(
                            model_name,
                            X_ctg,
                            X_maternal,
                            y,
                            train_idx,
                            test_idx,
                            base_cfg,
                            seed,
                            threshold,
                            epochs,
                            variant_dir,
                            fold_name,
                        )

                    row = {
                        "label": label_col,
                        "valid_ratio": valid_ratio,
                        "suffix": suffix,
                        "model": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        "train_size": int(len(train_idx)),
                        "test_size": int(len(test_idx)),
                        "train_positives": int(y[train_idx].sum()),
                        "test_positives": int(y[test_idx].sum()),
                    }
                    row.update(metrics)
                    all_rows.append(row)

                    for local_i, row_idx in enumerate(test_idx):
                        prediction_rows.append({
                            "label": label_col,
                            "valid_ratio": valid_ratio,
                            "suffix": suffix,
                            "model": model_name,
                            "repeat": repeat,
                            "fold": fold,
                            "row_index": int(row_idx),
                            "record_id": record_ids.iloc[row_idx]["record_id"],
                            "y_true": int(y[row_idx]),
                            "y_proba": float(test_proba[local_i]),
                        })

                fold_counter += 1
                if args.max_folds is not None and fold_counter >= args.max_folds:
                    break

    fold_metrics = pd.DataFrame(all_rows)
    fold_predictions = pd.DataFrame(prediction_rows)
    summary = summarize_metrics(fold_metrics)
    fold_metrics.to_csv(outputs_dir / "fold_metrics.csv", index=False)
    fold_predictions.to_csv(outputs_dir / "fold_predictions.csv", index=False)
    summary.to_csv(outputs_dir / "summary_metrics.csv", index=False)
    save_cv_summary_plots(summary, outputs_dir)
    save_prediction_curve_plots(fold_predictions, outputs_dir)
    (outputs_dir / "cv_run_config.json").write_text(
        json.dumps({"models": models, "epochs": epochs, "config": exp_cfg}, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
