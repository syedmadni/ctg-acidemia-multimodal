from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.read_wfdb import read_record_signals


STATIC_MATERNAL_COLUMNS = [
    "age",
    "gravidity",
    "parity",
    "gest_weeks",
    "diabetes",
    "hypertension",
    "preeclampsia",
]

TIMING_UNCERTAIN_COLUMNS = [
    "induced",
    "meconium",
    "pyrexia",
    "liq_praecox",
    "presentation",
    "i_stage",
    "ii_stage",
    "no_progress",
    "ck_kp",
    "deliv_type",
    "apgar1",
    "apgar5",
    "be",
    "weight_g",
    "sex",
]

EXCLUSION_TYPES = [
    "retained",
    "short_record_only",
    "low_quality_only",
    "short_record_and_low_quality",
    "missing_or_invalid_signal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit FHR valid-ratio filtering for CTU-UHB pH-labeled records."
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--label", default=None, help="Label column, e.g. y_acidemia_705.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Current padded-window valid-ratio retention threshold.",
    )
    parser.add_argument(
        "--raw-quality-reference-threshold",
        type=float,
        default=None,
        help=(
            "Provisional observed-window quality reference used only to classify "
            "short_record_only versus short_record_and_low_quality."
        ),
    )
    return parser.parse_args()


def finite_positive_ratio(x: np.ndarray) -> Tuple[int, int, float]:
    total = int(len(x))
    if total == 0:
        return 0, 0, float("nan")
    valid = int((np.isfinite(x) & (x > 0)).sum())
    return valid, total, float(valid / total)


def target_window_indices(length: int, window_samples: int) -> Tuple[int, int, int]:
    if length >= window_samples:
        return length - window_samples, length, 0
    return 0, length, window_samples - length


def classify_exclusion(
    is_signal_valid: bool,
    is_short_record: bool,
    retained: bool,
    observed_ratio: float,
    raw_quality_reference_threshold: float,
) -> str:
    if not is_signal_valid:
        return "missing_or_invalid_signal"
    if retained:
        return "retained"
    observed_is_low_quality = (
        not np.isfinite(observed_ratio)
        or observed_ratio < raw_quality_reference_threshold
    )
    if is_short_record and observed_is_low_quality:
        return "short_record_and_low_quality"
    if is_short_record:
        return "short_record_only"
    return "low_quality_only"


def audit_one_record(
    row: pd.Series,
    raw_dir: Path,
    label_col: str,
    fhr_channel: int,
    uc_channel: int,
    source_hz: int,
    window_minutes: int,
    threshold: float,
    raw_quality_reference_threshold: float,
) -> Dict[str, object]:
    rid = str(row["record_id"])
    window_samples = window_minutes * 60 * source_hz
    base: Dict[str, object] = {
        "record_id": rid,
        "ph": row.get("ph", np.nan),
        "primary_label": int(row[label_col]),
        "primary_label_name": "acidemia" if int(row[label_col]) == 1 else "normal",
        "configured_window_minutes": window_minutes,
        "configured_source_hz": source_hz,
        "configured_window_samples": window_samples,
        "current_threshold": threshold,
        "current_filter_basis": "padded_target_window_valid_ratio",
        "raw_quality_reference_threshold": raw_quality_reference_threshold,
        "signal_error": "",
    }

    for col in STATIC_MATERNAL_COLUMNS + TIMING_UNCERTAIN_COLUMNS:
        base[col] = row.get(col, np.nan)

    try:
        fhr, uc, fs = read_record_signals(raw_dir, rid, fhr_channel, uc_channel)
        if len(fhr) == 0 or len(uc) == 0:
            raise ValueError("empty_signal")

        start, end, padding_samples = target_window_indices(len(fhr), window_samples)
        observed_fhr = fhr[start:end]
        padded_fhr = np.pad(
            observed_fhr,
            (padding_samples, 0),
            mode="constant",
            constant_values=0,
        )
        observed_valid, observed_total, observed_ratio = finite_positive_ratio(observed_fhr)
        padded_valid, padded_total, padded_ratio = finite_positive_ratio(padded_fhr)
        is_signal_valid = observed_total > 0 and np.isfinite(observed_ratio)
        is_short_record = len(fhr) < window_samples
        retained = is_signal_valid and padded_ratio >= threshold
        exclusion_type = classify_exclusion(
            is_signal_valid,
            is_short_record,
            retained,
            observed_ratio,
            raw_quality_reference_threshold,
        )

        base.update(
            {
                "wfdb_fs": fs,
                "full_record_samples": int(len(fhr)),
                "full_record_duration_minutes": float(len(fhr) / source_hz / 60.0),
                "target_window_start_source_index": int(start),
                "target_window_end_source_index": int(end),
                "observed_target_window_samples": int(observed_total),
                "observed_target_window_minutes": float(observed_total / source_hz / 60.0),
                "padding_samples": int(padding_samples),
                "padding_minutes": float(padding_samples / source_hz / 60.0),
                "is_short_record": bool(is_short_record),
                "observed_target_window_valid_samples": int(observed_valid),
                "observed_target_window_total_samples": int(observed_total),
                "observed_target_window_valid_ratio": observed_ratio,
                "padded_target_window_valid_samples": int(padded_valid),
                "padded_target_window_total_samples": int(padded_total),
                "padded_target_window_valid_ratio": padded_ratio,
                "retained_at_current_threshold": bool(retained),
                "exclusion_type": exclusion_type,
            }
        )
    except Exception as exc:
        base.update(
            {
                "wfdb_fs": np.nan,
                "full_record_samples": np.nan,
                "full_record_duration_minutes": np.nan,
                "target_window_start_source_index": np.nan,
                "target_window_end_source_index": np.nan,
                "observed_target_window_samples": np.nan,
                "observed_target_window_minutes": np.nan,
                "padding_samples": np.nan,
                "padding_minutes": np.nan,
                "is_short_record": np.nan,
                "observed_target_window_valid_samples": np.nan,
                "observed_target_window_total_samples": np.nan,
                "observed_target_window_valid_ratio": np.nan,
                "padded_target_window_valid_samples": np.nan,
                "padded_target_window_total_samples": np.nan,
                "padded_target_window_valid_ratio": np.nan,
                "retained_at_current_threshold": False,
                "exclusion_type": "missing_or_invalid_signal",
                "signal_error": repr(exc),
            }
        )
    return base


def summarize_continuous(df: pd.DataFrame, value_col: str, group_col: str = "exclusion_type") -> pd.DataFrame:
    rows = []
    for group in EXCLUSION_TYPES:
        values = pd.to_numeric(df.loc[df[group_col] == group, value_col], errors="coerce")
        nonmissing = values.dropna()
        if len(nonmissing) == 0:
            rows.append(
                {
                    group_col: group,
                    "n": int(len(values)),
                    "nonmissing_n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "iqr": np.nan,
                    "min": np.nan,
                    "p05": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                    "max": np.nan,
                    "missing_n": int(values.isna().sum()),
                }
            )
            continue
        rows.append(
            {
                group_col: group,
                "n": int(len(values)),
                "nonmissing_n": int(len(nonmissing)),
                "mean": float(nonmissing.mean()),
                "std": float(nonmissing.std(ddof=1)) if len(nonmissing) > 1 else 0.0,
                "median": float(nonmissing.median()),
                "iqr": float(nonmissing.quantile(0.75) - nonmissing.quantile(0.25)),
                "min": float(nonmissing.min()),
                "p05": float(nonmissing.quantile(0.05)),
                "p25": float(nonmissing.quantile(0.25)),
                "p75": float(nonmissing.quantile(0.75)),
                "p95": float(nonmissing.quantile(0.95)),
                "max": float(nonmissing.max()),
                "missing_n": int(values.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_metadata_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    group_col: str = "exclusion_type",
) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        numeric_values = pd.to_numeric(df[col], errors="coerce")
        for group in EXCLUSION_TYPES:
            mask = df[group_col] == group
            raw_values = df.loc[mask, col]
            numeric_group = numeric_values.loc[mask]
            nonmissing_raw = raw_values.dropna()
            nonmissing_numeric = numeric_group.dropna()
            value_counts = nonmissing_raw.astype(str).value_counts(dropna=True).to_dict()
            rows.append(
                {
                    "variable": col,
                    group_col: group,
                    "n": int(mask.sum()),
                    "nonmissing_n": int(nonmissing_raw.shape[0]),
                    "missing_n": int(raw_values.isna().sum()),
                    "missing_percent": float(raw_values.isna().mean()) if mask.sum() else np.nan,
                    "numeric_nonmissing_n": int(nonmissing_numeric.shape[0]),
                    "mean": float(nonmissing_numeric.mean()) if len(nonmissing_numeric) else np.nan,
                    "std": float(nonmissing_numeric.std(ddof=1)) if len(nonmissing_numeric) > 1 else np.nan,
                    "median": float(nonmissing_numeric.median()) if len(nonmissing_numeric) else np.nan,
                    "p25": float(nonmissing_numeric.quantile(0.25)) if len(nonmissing_numeric) else np.nan,
                    "p75": float(nonmissing_numeric.quantile(0.75)) if len(nonmissing_numeric) else np.nan,
                    "min": float(nonmissing_numeric.min()) if len(nonmissing_numeric) else np.nan,
                    "max": float(nonmissing_numeric.max()) if len(nonmissing_numeric) else np.nan,
                    "value_counts_json": json.dumps(value_counts, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def summarize_missingness(
    df: pd.DataFrame,
    columns: Iterable[str],
    variable_group: str,
    group_col: str = "exclusion_type",
) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        for group in EXCLUSION_TYPES:
            values = df.loc[df[group_col] == group, col]
            rows.append(
                {
                    "variable_group": variable_group,
                    "variable": col,
                    group_col: group,
                    "n": int(len(values)),
                    "missing_n": int(values.isna().sum()),
                    "missing_percent": float(values.isna().mean()) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def cohort_flow(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"stage": "primary_label_records", "n": int(len(df))},
        {"stage": "retained", "n": int((df["exclusion_type"] == "retained").sum())},
        {"stage": "excluded_total", "n": int((df["exclusion_type"] != "retained").sum())},
    ]
    for exclusion_type in EXCLUSION_TYPES[1:]:
        rows.append(
            {
                "stage": exclusion_type,
                "n": int((df["exclusion_type"] == exclusion_type).sum()),
            }
        )
    return pd.DataFrame(rows)


def odds_or_risk_stats(acidemia_exposed: int, normal_exposed: int, acidemia_ref: int, normal_ref: int) -> Dict[str, float]:
    # Haldane-Anscombe correction keeps small-sample estimates finite.
    a = acidemia_exposed + 0.5
    b = normal_exposed + 0.5
    c = acidemia_ref + 0.5
    d = normal_ref + 0.5
    risk_exposed = a / (a + b)
    risk_ref = c / (c + d)
    rr = risk_exposed / risk_ref
    se_log_rr = math.sqrt((1 / a) - (1 / (a + b)) + (1 / c) - (1 / (c + d)))
    ci_low = math.exp(math.log(rr) - 1.96 * se_log_rr)
    ci_high = math.exp(math.log(rr) + 1.96 * se_log_rr)
    odds_ratio = (a * d) / (b * c)
    return {
        "risk_ratio_vs_retained": float(rr),
        "risk_ratio_ci_low": float(ci_low),
        "risk_ratio_ci_high": float(ci_high),
        "odds_ratio_vs_retained": float(odds_ratio),
    }


def fisher_exact_p(table: np.ndarray) -> float:
    try:
        from scipy.stats import fisher_exact

        return float(fisher_exact(table)[1])
    except Exception:
        return float("nan")


def label_by_exclusion_type(df: pd.DataFrame) -> pd.DataFrame:
    retained = df[df["exclusion_type"] == "retained"]
    ref_acidemia = int((retained["primary_label"] == 1).sum())
    ref_normal = int((retained["primary_label"] == 0).sum())
    rows = []
    for exclusion_type in EXCLUSION_TYPES:
        group = df[df["exclusion_type"] == exclusion_type]
        acidemia = int((group["primary_label"] == 1).sum())
        normal = int((group["primary_label"] == 0).sum())
        n = int(len(group))
        row = {
            "exclusion_type": exclusion_type,
            "n": n,
            "acidemia_n": acidemia,
            "normal_n": normal,
            "acidemia_prevalence": float(acidemia / n) if n else np.nan,
        }
        if exclusion_type == "retained" or n == 0 or len(retained) == 0:
            row.update(
                {
                    "risk_ratio_vs_retained": 1.0 if exclusion_type == "retained" else np.nan,
                    "risk_ratio_ci_low": np.nan,
                    "risk_ratio_ci_high": np.nan,
                    "odds_ratio_vs_retained": 1.0 if exclusion_type == "retained" else np.nan,
                    "fisher_exact_p_vs_retained": np.nan,
                }
            )
        else:
            row.update(odds_or_risk_stats(acidemia, normal, ref_acidemia, ref_normal))
            row["fisher_exact_p_vs_retained"] = fisher_exact_p(
                np.asarray([[acidemia, normal], [ref_acidemia, ref_normal]], dtype=int)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def label_by_retention(df: pd.DataFrame) -> pd.DataFrame:
    retained = df[df["exclusion_type"] == "retained"]
    excluded = df[df["exclusion_type"] != "retained"]
    ref_acidemia = int((retained["primary_label"] == 1).sum())
    ref_normal = int((retained["primary_label"] == 0).sum())
    rows = []
    for retention_group, group in [("retained", retained), ("excluded", excluded)]:
        acidemia = int((group["primary_label"] == 1).sum())
        normal = int((group["primary_label"] == 0).sum())
        n = int(len(group))
        row = {
            "retention_group": retention_group,
            "n": n,
            "acidemia_n": acidemia,
            "normal_n": normal,
            "acidemia_prevalence": float(acidemia / n) if n else np.nan,
            "risk_ratio_ci_method": "haldane_anscombe_log_rr",
        }
        if retention_group == "retained" or n == 0 or len(retained) == 0:
            row.update(
                {
                    "risk_ratio_excluded_vs_retained": 1.0 if retention_group == "retained" else np.nan,
                    "risk_ratio_ci_low": np.nan,
                    "risk_ratio_ci_high": np.nan,
                    "odds_ratio_excluded_vs_retained": 1.0 if retention_group == "retained" else np.nan,
                    "fisher_exact_p": np.nan,
                }
            )
        else:
            stats = odds_or_risk_stats(acidemia, normal, ref_acidemia, ref_normal)
            row.update(
                {
                    "risk_ratio_excluded_vs_retained": stats["risk_ratio_vs_retained"],
                    "risk_ratio_ci_low": stats["risk_ratio_ci_low"],
                    "risk_ratio_ci_high": stats["risk_ratio_ci_high"],
                    "odds_ratio_excluded_vs_retained": stats["odds_ratio_vs_retained"],
                    "fisher_exact_p": fisher_exact_p(
                        np.asarray([[acidemia, normal], [ref_acidemia, ref_normal]], dtype=int)
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_sensitivity(df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    valid_df = df[df["exclusion_type"] != "missing_or_invalid_signal"].copy()
    for basis in ["observed_target_window_valid_ratio", "padded_target_window_valid_ratio"]:
        for threshold in thresholds:
            retained_mask = pd.to_numeric(valid_df[basis], errors="coerce") >= threshold
            retained = valid_df[retained_mask]
            excluded = valid_df[~retained_mask]
            retained_acidemia = int((retained["primary_label"] == 1).sum())
            retained_normal = int((retained["primary_label"] == 0).sum())
            excluded_acidemia = int((excluded["primary_label"] == 1).sum())
            excluded_normal = int((excluded["primary_label"] == 0).sum())
            stats = (
                odds_or_risk_stats(excluded_acidemia, excluded_normal, retained_acidemia, retained_normal)
                if len(retained) and len(excluded)
                else {
                    "risk_ratio_vs_retained": np.nan,
                    "risk_ratio_ci_low": np.nan,
                    "risk_ratio_ci_high": np.nan,
                    "odds_ratio_vs_retained": np.nan,
                }
            )
            rows.append(
                {
                    "basis": basis,
                    "threshold": float(threshold),
                    "retained_n": int(len(retained)),
                    "excluded_n": int(len(excluded)),
                    "retained_acidemia_n": retained_acidemia,
                    "retained_normal_n": retained_normal,
                    "excluded_acidemia_n": excluded_acidemia,
                    "excluded_normal_n": excluded_normal,
                    "retained_acidemia_prevalence": float(retained_acidemia / len(retained)) if len(retained) else np.nan,
                    "excluded_acidemia_prevalence": float(excluded_acidemia / len(excluded)) if len(excluded) else np.nan,
                    "risk_ratio_excluded_vs_retained": stats["risk_ratio_vs_retained"],
                    "risk_ratio_ci_low": stats["risk_ratio_ci_low"],
                    "risk_ratio_ci_high": stats["risk_ratio_ci_high"],
                    "odds_ratio_excluded_vs_retained": stats["odds_ratio_vs_retained"],
                    "fisher_exact_p": fisher_exact_p(
                        np.asarray(
                            [
                                [excluded_acidemia, excluded_normal],
                                [retained_acidemia, retained_normal],
                            ],
                            dtype=int,
                        )
                    )
                    if len(retained) and len(excluded)
                    else np.nan,
                    "median_duration_retained": float(retained["full_record_duration_minutes"].median()) if len(retained) else np.nan,
                    "median_duration_excluded": float(excluded["full_record_duration_minutes"].median()) if len(excluded) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_bar_counts(flow: pd.DataFrame, out_path: Path) -> None:
    plot_df = flow[flow["stage"] != "primary_label_records"].copy()
    plt.figure(figsize=(8, 4.8))
    plt.bar(plot_df["stage"], plot_df["n"], color="#4C78A8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Records")
    plt.title("Primary pH<7.05 Signal-Quality Cohort Flow")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_label_exclusion_plot(label_table: pd.DataFrame, out_path: Path) -> None:
    plot_df = label_table.copy()
    x = np.arange(len(plot_df))
    plt.figure(figsize=(8, 4.8))
    plt.bar(x, plot_df["normal_n"], label="normal", color="#72B7B2")
    plt.bar(x, plot_df["acidemia_n"], bottom=plot_df["normal_n"], label="acidemia", color="#E45756")
    plt.xticks(x, plot_df["exclusion_type"], rotation=35, ha="right")
    plt.ylabel("Records")
    plt.title("Primary Label Class by Exclusion Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_boxplot(df: pd.DataFrame, value_col: str, title: str, ylabel: str, out_path: Path) -> None:
    data = [
        pd.to_numeric(df.loc[df["exclusion_type"] == group, value_col], errors="coerce").dropna().to_numpy()
        for group in EXCLUSION_TYPES
    ]
    plt.figure(figsize=(8, 4.8))
    plt.boxplot(data, tick_labels=EXCLUSION_TYPES, showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path, threshold: float | None = None) -> None:
    colors = df["primary_label"].map({0: "#4C78A8", 1: "#E45756"}).fillna("#999999")
    plt.figure(figsize=(6.6, 5.4))
    plt.scatter(df[x_col], df[y_col], c=colors, alpha=0.72, edgecolors="none")
    if threshold is not None:
        plt.axhline(threshold, color="0.25", linestyle="--", linewidth=1.1)
    plt.xlabel(x_col.replace("_", " "))
    plt.ylabel(y_col.replace("_", " "))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_threshold_plots(sens: pd.DataFrame, figures_dir: Path) -> None:
    for basis, group in sens.groupby("basis"):
        stem = "observed" if basis.startswith("observed") else "padded"
        plt.figure(figsize=(7, 4.6))
        plt.plot(group["threshold"], group["retained_n"], marker="o", label="retained total")
        plt.plot(group["threshold"], group["retained_acidemia_n"], marker="o", label="retained acidemia")
        plt.xlabel("Valid-ratio threshold")
        plt.ylabel("Records")
        plt.title(f"Threshold Sensitivity Counts ({stem})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"primary_705_threshold_sensitivity_counts_{stem}.png", dpi=220)
        plt.close()

        plt.figure(figsize=(7, 4.6))
        plt.plot(group["threshold"], group["retained_acidemia_prevalence"], marker="o", label="retained")
        plt.plot(group["threshold"], group["excluded_acidemia_prevalence"], marker="o", label="excluded")
        plt.xlabel("Valid-ratio threshold")
        plt.ylabel("Acidemia prevalence")
        plt.ylim(0, 1)
        plt.title(f"Threshold Sensitivity Prevalence ({stem})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"primary_705_threshold_sensitivity_prevalence_{stem}.png", dpi=220)
        plt.close()


def save_missingness_plot(missingness: pd.DataFrame, title: str, out_path: Path) -> None:
    if missingness.empty:
        return
    pivot = missingness.pivot(index="variable", columns="exclusion_type", values="missing_percent")
    pivot = pivot.reindex(columns=EXCLUSION_TYPES)
    plt.figure(figsize=(8.5, max(3.8, 0.35 * len(pivot))))
    image = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(image, label="Missing fraction")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_outputs(df: pd.DataFrame, out_dir: Path, threshold: float) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "primary_705_valid_ratio_audit_records.csv", index=False)
    df[df["exclusion_type"] == "retained"].to_csv(out_dir / "primary_705_retained_records.csv", index=False)
    df[df["exclusion_type"] != "retained"].to_csv(out_dir / "primary_705_excluded_records.csv", index=False)
    df[df["exclusion_type"] != "retained"].to_csv(out_dir / "primary_705_excluded_by_type.csv", index=False)

    flow = cohort_flow(df)
    labels = label_by_exclusion_type(df)
    retention_labels = label_by_retention(df)
    flow.to_csv(out_dir / "primary_705_cohort_flow.csv", index=False)
    labels.to_csv(out_dir / "primary_705_label_by_exclusion_type.csv", index=False)
    retention_labels.to_csv(out_dir / "primary_705_label_by_retention.csv", index=False)

    summarize_continuous(df, "ph").to_csv(out_dir / "primary_705_ph_summary_by_exclusion_type.csv", index=False)
    summarize_continuous(df, "full_record_duration_minutes").to_csv(
        out_dir / "primary_705_duration_summary_by_exclusion_type.csv", index=False
    )
    summarize_continuous(df, "observed_target_window_valid_ratio").to_csv(
        out_dir / "primary_705_observed_valid_ratio_summary_by_exclusion_type.csv", index=False
    )
    summarize_continuous(df, "padded_target_window_valid_ratio").to_csv(
        out_dir / "primary_705_padded_valid_ratio_summary_by_exclusion_type.csv", index=False
    )
    static_summary = summarize_metadata_columns(df, STATIC_MATERNAL_COLUMNS)
    timing_summary = summarize_metadata_columns(df, TIMING_UNCERTAIN_COLUMNS)
    static_missing = summarize_missingness(df, STATIC_MATERNAL_COLUMNS, "static_maternal")
    timing_missing = summarize_missingness(df, TIMING_UNCERTAIN_COLUMNS, "timing_uncertain")
    static_summary.to_csv(out_dir / "primary_705_static_maternal_summary_by_exclusion_type.csv", index=False)
    timing_summary.to_csv(out_dir / "primary_705_timing_uncertain_metadata_summary_by_exclusion_type.csv", index=False)
    pd.concat([static_missing, timing_missing], ignore_index=True).to_csv(
        out_dir / "primary_705_missingness_by_exclusion_type.csv", index=False
    )

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
    sens = threshold_sensitivity(df, thresholds)
    sens.to_csv(out_dir / "primary_705_threshold_sensitivity.csv", index=False)

    save_bar_counts(flow, figures_dir / "primary_705_cohort_flow.png")
    save_label_exclusion_plot(labels, figures_dir / "primary_705_label_by_exclusion_type.png")
    save_boxplot(df, "ph", "pH by Exclusion Type", "pH", figures_dir / "primary_705_ph_by_exclusion_type.png")
    save_boxplot(
        df,
        "full_record_duration_minutes",
        "Full Record Duration by Exclusion Type",
        "Minutes",
        figures_dir / "primary_705_duration_by_exclusion_type.png",
    )
    save_boxplot(
        df,
        "observed_target_window_valid_ratio",
        "Observed Target-Window Valid Ratio by Exclusion Type",
        "Observed valid ratio",
        figures_dir / "primary_705_observed_valid_ratio_by_exclusion_type.png",
    )
    save_boxplot(
        df,
        "padded_target_window_valid_ratio",
        "Padded Target-Window Valid Ratio by Exclusion Type",
        "Padded valid ratio",
        figures_dir / "primary_705_padded_valid_ratio_by_exclusion_type.png",
    )
    save_scatter(
        df,
        "observed_target_window_valid_ratio",
        "padded_target_window_valid_ratio",
        "Observed vs Padded Target-Window FHR Valid Ratio",
        figures_dir / "primary_705_observed_vs_padded_valid_ratio_scatter.png",
        threshold=threshold,
    )
    save_scatter(
        df,
        "full_record_duration_minutes",
        "padded_target_window_valid_ratio",
        "Duration vs Padded Target-Window Valid Ratio",
        figures_dir / "primary_705_duration_vs_padded_valid_ratio.png",
        threshold=threshold,
    )
    save_threshold_plots(sens, figures_dir)
    save_missingness_plot(
        static_missing,
        "Static Maternal Missingness by Exclusion Type",
        figures_dir / "primary_705_static_maternal_missingness.png",
    )
    save_missingness_plot(
        timing_missing,
        "Timing-Uncertain Metadata Missingness by Exclusion Type",
        figures_dir / "primary_705_timing_uncertain_metadata_missingness.png",
    )


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"]) / "audit"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    label_col = args.label or cfg["labels"]["primary_name"]
    threshold = float(args.threshold if args.threshold is not None else cfg["signals"]["min_fhr_valid_ratio"])
    raw_quality_reference_threshold = float(
        args.raw_quality_reference_threshold
        if args.raw_quality_reference_threshold is not None
        else threshold
    )
    source_hz = int(cfg["signals"]["source_sample_rate_hz"])
    window_minutes = int(cfg["signals"]["window_minutes"])
    fhr_channel = int(cfg["signals"].get("fhr_channel", 0))
    uc_channel = int(cfg["signals"].get("uc_channel", 1))

    labels_path = processed_dir / "record_meta_labels_acidemia.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}. Run build_labels.py first.")

    meta = pd.read_csv(labels_path)
    meta["record_id"] = meta["record_id"].astype(int).astype(str)
    if label_col not in meta.columns:
        raise ValueError(f"Label column not found in {labels_path}: {label_col}")
    primary = meta.dropna(subset=[label_col]).copy()
    primary[label_col] = primary[label_col].astype(int)

    rows = [
        audit_one_record(
            row,
            raw_dir,
            label_col,
            fhr_channel,
            uc_channel,
            source_hz,
            window_minutes,
            threshold,
            raw_quality_reference_threshold,
        )
        for _, row in primary.iterrows()
    ]
    audit_df = pd.DataFrame(rows)
    audit_df["exclusion_type"] = pd.Categorical(
        audit_df["exclusion_type"],
        categories=EXCLUSION_TYPES,
        ordered=True,
    )
    audit_df = audit_df.sort_values(["exclusion_type", "record_id"]).reset_index(drop=True)

    write_outputs(audit_df, outputs_dir, threshold)
    print(
        cohort_flow(audit_df).to_string(index=False)
    )


if __name__ == "__main__":
    main()
