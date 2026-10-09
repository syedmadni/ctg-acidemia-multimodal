"""Ensemble weight/pairing sweep -- the generalisation of run_ensemble_analysis.py.

run_ensemble_analysis.py tests exactly one blend: 0.5*deep + 0.5*logistic_regression,
probability-averaged. This sweeps the three axes that fixes arbitrarily:

  1. blend weight w (deep gets w, tabular gets 1-w), w in [0, 1]
  2. tabular partner: logistic_regression AND random_forest, not just LR
  3. combination method: probability averaging AND rank averaging

Rank averaging matters here because the models are differently calibrated -- LR
outputs well-spread probabilities while random_forest clusters near 0 at this
prevalence (see CLAUDE.md's Brier-score caveat). Averaging raw probabilities lets
the better-calibrated model dominate the blend for reasons unrelated to ranking
skill; averaging ranks removes that confound. Both are reported.

Written as a NEW script rather than an extension of run_ensemble_analysis.py so
the previously reported 0.5/0.5 numbers stay byte-reproducible from the original
code path -- same convention as run_anytime_cohort vs run_quality_cohorts.

NO retraining: this is pure post-hoc combination of OOF predictions that already
exist on disk.

*** SELECTION-BIAS WARNING, read before citing any number this produces ***
This sweep evaluates hundreds of (weight, pairing, method) combinations on the
SAME out-of-fold predictions used to pick the winner. The best combination's
score is therefore optimistically biased -- it is a selected maximum, not a
held-out estimate. The honest way to report it is (a) alongside the
pre-specified 0.5/0.5 baseline, and (b) with the selection procedure disclosed
in Limitations. The script prints the gap between "best selected" and
"pre-specified 0.5" for exactly this reason: a large gap means most of the
apparent gain is selection noise.

Usage:
  python -m src.analysis.run_ensemble_sweep --cohort-dir outputs/quality_cohorts/anytime_cap64 \
      --tabular-cohort-dir outputs/quality_cohorts/primary_705_anytime_vr060
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.training.train_cv import aggregate_repeated_oof_anytime, bootstrap_metric_ci_anytime
from src.utils.metrics import binary_metrics

CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]
DEEP_MODELS = ["ctg_only", "fusion", "gated_fusion"]
TABULAR_MODELS = ["logistic_regression", "random_forest"]
WEIGHTS = [round(w, 2) for w in np.arange(0.0, 1.01, 0.05)]
METHODS = ["prob", "rank"]
THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 1000
SEED = 42
BASELINE_WEIGHT = 0.5
BASELINE_METHOD = "prob"


def load_averaged(cohort_dir: Path) -> pd.DataFrame:
    oof = pd.read_csv(cohort_dir / "oof_predictions.csv")
    return aggregate_repeated_oof_anytime(oof)


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    """Rank-transform to [0, 1]. Ties get their average rank, matching how AUC
    itself treats ties, so a rank-averaged blend stays consistent with the
    metric it is being scored on."""
    order = pd.Series(values).rank(method="average").to_numpy()
    if len(order) <= 1:
        return np.zeros_like(order, dtype=float)
    return (order - 1.0) / (len(order) - 1.0)


def blend(deep: np.ndarray, tabular: np.ndarray, weight: float, method: str) -> np.ndarray:
    if method == "prob":
        return weight * deep + (1.0 - weight) * tabular
    if method == "rank":
        return weight * normalized_ranks(deep) + (1.0 - weight) * normalized_ranks(tabular)
    raise ValueError(f"Unknown combination method: {method}")


def paired_frame(
    deep_avg: pd.DataFrame,
    tabular_avg: pd.DataFrame,
    deep_model: str,
    tabular_model: str,
    checkpoint: float,
) -> pd.DataFrame:
    """Join one deep model's OOF to one tabular model's OOF at one checkpoint,
    on record_id. Raises if the two disagree on any label -- that would mean the
    cohorts do not actually share held-out records and every blend below would
    be silently meaningless."""
    d = deep_avg[
        (deep_avg["model"] == deep_model) & (deep_avg["checkpoint_minutes"] == checkpoint)
    ][["record_id", "y_true", "y_proba"]].rename(columns={"y_proba": "proba_deep"})
    t = tabular_avg[
        (tabular_avg["model"] == tabular_model) & (tabular_avg["checkpoint_minutes"] == checkpoint)
    ][["record_id", "y_true", "y_proba"]].rename(columns={"y_proba": "proba_tab"})
    merged = d.merge(t, on="record_id", suffixes=("", "_tab"))
    mismatched = merged[merged["y_true"] != merged["y_true_tab"]]
    if len(mismatched) > 0:
        raise ValueError(
            f"y_true mismatch on {len(mismatched)} record(s) for "
            f"{deep_model} x {tabular_model} @ {checkpoint}min -- cohorts do not share record ids/labels."
        )
    if merged.empty:
        raise ValueError(f"No overlapping records for {deep_model} x {tabular_model} @ {checkpoint}min.")
    return merged


def sweep(deep_avg: pd.DataFrame, tabular_avg: pd.DataFrame) -> pd.DataFrame:
    available_deep = [m for m in DEEP_MODELS if m in set(deep_avg["model"].unique())]
    available_tabular = [m for m in TABULAR_MODELS if m in set(tabular_avg["model"].unique())]
    if not available_deep:
        raise ValueError(f"No deep models {DEEP_MODELS} found in the deep cohort.")
    if not available_tabular:
        raise ValueError(f"No tabular models {TABULAR_MODELS} found in the tabular cohort.")

    rows: List[Dict] = []
    for deep_model in available_deep:
        for tabular_model in available_tabular:
            for ck in CHECKPOINTS:
                merged = paired_frame(deep_avg, tabular_avg, deep_model, tabular_model, ck)
                y_true = merged["y_true"].astype(int).to_numpy()
                p_deep = merged["proba_deep"].astype(float).to_numpy()
                p_tab = merged["proba_tab"].astype(float).to_numpy()
                for method in METHODS:
                    for w in WEIGHTS:
                        blended = blend(p_deep, p_tab, w, method)
                        m = binary_metrics(y_true, blended, threshold=THRESHOLD)
                        rows.append({
                            "deep_model": deep_model,
                            "tabular_model": tabular_model,
                            "checkpoint_minutes": ck,
                            "method": method,
                            "weight_deep": w,
                            "n": len(y_true),
                            "n_positive": int(y_true.sum()),
                            "roc_auc": m["roc_auc"],
                            "pr_auc": m["pr_auc"],
                            "recall_sensitivity": m["recall_sensitivity"],
                            "specificity": m["specificity"],
                            "f1": m["f1"],
                        })
    return pd.DataFrame(rows)


def verify_against_existing(sweep_df: pd.DataFrame, deep_dir: Path) -> None:
    """The w=0.5 probability-averaged rows ARE the configuration
    run_ensemble_analysis.py already published. If this sweep is implemented
    correctly they must match that file exactly. Checking this before trusting
    any NEW combination is the whole point -- a sweep that cannot reproduce the
    known case is not evidence of anything."""
    existing_path = deep_dir / "ensemble_bootstrap_confidence_intervals.csv"
    if not existing_path.exists():
        print(f"  [skip] {existing_path.name} not present -- cannot cross-check.")
        return

    existing = pd.read_csv(existing_path)
    existing = existing[existing["metric"] == "roc_auc"]
    baseline = sweep_df[
        (sweep_df["weight_deep"] == BASELINE_WEIGHT)
        & (sweep_df["method"] == BASELINE_METHOD)
        & (sweep_df["tabular_model"] == "logistic_regression")
    ]

    checked = 0
    mismatches = []
    for _, row in baseline.iterrows():
        model_name = f"ensemble_{row['deep_model']}_lr"
        match = existing[
            (existing["model"] == model_name)
            & (existing["checkpoint_minutes"] == row["checkpoint_minutes"])
        ]
        if match.empty:
            continue
        published = float(match.iloc[0]["point_estimate"])
        if abs(published - row["roc_auc"]) > 1e-9:
            mismatches.append(
                f"{model_name}@{row['checkpoint_minutes']:.0f}min: "
                f"sweep={row['roc_auc']:.10f} vs published={published:.10f}"
            )
        checked += 1

    if checked == 0:
        print("  [skip] no overlapping rows to cross-check.")
    elif mismatches:
        raise ValueError(
            "Sweep does NOT reproduce the published 0.5/0.5 ensemble -- the blend code is wrong:\n  "
            + "\n  ".join(mismatches)
        )
    else:
        print(f"  [OK] reproduced all {checked} published 0.5/0.5 ROC-AUC values exactly (< 1e-9).")


def best_by(sweep_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Best combination per checkpoint, excluding the degenerate single-model
    endpoints (w=0 and w=1), which are not ensembles at all."""
    blends_only = sweep_df[(sweep_df["weight_deep"] > 0.0) & (sweep_df["weight_deep"] < 1.0)]
    idx = blends_only.groupby("checkpoint_minutes")[metric].idxmax()
    return blends_only.loc[idx].sort_values("checkpoint_minutes").reset_index(drop=True)


def baseline_rows(sweep_df: pd.DataFrame, deep_model: str = "ctg_only") -> pd.DataFrame:
    return sweep_df[
        (sweep_df["weight_deep"] == BASELINE_WEIGHT)
        & (sweep_df["method"] == BASELINE_METHOD)
        & (sweep_df["tabular_model"] == "logistic_regression")
        & (sweep_df["deep_model"] == deep_model)
    ].sort_values("checkpoint_minutes")


def bootstrap_best(
    best_df: pd.DataFrame,
    deep_avg: pd.DataFrame,
    tabular_avg: pd.DataFrame,
) -> pd.DataFrame:
    """Bootstrap CIs for the selected winners only. Bootstrapping all ~1500
    combinations would cost far more time than it is worth, and the CIs for
    non-selected combinations are never reported anyway."""
    frames = []
    for _, row in best_df.iterrows():
        merged = paired_frame(
            deep_avg, tabular_avg, row["deep_model"], row["tabular_model"], row["checkpoint_minutes"]
        )
        blended = blend(
            merged["proba_deep"].astype(float).to_numpy(),
            merged["proba_tab"].astype(float).to_numpy(),
            row["weight_deep"],
            row["method"],
        )
        label = f"best_{row['deep_model']}_x_{row['tabular_model']}_{row['method']}_w{row['weight_deep']:.2f}"
        frames.append(pd.DataFrame({
            "cohort_name": label,
            "checkpoint_minutes": row["checkpoint_minutes"],
            "model": label,
            "record_id": merged["record_id"].to_numpy(),
            "y_true": merged["y_true"].astype(int).to_numpy(),
            "y_proba": blended,
        }))
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, ignore_index=True)
    return bootstrap_metric_ci_anytime(stacked, THRESHOLD, BOOTSTRAP_ITERATIONS, SEED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ensemble blend weights, pairings, and combination methods.")
    parser.add_argument("--cohort-dir", required=True, help="Cohort providing the deep model OOF.")
    parser.add_argument(
        "--tabular-cohort-dir",
        default=None,
        help="Cohort providing logistic_regression/random_forest OOF (default: same as --cohort-dir).",
    )
    args = parser.parse_args()

    deep_dir = Path(args.cohort_dir)
    tabular_dir = Path(args.tabular_cohort_dir) if args.tabular_cohort_dir else deep_dir

    deep_avg = load_averaged(deep_dir)
    tabular_avg = load_averaged(tabular_dir)

    print(f"Deep models from:    {deep_dir}")
    print(f"Tabular models from: {tabular_dir}")
    print(f"Sweeping {len(WEIGHTS)} weights x {len(METHODS)} methods x deep x tabular x {len(CHECKPOINTS)} checkpoints...")

    sweep_df = sweep(deep_avg, tabular_avg)
    print(f"Evaluated {len(sweep_df)} (combination, checkpoint) pairs.\n")

    print("Verification -- does the sweep reproduce the already-published 0.5/0.5 ensemble?")
    verify_against_existing(sweep_df, deep_dir)

    prevalence = sweep_df.iloc[0]["n_positive"] / sweep_df.iloc[0]["n"]

    best_pr = best_by(sweep_df, "pr_auc")
    best_roc = best_by(sweep_df, "roc_auc")
    base = baseline_rows(sweep_df)

    print(f"\n=== Best blend per checkpoint by PR-AUC (no-skill floor = {prevalence:.4f}) ===")
    print(f"{'ckpt':>6} {'deep':>13} {'tabular':>20} {'meth':>5} {'w':>5} {'PR-AUC':>8} {'vs 0.5 base':>12}")
    for _, row in best_pr.iterrows():
        b = base[base["checkpoint_minutes"] == row["checkpoint_minutes"]]
        base_pr = float(b.iloc[0]["pr_auc"]) if not b.empty else float("nan")
        print(
            f"{row['checkpoint_minutes']:6.0f} {row['deep_model']:>13} {row['tabular_model']:>20} "
            f"{row['method']:>5} {row['weight_deep']:5.2f} {row['pr_auc']:8.4f} "
            f"{row['pr_auc'] - base_pr:+12.4f}"
        )

    print("\n=== Best blend per checkpoint by ROC-AUC ===")
    print(f"{'ckpt':>6} {'deep':>13} {'tabular':>20} {'meth':>5} {'w':>5} {'ROC-AUC':>8} {'vs 0.5 base':>12}")
    for _, row in best_roc.iterrows():
        b = base[base["checkpoint_minutes"] == row["checkpoint_minutes"]]
        base_roc = float(b.iloc[0]["roc_auc"]) if not b.empty else float("nan")
        print(
            f"{row['checkpoint_minutes']:6.0f} {row['deep_model']:>13} {row['tabular_model']:>20} "
            f"{row['method']:>5} {row['weight_deep']:5.2f} {row['roc_auc']:8.4f} "
            f"{row['roc_auc'] - base_roc:+12.4f}"
        )

    print("\nBootstrapping CIs for the selected winners only...")
    ci_pr = bootstrap_best(best_pr, deep_avg, tabular_avg)
    ci_roc = bootstrap_best(best_roc, deep_avg, tabular_avg)

    sweep_path = deep_dir / "ensemble_sweep_full.csv"
    best_path = deep_dir / "ensemble_sweep_best.csv"
    ci_path = deep_dir / "ensemble_sweep_best_bootstrap_ci.csv"

    sweep_df.to_csv(sweep_path, index=False)
    pd.concat(
        [best_pr.assign(selected_by="pr_auc"), best_roc.assign(selected_by="roc_auc")],
        ignore_index=True,
    ).to_csv(best_path, index=False)
    pd.concat(
        [ci_pr.assign(selected_by="pr_auc"), ci_roc.assign(selected_by="roc_auc")],
        ignore_index=True,
    ).to_csv(ci_path, index=False)

    print(f"\nWrote {sweep_path}")
    print(f"Wrote {best_path}")
    print(f"Wrote {ci_path}")

    mean_gain_pr = float(
        best_pr.set_index("checkpoint_minutes")["pr_auc"].sub(
            base.set_index("checkpoint_minutes")["pr_auc"]
        ).mean()
    )
    print(
        "\nSELECTION-BIAS NOTE: the winners above were chosen on the same OOF predictions they are\n"
        f"scored on, across {len(sweep_df) // len(CHECKPOINTS)} combinations per checkpoint. Mean PR-AUC gain over the\n"
        f"pre-specified 0.5/0.5 blend is {mean_gain_pr:+.4f}. Report the selected result WITH this caveat,\n"
        "not as a clean held-out improvement -- see Limitations."
    )


if __name__ == "__main__":
    main()
