"""Phase 1b: extract feature importance from the already-trained LR / RF models
of the anytime cohort (primary_705_anytime_vr060) -- NO retraining.

The strongest model in the whole project is logistic_regression (ROC-AUC 0.757),
yet its coefficients have never been inspected. This script loads the saved
Pipelines and reports, per checkpoint:
  - LR: standardized-feature coefficients (comparable in magnitude because a
    StandardScaler precedes the model) -> signed coef + |coef| rank.
  - RF: Gini feature_importances_ -> importance + rank.

Falsifiable prediction (Finding 1's sanity check): the 10 CTG summary stats
dominate and the 4 binary maternal flags (diabetes/hypertension/preeclampsia/
induced) sit near the bottom. If a dead flag ranks high, Finding 1 is wrong and
the improvement plan must be revisited.

Scope limitation (same as the gate analysis): run_anytime_cohort keeps only the
single best-val-PR-AUC fold's model per (model, checkpoint), so this is one
fold's fitted model, not a repeat-averaged sweep. It is a strong indicator of
what the model leans on, not a 15-fold-averaged importance.

Feature order (verified against code):
  ctg_summary_features (train_cv.py:61) = concat[mean,std,min,max,median], each
  [N, 2] with channel 0 = FHR, channel 1 = UC (build_windows.py:181), stat-major
  channel-minor -> 10 features; then the 8 maternal features appended in
  maternal_feature_names order. Total 18.

Usage: python -m src.analysis.run_feature_importance
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
COHORT_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060")
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]

CTG_STAT_NAMES = ["mean", "std", "min", "max", "median"]
CTG_CHANNELS = ["fhr", "uc"]  # channel 0 = FHR, channel 1 = UC (build_windows.py:181)
DEAD_FLAGS = {"diabetes", "hypertension", "preeclampsia", "induced"}


def build_feature_names(maternal_names: list[str]) -> list[str]:
    ctg_names = [f"{ch}_{stat}" for stat in CTG_STAT_NAMES for ch in CTG_CHANNELS]
    return ctg_names + list(maternal_names)


def main() -> None:
    cv_summary = json.loads((COHORT_DIR / "cv_summary.json").read_text(encoding="utf-8"))
    suffix = cv_summary["suffix"]
    maternal_names = json.loads(
        (PROCESSED_DIR / f"maternal_feature_names{suffix}.json").read_text(encoding="utf-8")
    )
    feature_names = build_feature_names(maternal_names)
    n_expected = len(CTG_STAT_NAMES) * len(CTG_CHANNELS) + len(maternal_names)
    if len(feature_names) != n_expected:
        raise ValueError(f"feature-name count {len(feature_names)} != expected {n_expected}")

    rows = []
    for model_name, importance_kind in [("logistic_regression", "coef"), ("random_forest", "gini")]:
        for ckpt in CHECKPOINTS:
            model_path = COHORT_DIR / "models" / f"{model_name}__ckpt{ckpt:g}" / "best" / f"{model_name}.joblib"
            if not model_path.exists():
                print(f"WARNING missing {model_path}")
                continue
            pipeline = joblib.load(model_path)
            estimator = pipeline.named_steps["model"]

            if importance_kind == "coef":
                values = np.asarray(estimator.coef_).reshape(-1)
            else:
                values = np.asarray(estimator.feature_importances_).reshape(-1)
            if len(values) != len(feature_names):
                raise ValueError(
                    f"{model_name}@{ckpt}: {len(values)} importances vs {len(feature_names)} names"
                )

            abs_values = np.abs(values)
            order = np.argsort(-abs_values)
            rank_of = {int(idx): r + 1 for r, idx in enumerate(order)}
            for j, name in enumerate(feature_names):
                rows.append({
                    "model": model_name,
                    "checkpoint_minutes": ckpt,
                    "feature": name,
                    "is_maternal": name in maternal_names,
                    "is_dead_flag": name in DEAD_FLAGS,
                    "importance": round(float(values[j]), 6),
                    "abs_importance": round(float(abs_values[j]), 6),
                    "rank": rank_of[j],
                })

    df = pd.DataFrame(rows)
    out_path = COHORT_DIR / "feature_importance.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}\n")

    # Summary: top-5 features per model averaged over checkpoints, and where the
    # dead maternal flags land on average.
    for model_name in ["logistic_regression", "random_forest"]:
        sub = df[df["model"] == model_name]
        if sub.empty:
            continue
        avg_abs = (
            sub.groupby("feature", as_index=False)
            .agg(mean_abs=("abs_importance", "mean"), mean_rank=("rank", "mean"),
                 is_dead_flag=("is_dead_flag", "first"))
            .sort_values("mean_abs", ascending=False)
        )
        print(f"=== {model_name}: top 6 features (mean |importance| over checkpoints) ===")
        for _, r in avg_abs.head(6).iterrows():
            print(f"  {r['feature']:14s} mean|imp|={r['mean_abs']:.4f}  mean_rank={r['mean_rank']:.1f}")
        dead = avg_abs[avg_abs["is_dead_flag"]].sort_values("mean_rank")
        print(f"  -- dead maternal flags (of {len(feature_names)} features): "
              + ", ".join(f"{r['feature']}(rank {r['mean_rank']:.1f})" for _, r in dead.iterrows()))
        print()


if __name__ == "__main__":
    main()
