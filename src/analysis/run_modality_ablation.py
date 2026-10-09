"""Modality ablation for the tabular models: does adding the 8 maternal
variables to CTG summary features actually improve prediction?

WHY THIS EXISTS
---------------
Every tabular model reported in this project (`logistic_regression`,
`random_forest`) is trained on `make_tabular_features()`, which concatenates 10
CTG summary statistics with the 8 maternal variables. Those models are therefore
multimodal by construction -- and they are the project's best performers (LR
0.757; deep+LR ensemble 0.777).

But the project never tested the counterfactual: LR/RF on the CTG features
ALONE. So the claim "combining CTG and maternal data helps" rested on indirect
evidence (parity ranking #1 of 18 features by |coef|) rather than a direct
comparison. This script supplies the direct comparison, which is the central
claim of the thesis title.

DESIGN
------
Three feature sets, everything else held identical:
  ctg_only_features      10 CTG summary stats
  maternal_only_features  8 maternal variables
  both                   all 18  (== the reported configuration)

The same `precompute_quality_splits()` split objects, the same seed, the same
`train_sklearn_quality_fold()` training path, the same checkpoint cropping, and
the same `aggregate_repeated_oof_anytime()` pooling as every other result in the
project -- so the numbers are directly comparable to the published ones, and the
`both` arm must reproduce them.

VERIFICATION BUILT IN
---------------------
The `both` arm is scored against the published cv_summary.json values before any
ablation conclusion is drawn. If `both` does not reproduce, the harness is wrong
and no comparison here means anything.

Paired DeLong + permutation tests compare `both` vs `ctg_only_features` on
identical records. Given 23 positives, expect wide CIs; report accordingly.

Usage:
  python -m src.analysis.run_modality_ablation
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.analysis.significance import delong_test, paired_permutation_test
from src.training.train_cv import (
    aggregate_repeated_oof_anytime,
    bootstrap_metric_ci_anytime,
    crop_ctg_array,
    ctg_summary_features,
    load_cohort_arrays,
    precompute_quality_splits,
    train_sklearn_quality_fold,
)
from src.utils.metrics import binary_metrics

CONFIG_PATH = Path("configs/anytime_cohort.yaml")
COHORT_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060")
OUT_DIR = COHORT_DIR
PROCESSED_DIR = Path("data/processed")
MODELS = ["logistic_regression", "random_forest"]
FEATURE_SETS = ["ctg_only_features", "maternal_only_features", "both"]
THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 1000
PERMUTATION_ITERATIONS = 2000


def build_features(feature_set: str, cropped_ctg: np.ndarray, X_maternal: np.ndarray) -> np.ndarray:
    """Mirrors make_tabular_features()'s block order exactly; `both` is the
    identical array that function returns with include_clinical=False."""
    if feature_set == "ctg_only_features":
        return ctg_summary_features(cropped_ctg)
    if feature_set == "maternal_only_features":
        return X_maternal.astype(np.float32)
    if feature_set == "both":
        return np.concatenate([ctg_summary_features(cropped_ctg), X_maternal], axis=1)
    raise ValueError(f"Unknown feature set: {feature_set}")


def main() -> None:
    exp_cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(exp_cfg["base_config"]).read_text(encoding="utf-8"))
    cohort = exp_cfg["cohorts"][0]
    suffix = str(cohort["suffix"])
    cv = exp_cfg["cross_validation"]
    n_splits, n_repeats, seed = int(cv["n_splits"]), int(cv["n_repeats"]), int(cv["seed"])
    val_size = float(base_cfg["training"]["val_size"])
    sample_rate = int(base_cfg["signals"]["target_sample_rate_hz"])
    checkpoints = [float(m) for m in exp_cfg["checkpoints_minutes"]]

    X_ctg, X_maternal, y, record_ids, _ = load_cohort_arrays(PROCESSED_DIR, suffix)
    print(f"Cohort {cohort['name']}: n={len(y)}, positives={int(y.sum())}")
    print(f"Feature-set sizes -- CTG-only: {ctg_summary_features(X_ctg[:1]).shape[1]}, "
          f"maternal-only: {X_maternal.shape[1]}, both: {ctg_summary_features(X_ctg[:1]).shape[1] + X_maternal.shape[1]}\n")

    splits = precompute_quality_splits(y, n_splits, n_repeats, seed, val_size)

    # train_sklearn_quality_fold persists a .joblib per call; we only need the
    # predictions, so point it at one scratch dir that each call overwrites.
    scratch = Path(tempfile.mkdtemp(prefix="modality_ablation_"))

    # Crop once per checkpoint rather than once per (feature set, model, fold):
    # crop_ctg_array copies the whole [345, 5400, 2] array, and recomputing it
    # 540 times instead of 6 is pure waste.
    cropped_by_ckpt = {ck: crop_ctg_array(X_ctg, ck, sample_rate) for ck in checkpoints}

    oof_rows: List[Dict] = []
    try:
        for feature_set in FEATURE_SETS:
            # Feature matrices depend only on (feature_set, checkpoint) -- build
            # them once and reuse across models and folds.
            features_by_ckpt = {
                ck: build_features(feature_set, cropped_by_ckpt[ck], X_maternal)
                for ck in checkpoints
            }
            for model_name in MODELS:
                print(f"  {feature_set:24s} | {model_name}")
                for split in splits:
                    outer_test_idx = np.asarray(split["outer_test_idx"], dtype=int)
                    for ck in checkpoints:
                        _, _, test_proba, _ = train_sklearn_quality_fold(
                            model_name, features_by_ckpt[ck], y, split, seed, scratch,
                        )
                        for local_i, row_idx in enumerate(outer_test_idx):
                            oof_rows.append({
                                "cohort_name": feature_set,
                                "checkpoint_minutes": ck,
                                "repeat": int(split["repeat"]),
                                "fold": int(split["fold"]),
                                "record_id": record_ids.iloc[row_idx]["record_id"],
                                "y_true": int(y[row_idx]),
                                "y_proba": float(test_proba[local_i]),
                                "model": model_name,
                            })
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    oof = pd.DataFrame(oof_rows)
    averaged = aggregate_repeated_oof_anytime(oof)

    # ---- verification: the `both` arm must reproduce the published numbers ----
    print("\n" + "=" * 78)
    print("VERIFICATION -- does the `both` arm reproduce the published results?")
    print("=" * 78)
    published = json.loads((COHORT_DIR / "cv_summary.json").read_text(encoding="utf-8"))
    pub_metrics = published["model_metrics_from_averaged_oof"]
    checked = mismatches = 0
    for model_name in MODELS:
        for ck in checkpoints:
            grp = averaged[
                (averaged["cohort_name"] == "both")
                & (averaged["model"] == model_name)
                & (averaged["checkpoint_minutes"] == ck)
            ]
            got = binary_metrics(grp["y_true"].to_numpy(), grp["y_proba"].to_numpy(), THRESHOLD)["roc_auc"]
            want = pub_metrics[f"{model_name}@{ck}min"]["roc_auc"]
            checked += 1
            if abs(got - want) > 1e-9:
                mismatches += 1
                print(f"  MISMATCH {model_name}@{ck:g}: got {got:.10f}, published {want:.10f}")
    if mismatches:
        raise SystemExit(
            f"\nABORT: {mismatches}/{checked} published values not reproduced. "
            "The ablation harness does not match the reported pipeline, so no "
            "conclusion below would be trustworthy."
        )
    print(f"  [OK] all {checked} published ROC-AUC values reproduced exactly (<1e-9).")

    # ---- results table ----
    rows = []
    for feature_set in FEATURE_SETS:
        for model_name in MODELS:
            for ck in checkpoints:
                grp = averaged[
                    (averaged["cohort_name"] == feature_set)
                    & (averaged["model"] == model_name)
                    & (averaged["checkpoint_minutes"] == ck)
                ]
                m = binary_metrics(grp["y_true"].to_numpy(), grp["y_proba"].to_numpy(), THRESHOLD)
                rows.append({
                    "feature_set": feature_set, "model": model_name, "checkpoint_minutes": ck,
                    "n": len(grp), "n_positive": int(grp["y_true"].sum()),
                    "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                })
    results = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("ABLATION -- ROC-AUC by feature set")
    print("=" * 78)
    for model_name in MODELS:
        print(f"\n{model_name}")
        print(f"{'ckpt':>6} {'CTG only':>10} {'maternal':>10} {'both':>10} {'both-CTG':>10}")
        for ck in checkpoints:
            r = {fs: results[(results.feature_set == fs) & (results.model == model_name)
                             & (results.checkpoint_minutes == ck)]["roc_auc"].iloc[0]
                 for fs in FEATURE_SETS}
            print(f"{ck:6.0f} {r['ctg_only_features']:10.4f} {r['maternal_only_features']:10.4f} "
                  f"{r['both']:10.4f} {r['both'] - r['ctg_only_features']:+10.4f}")

    # ---- paired significance: both vs CTG-only ----
    print("\n" + "=" * 78)
    print("PAIRED TESTS -- both vs CTG-features-only (same records)")
    print("=" * 78)
    sig_rows = []
    for model_name in MODELS:
        for ck in checkpoints:
            a = averaged[(averaged.cohort_name == "both") & (averaged.model == model_name)
                         & (averaged.checkpoint_minutes == ck)].sort_values("record_id")
            b = averaged[(averaged.cohort_name == "ctg_only_features") & (averaged.model == model_name)
                         & (averaged.checkpoint_minutes == ck)].sort_values("record_id")
            assert (a["record_id"].to_numpy() == b["record_id"].to_numpy()).all(), "record mismatch"
            y_true = a["y_true"].to_numpy()
            p_both, p_ctg = a["y_proba"].to_numpy(), b["y_proba"].to_numpy()
            d = delong_test(y_true, p_both, p_ctg)
            perm = paired_permutation_test(
                y_true, p_both, p_ctg, n_permutations=PERMUTATION_ITERATIONS, seed=seed,
            )
            sig_rows.append({
                "model": model_name, "checkpoint_minutes": ck,
                "auc_both": d["auc_a"], "auc_ctg_only": d["auc_b"],
                "delta": d["auc_a"] - d["auc_b"],
                "delong_p": d["p_value"], "permutation_p": perm["p_value"],
            })
    sig = pd.DataFrame(sig_rows)
    print(sig.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ci = bootstrap_metric_ci_anytime(averaged, THRESHOLD, BOOTSTRAP_ITERATIONS, seed)

    results.to_csv(OUT_DIR / "modality_ablation_results.csv", index=False)
    sig.to_csv(OUT_DIR / "modality_ablation_significance.csv", index=False)
    ci.to_csv(OUT_DIR / "modality_ablation_bootstrap_ci.csv", index=False)
    averaged.to_csv(OUT_DIR / "modality_ablation_oof.csv", index=False)

    print(f"\nWrote modality_ablation_{{results,significance,bootstrap_ci,oof}}.csv to {OUT_DIR}")

    # ---- plain-language verdict ----
    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    for model_name in MODELS:
        s = sig[sig.model == model_name]
        helped = int((s["delta"] > 0).sum())
        mean_delta = float(s["delta"].mean())
        best_p = float(s[["delong_p", "permutation_p"]].min().min())
        print(f"\n{model_name}: adding maternal features helped at {helped}/6 checkpoints, "
              f"mean delta {mean_delta:+.4f}, smallest p = {best_p:.3f}")
        if helped >= 5 and mean_delta > 0:
            print("  -> supports the multimodal claim in DIRECTION (significance separate).")
        elif helped <= 1:
            print("  -> does NOT support the multimodal claim for this model.")
        else:
            print("  -> MIXED. Do not report as support for the multimodal claim.")
    print("\nNote: with 23 positives, expect no p < 0.05. Absence of significance here")
    print("is the project-wide power ceiling, not specific evidence against fusion.")


if __name__ == "__main__":
    main()
