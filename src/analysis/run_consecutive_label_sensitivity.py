"""7b: consecutive-label sensitivity analysis (HANDOFF.md section 7b).

The project's primary label is GAPPED: positive pH < 7.05, negative pH >= 7.15, and the
65 records in the 7.05-7.15 band are discarded. That design inflates discrimination
relative to a real cohort, in which every delivered patient has to be classified. This
script rebuilds the analysis with a CONSECUTIVE label: positive pH < 7.05, negative
everything else INCLUDING the 7.05-7.15 band.

Cohort build (canonical pipeline, not reimplemented here):
  src/data/build_labels.py regenerated record_meta_labels_acidemia.csv from the raw .hea
  files and reproduced the documented gapped flow exactly (552 source, 487 labelled, 40
  positives). A column y_acidemia_705_consec was then ADDED to that file (additive; the
  original was backed up alongside it), and src/data/build_windows.py was run unchanged:
    --label y_acidemia_705_consec --window-minutes 90 --window-position first
    --min-valid-ratio 0.60 --cohort-run --cohort-name primary_705_consec_vr060
    --suffix _705_consec_vr060
  so the quality filter, window extraction and interpolation are bit-identical to the
  gapped cohort's.

Both cohorts are run through the SAME code path here, so the side-by-side table compares
like with like. The gapped arm doubles as a self-verification: its LR/RF values must
reproduce the published featfix_results.csv before any consecutive number is trusted.

Conventions: fixed CTG features (src/data/fixed_ctg_features.py), per-fold imputation
inside the sklearn Pipeline for the CV arms, whole-cohort imputation for the single-fit
Firth LRTs, folds regenerated per cohort with seed 42 (n differs, so the fold objects
necessarily differ), checkpoints 10-60 only, risk-set scoring, Holm across 5 checkpoints.

Outputs: outputs/quality_cohorts/primary_705_consec_vr060/
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.firth import firth_lrt  # noqa: E402
from src.data.fixed_ctg_features import ctg_summary_features_v2  # noqa: E402
from src.training.train_cv import (  # noqa: E402
    aggregate_repeated_oof_anytime,
    load_cohort_arrays,
    precompute_quality_splits,
    train_sklearn_quality_fold,
)

CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]
MATERNAL_NAMES = ["age", "gravidity", "parity", "gest_weeks", "diabetes", "hypertension", "preeclampsia", "induced"]
MODELS = ["logistic_regression", "random_forest"]
FEATURE_SETS = ["ctg_only_features", "both", "ctg_plus_parity", "maternal_only_features"]
SEED = 42
BOOT_ITERS = 1000  # project convention for metric CIs (CLAUDE.md Validation Standard)
SOURCE_HZ = 4
DEC = 4

COHORTS = {
    "gapped": {"suffix": "_705_anytime_vr060", "desc": "PRIMARY: pos pH<7.05, neg pH>=7.15, 7.05-7.15 band discarded"},
    "consecutive": {"suffix": "_705_consec_vr060", "desc": "SENSITIVITY: pos pH<7.05, neg everything else incl. 7.05-7.15"},
}
OUT_DIR = Path("outputs/quality_cohorts/primary_705_consec_vr060")
FEATFIX_RESULTS = Path("outputs/quality_cohorts/primary_705_anytime_vr060_featfix/featfix_results.csv")


def holm(pvals):
    from statsmodels.stats.multitest import multipletests
    return multipletests(pvals, method="holm")[1].tolist()


def durations_from_record_ids(suffix: str) -> np.ndarray:
    rid = pd.read_csv(Path("data/processed") / f"record_ids{suffix}.csv")
    return (rid["recording_length_source_samples"] / (SOURCE_HZ * 60.0)).to_numpy()


def build_feature_set(name, X_ctg, X_maternal, ck):
    """Raw (un-imputed) features; each fold's sklearn Pipeline imputes on its own
    inner-train split. Matches run_featfix_cohort.py exactly."""
    ctg = ctg_summary_features_v2(X_ctg, checkpoint_minutes=ck)
    parity = X_maternal[:, MATERNAL_NAMES.index("parity")]
    if name == "ctg_only_features":
        return ctg
    if name == "both":
        return np.concatenate([ctg, X_maternal], axis=1)
    if name == "ctg_plus_parity":
        return np.column_stack([ctg, parity])
    if name == "maternal_only_features":
        return X_maternal.astype(np.float32)
    raise ValueError(name)


def bootstrap_ci(y_true, y_proba, seed=SEED, iters=BOOT_ITERS):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rocs, prs = [], []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        rocs.append(roc_auc_score(y_true[idx], y_proba[idx]))
        prs.append(average_precision_score(y_true[idx], y_proba[idx]))
    return (float(np.percentile(rocs, 2.5)), float(np.percentile(rocs, 97.5)),
            float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5)))


def run_cv_grid(cohort_key, X_ctg, X_maternal, y, record_ids, splits, dur):
    scratch = Path(tempfile.mkdtemp(prefix=f"consec_{cohort_key}_"))
    oof_rows = []
    try:
        for ck in CHECKPOINTS:
            for fs in FEATURE_SETS:
                X_fs = build_feature_set(fs, X_ctg, X_maternal, ck)
                for model in MODELS:
                    for sp in splits:
                        _, _, tp, _ = train_sklearn_quality_fold(model, X_fs, y, sp, SEED, scratch)
                        for j, ix in enumerate(np.asarray(sp["outer_test_idx"], dtype=int)):
                            oof_rows.append({
                                "checkpoint_minutes": ck, "cohort_name": fs, "model": model,
                                "record_id": str(record_ids.iloc[ix]["record_id"]), "y_true": int(y[ix]),
                                "y_proba": float(tp[j]),
                            })
                print(f"  [{cohort_key}] ck={ck:.0f} {fs} done")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    avg = aggregate_repeated_oof_anytime(pd.DataFrame(oof_rows))
    rid_str = record_ids["record_id"].astype(str).to_numpy()
    dur_by_rid = dict(zip(rid_str, dur))
    avg["duration"] = avg["record_id"].map(dur_by_rid)

    rows = []
    for ck in CHECKPOINTS:
        for fs in FEATURE_SETS:
            for model in MODELS:
                g = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == fs) & (avg.model == model)]
                rs = g[g["duration"] >= ck]
                yt = rs["y_true"].to_numpy()
                yp = rs["y_proba"].to_numpy()
                roc = roc_auc_score(yt, yp)
                pr = average_precision_score(yt, yp)
                rl, rh, pl, ph = bootstrap_ci(yt, yp)
                rows.append({
                    "label_design": cohort_key, "checkpoint_minutes": ck, "feature_set": fs, "model": model,
                    "n_risk_set": len(rs), "positives_risk_set": int(yt.sum()),
                    "prevalence": round(float(yt.mean()), DEC),
                    "roc_auc": round(float(roc), DEC), "roc_ci_low": round(rl, DEC), "roc_ci_high": round(rh, DEC),
                    "pr_auc": round(float(pr), DEC), "pr_ci_low": round(pl, DEC), "pr_ci_high": round(ph, DEC),
                })
    return pd.DataFrame(rows), avg


def run_firth(cohort_key, X_ctg, X_maternal, y):
    """Whole-cohort median imputation (single global fit, as in Steps 1-3)."""
    Xmat_imp = SimpleImputer(strategy="median").fit_transform(X_maternal)
    parity = Xmat_imp[:, MATERNAL_NAMES.index("parity")]
    rows = []
    for ck in CHECKPOINTS:
        ctg = SimpleImputer(strategy="median").fit_transform(
            ctg_summary_features_v2(X_ctg, checkpoint_minutes=ck))
        n = len(y)
        dropped = [nm for j, nm in enumerate(
            ["fhr_mean", "uc_mean", "fhr_std", "uc_std", "fhr_min", "uc_min", "fhr_max", "uc_max",
             "fhr_median", "uc_median"]) if len(np.unique(ctg[:, j])) <= 1]

        X1 = np.column_stack([np.ones(n), ctg, Xmat_imp])
        mat_idx = list(range(1 + ctg.shape[1], 1 + ctg.shape[1] + 8))
        lrt_mat = firth_lrt(X1, y, test_idx=mat_idx)

        X2 = np.column_stack([np.ones(n), ctg, parity])
        lrt_par = firth_lrt(X2, y, test_idx=[X2.shape[1] - 1])
        beta = float(lrt_par.full_fit.beta[X2.shape[1] - 1])
        se = float(lrt_par.full_fit.se[X2.shape[1] - 1])

        rows.append({
            "label_design": cohort_key, "checkpoint_minutes": ck,
            "ctg_cols_dropped": ";".join(dropped) if dropped else "(none)",
            "maternal_lr_stat": round(lrt_mat.lr_stat, DEC), "maternal_df": lrt_mat.df,
            "maternal_raw_p": lrt_mat.p_value,
            "parity_lr_stat": round(lrt_par.lr_stat, DEC), "parity_df": lrt_par.df,
            "parity_raw_p": lrt_par.p_value,
            "parity_or_per_additional_birth": round(float(np.exp(beta)), DEC),
            "parity_or_95ci_low": round(float(np.exp(beta - 1.96 * se)), DEC),
            "parity_or_95ci_high": round(float(np.exp(beta + 1.96 * se)), DEC),
            "all_converged": bool(lrt_mat.full_fit.converged and lrt_mat.restricted_fit.converged
                                  and lrt_par.full_fit.converged and lrt_par.restricted_fit.converged),
        })
    df = pd.DataFrame(rows)
    df["maternal_holm_p"] = holm(df["maternal_raw_p"].tolist())
    df["parity_holm_p"] = holm(df["parity_raw_p"].tolist())
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    flow_rows, cv_frames, firth_frames = [], [], []
    for key, meta in COHORTS.items():
        suffix = meta["suffix"]
        X_ctg, X_maternal, y, record_ids, manifest = load_cohort_arrays(Path("data/processed"), suffix)
        dur = durations_from_record_ids(suffix)
        splits = precompute_quality_splits(y, 5, 3, SEED, 0.2)

        inc = manifest[manifest.inclusion_status == "included"]
        exc = manifest[manifest.inclusion_status == "excluded"]
        flow_rows.append({
            "label_design": key, "description": meta["desc"], "suffix": suffix,
            "n_source_records": len(manifest), "n_labelled": len(manifest),
            "n_included": len(inc), "positives_included": int(y.sum()),
            "n_excluded_low_valid_ratio": len(exc), "positives_excluded": int(exc["label"].sum()),
            "prevalence_included": round(float(y.mean()), DEC),
            "duration_min_min": round(float(dur.min()), 2), "duration_min_median": round(float(np.median(dur)), 2),
            "duration_min_max": round(float(dur.max()), 2),
        })

        print(f"\n=== CV grid: {key} (n={len(y)}, pos={int(y.sum())}) ===")
        cv_df, _ = run_cv_grid(key, X_ctg, X_maternal, y, record_ids, splits, dur)
        cv_frames.append(cv_df)
        firth_frames.append(run_firth(key, X_ctg, X_maternal, y))

    flow = pd.DataFrame(flow_rows)
    cv_all = pd.concat(cv_frames, ignore_index=True)
    firth_all = pd.concat(firth_frames, ignore_index=True)

    # ---- self-verification: gapped arm must reproduce published featfix_results.csv ----
    pub = pd.read_csv(FEATFIX_RESULTS)
    ver_rows = []
    for _, r in cv_all[(cv_all.label_design == "gapped")
                       & (cv_all.feature_set.isin(["ctg_only_features", "both", "maternal_only_features"]))].iterrows():
        p = pub[(pub.checkpoint_minutes == r.checkpoint_minutes) & (pub.feature_set == r.feature_set)
                & (pub.model == r.model)]
        pub_roc = float(p.iloc[0]["roc_auc_risk_set"]) if len(p) else np.nan
        ver_rows.append({"checkpoint_minutes": r.checkpoint_minutes, "feature_set": r.feature_set,
                        "model": r.model, "mine": r.roc_auc, "published": pub_roc,
                        "match_4dp": bool(round(r.roc_auc, 4) == round(pub_roc, 4))})
    ver = pd.DataFrame(ver_rows)
    n_match = int(ver["match_4dp"].sum())
    print(f"\nSelf-verification (gapped vs published featfix): {n_match}/{len(ver)} match at 4dp")
    if n_match != len(ver):
        print(ver[~ver.match_4dp].to_string(index=False))
        raise SystemExit("ABORT: gapped arm does not reproduce published results; consecutive numbers not trusted.")

    # ---- side-by-side ----
    side = cv_all.pivot_table(index=["checkpoint_minutes", "feature_set", "model"],
                            columns="label_design", values=["roc_auc", "pr_auc"]).reset_index()
    side.columns = ["_".join([c for c in col if c]).strip() for col in side.columns.values]
    side["roc_delta_consec_minus_gapped"] = (side["roc_auc_consecutive"] - side["roc_auc_gapped"]).round(DEC)
    side["pr_delta_consec_minus_gapped"] = (side["pr_auc_consecutive"] - side["pr_auc_gapped"]).round(DEC)

    flow.to_csv(OUT_DIR / "consec_cohort_flow.csv", index=False)
    cv_all.to_csv(OUT_DIR / "consec_cv_results.csv", index=False)
    firth_all.to_csv(OUT_DIR / "consec_firth_lrt.csv", index=False)
    side.to_csv(OUT_DIR / "consec_side_by_side.csv", index=False)
    ver.to_csv(OUT_DIR / "consec_self_verification.csv", index=False)

    print("\n=== COHORT FLOW ===")
    print(flow.T.to_string())
    print("\n=== FIRTH LRT ===")
    print(firth_all.to_string(index=False))
    print("\n=== SIDE BY SIDE (ROC) ===")
    print(side.to_string(index=False))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
