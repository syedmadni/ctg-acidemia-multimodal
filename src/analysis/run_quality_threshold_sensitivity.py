"""7c: quality-threshold (valid-ratio) sensitivity analysis.

The primary cohort filters on FHR valid ratio >= 0.60, which removed 142 records and
17 of 40 positives (42.5%). This script asks what the results look like at looser and
stricter thresholds.

Thresholds: 0.40, 0.50, 0.60 (primary), 0.70.
  0.50 / 0.60 / 0.70 are the project's own predefined set
  (configs/quality_cohorts.yaml: predefined_thresholds: [0.50, 0.60, 0.70]), so this is a
  pre-specified sensitivity family rather than a new invention. 0.40 is added on the
  user's instruction as an extra-loose bound, and is flagged as beyond the predefined set.
  (HANDOFF.md refers to a cohort_registry.yaml; no such file exists in the repo, so the
  configs/quality_cohorts.yaml list is used as the authoritative predefined set.)

Label: GAPPED (y_acidemia_705) throughout, as the primary design. Cohorts were built with
the canonical pipeline, unchanged:
  python -m src.data.build_windows --config configs/baseline.yaml --label y_acidemia_705
    --window-minutes 90 --window-position first --min-valid-ratio <r>
    --cohort-run --cohort-name primary_705_anytime_vr<vvv> --suffix _705_anytime_vr<vvv>

Conventions match Steps 1-3 and 7b: fixed CTG features, per-fold imputation inside the
sklearn Pipeline for CV arms, whole-cohort imputation for single-fit Firth LRTs, folds
regenerated per cohort with seed 42, checkpoints 10-60, risk-set scoring, Holm across 5.

Outputs: outputs/quality_cohorts/quality_threshold_sensitivity/
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.firth import firth_lrt  # noqa: E402
from src.data.fixed_ctg_features import FEATURE_NAMES, ctg_summary_features_v2  # noqa: E402
from src.training.train_cv import (  # noqa: E402
    aggregate_repeated_oof_anytime,
    load_cohort_arrays,
    precompute_quality_splits,
    train_sklearn_quality_fold,
)

CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0]
MATERNAL_NAMES = ["age", "gravidity", "parity", "gest_weeks", "diabetes", "hypertension", "preeclampsia", "induced"]
MODELS = ["logistic_regression", "random_forest"]
FEATURE_SETS = ["ctg_only_features", "maternal_only_features", "both", "ctg_plus_parity"]
SEED = 42
BOOT_ITERS = 1000
SOURCE_HZ = 4
DEC = 4
PRIMARY = "vr060"

THRESHOLDS = {
    "vr040": {"suffix": "_705_anytime_vr040", "ratio": 0.40, "predefined": False},
    "vr050": {"suffix": "_705_anytime_vr050", "ratio": 0.50, "predefined": True},
    "vr060": {"suffix": "_705_anytime_vr060", "ratio": 0.60, "predefined": True},
    "vr070": {"suffix": "_705_anytime_vr070", "ratio": 0.70, "predefined": True},
}
OUT_DIR = Path("outputs/quality_cohorts/quality_threshold_sensitivity")
META = Path("data/processed/record_meta_labels_acidemia.csv")


def holm(pvals):
    from statsmodels.stats.multitest import multipletests
    return multipletests(pvals, method="holm")[1].tolist()


def build_feature_set(name, X_ctg, X_maternal, ck):
    ctg = ctg_summary_features_v2(X_ctg, checkpoint_minutes=ck)
    if name == "ctg_only_features":
        return ctg
    if name == "maternal_only_features":
        return X_maternal.astype(np.float32)
    if name == "both":
        return np.concatenate([ctg, X_maternal], axis=1)
    if name == "ctg_plus_parity":
        return np.column_stack([ctg, X_maternal[:, MATERNAL_NAMES.index("parity")]])
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


def zero_signal_counts(X_ctg, ck):
    """Records with NO valid (non-zero) sample in [0, ck) for a channel. These become NaN
    in the fixed features and are median-imputed by the downstream pipeline."""
    n_samp = int(round(ck * 60 * 1))
    w = X_ctg[:, :n_samp, :]
    return int((~(w[:, :, 0] != 0).any(axis=1)).sum()), int((~(w[:, :, 1] != 0).any(axis=1)).sum())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META)
    meta["record_id"] = meta["record_id"].astype(int).astype(str)
    parity_by_rid = dict(zip(meta["record_id"], meta["parity"]))

    # reference: which records/positives the PRIMARY 0.60 cohort kept
    _, _, y60, rid60, _ = load_cohort_arrays(Path("data/processed"), THRESHOLDS[PRIMARY]["suffix"])
    primary_rids = set(rid60["record_id"].astype(str))
    primary_pos_rids = set(rid60["record_id"].astype(str)[y60 == 1])

    flow_rows, ret_rows, zero_rows, cv_frames, firth_frames, twobytwo_rows = [], [], [], [], [], []

    for key, cfg in THRESHOLDS.items():
        suffix = cfg["suffix"]
        X_ctg, X_maternal, y, record_ids, manifest = load_cohort_arrays(Path("data/processed"), suffix)
        rid_str = record_ids["record_id"].astype(str).to_numpy()
        dur = (record_ids["recording_length_source_samples"] / (SOURCE_HZ * 60.0)).to_numpy()
        splits = precompute_quality_splits(y, 5, 3, SEED, 0.2)
        exc = manifest[manifest.inclusion_status == "excluded"]

        # ---- returning positives (relative to the 0.60 primary) ----
        pos_rids = set(rid_str[y == 1])
        returning = sorted(pos_rids - primary_pos_rids)
        lost = sorted(primary_pos_rids - pos_rids)
        n_nulli = sum(1 for r in returning if parity_by_rid.get(r) == 0)
        n_parous = len(returning) - n_nulli
        for r in returning:
            ret_rows.append({"threshold": key, "record_id": r, "parity": parity_by_rid.get(r),
                            "parity_group": "nulliparous" if parity_by_rid.get(r) == 0 else "parous",
                            "direction": "returned vs vr060"})
        for r in lost:
            ret_rows.append({"threshold": key, "record_id": r, "parity": parity_by_rid.get(r),
                            "parity_group": "nulliparous" if parity_by_rid.get(r) == 0 else "parous",
                            "direction": "LOST vs vr060"})

        risk_full = all((dur >= ck).all() for ck in CHECKPOINTS)
        flow_rows.append({
            "threshold": key, "min_valid_ratio": cfg["ratio"], "predefined_in_config": cfg["predefined"],
            "is_primary": key == PRIMARY,
            "n_included": len(y), "positives": int(y.sum()), "prevalence": round(float(y.mean()), DEC),
            "n_excluded": len(exc), "positives_excluded": int(exc["label"].sum()),
            "positives_returning_vs_vr060": len(returning),
            "returning_nulliparous": n_nulli, "returning_parous": n_parous,
            "positives_lost_vs_vr060": len(lost),
            "duration_min": round(float(dur.min()), 2), "duration_median": round(float(np.median(dur)), 2),
            "duration_max": round(float(dur.max()), 2),
            "risk_set_full_at_10_60": risk_full,
            "epv_full_cohort_m1_18vars": round(float(y.sum()) / 18, DEC),
        })

        # ---- zero-signal counts per checkpoint ----
        for ck in CHECKPOINTS:
            zf, zu = zero_signal_counts(X_ctg, ck)
            zero_rows.append({"threshold": key, "checkpoint_minutes": ck, "n_records": len(y),
                            "records_zero_valid_fhr": zf, "records_zero_valid_uc": zu})

        # ---- 2x2 nulliparous vs parous ----
        parity = SimpleImputer(strategy="median").fit_transform(X_maternal)[:, MATERNAL_NAMES.index("parity")]
        nulli = (parity == 0).astype(int)
        a = int(((nulli == 1) & (y == 1)).sum()); b = int(((nulli == 1) & (y == 0)).sum())
        c = int(((nulli == 0) & (y == 1)).sum()); d = int(((nulli == 0) & (y == 0)).sum())
        cells = [a, b, c, d]
        ha = any(x == 0 for x in cells)
        aa, bb, cc, dd = (x + 0.5 for x in cells) if ha else cells
        or_ = (aa * dd) / (bb * cc)
        se = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
        _, fisher_p = stats.fisher_exact(np.array([[a, b], [c, d]]))
        twobytwo_rows.append({
            "threshold": key, "label": "POST HOC (parity selected after 2026-08-12 diagnostic)",
            "nulliparous_positive": a, "nulliparous_negative": b,
            "parous_positive": c, "parous_negative": d,
            "fisher_p": float(fisher_p),
            "or_nulliparous_vs_parous": round(float(or_), DEC),
            "or_95ci_low": round(float(np.exp(np.log(or_) - 1.96 * se)), DEC),
            "or_95ci_high": round(float(np.exp(np.log(or_) + 1.96 * se)), DEC),
            "haldane_anscombe_used": ha,
        })

        # ---- CV grid ----
        print(f"\n=== CV grid: {key} (n={len(y)}, pos={int(y.sum())}) ===")
        scratch = Path(tempfile.mkdtemp(prefix=f"vrsens_{key}_"))
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
                                    "record_id": rid_str[ix], "y_true": int(y[ix]), "y_proba": float(tp[j]),
                                })
                print(f"  ck={ck:.0f} done")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        avg = aggregate_repeated_oof_anytime(pd.DataFrame(oof_rows))
        avg["duration"] = avg["record_id"].map(dict(zip(rid_str, dur)))
        rows = []
        for ck in CHECKPOINTS:
            for fs in FEATURE_SETS:
                for model in MODELS:
                    g = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == fs) & (avg.model == model)]
                    rs = g[g["duration"] >= ck]
                    yt, yp = rs["y_true"].to_numpy(), rs["y_proba"].to_numpy()
                    rl, rh, pl, ph = bootstrap_ci(yt, yp)
                    rows.append({
                        "threshold": key, "checkpoint_minutes": ck, "feature_set": fs, "model": model,
                        "n_risk_set": len(rs), "positives_risk_set": int(yt.sum()),
                        "roc_auc": round(float(roc_auc_score(yt, yp)), DEC),
                        "roc_ci_low": round(rl, DEC), "roc_ci_high": round(rh, DEC),
                        "pr_auc": round(float(average_precision_score(yt, yp)), DEC),
                        "pr_ci_low": round(pl, DEC), "pr_ci_high": round(ph, DEC),
                    })
        cv_frames.append(pd.DataFrame(rows))

        # ---- Firth LRTs ----
        Xmat_imp = SimpleImputer(strategy="median").fit_transform(X_maternal)
        par = Xmat_imp[:, MATERNAL_NAMES.index("parity")]
        frows = []
        for ck in CHECKPOINTS:
            ctg = SimpleImputer(strategy="median").fit_transform(
                ctg_summary_features_v2(X_ctg, checkpoint_minutes=ck))
            dropped = [nm for j, nm in enumerate(FEATURE_NAMES) if len(np.unique(ctg[:, j])) <= 1]
            keep = [j for j, nm in enumerate(FEATURE_NAMES) if nm not in dropped]
            ctg = ctg[:, keep]
            n = len(y)
            X1 = np.column_stack([np.ones(n), ctg, Xmat_imp])
            lm = firth_lrt(X1, y, test_idx=list(range(1 + ctg.shape[1], 1 + ctg.shape[1] + 8)))
            X2 = np.column_stack([np.ones(n), ctg, par])
            lp = firth_lrt(X2, y, test_idx=[X2.shape[1] - 1])
            beta = float(lp.full_fit.beta[X2.shape[1] - 1]); sep = float(lp.full_fit.se[X2.shape[1] - 1])
            frows.append({
                "threshold": key, "checkpoint_minutes": ck,
                "ctg_cols_dropped": ";".join(dropped) if dropped else "(none)",
                "maternal_lr_stat": round(lm.lr_stat, DEC), "maternal_df": lm.df, "maternal_raw_p": lm.p_value,
                "parity_lr_stat": round(lp.lr_stat, DEC), "parity_df": lp.df, "parity_raw_p": lp.p_value,
                "parity_or_per_additional_birth": round(float(np.exp(beta)), DEC),
                "parity_or_95ci_low": round(float(np.exp(beta - 1.96 * sep)), DEC),
                "parity_or_95ci_high": round(float(np.exp(beta + 1.96 * sep)), DEC),
                "all_converged": bool(lm.full_fit.converged and lm.restricted_fit.converged
                                      and lp.full_fit.converged and lp.restricted_fit.converged),
                "warnings": ";".join(lm.full_fit.warnings + lm.restricted_fit.warnings
                                    + lp.full_fit.warnings + lp.restricted_fit.warnings) or "(none)",
            })
        fdf = pd.DataFrame(frows)
        fdf["maternal_holm_p"] = holm(fdf["maternal_raw_p"].tolist())
        fdf["parity_holm_p"] = holm(fdf["parity_raw_p"].tolist())
        firth_frames.append(fdf)

    flow = pd.DataFrame(flow_rows)
    cv_all = pd.concat(cv_frames, ignore_index=True)
    firth_all = pd.concat(firth_frames, ignore_index=True)
    ret = pd.DataFrame(ret_rows)
    zero = pd.DataFrame(zero_rows)
    twobytwo = pd.DataFrame(twobytwo_rows)

    side = cv_all.pivot_table(index=["checkpoint_minutes", "feature_set", "model"],
                            columns="threshold", values="roc_auc").reset_index()
    for k in THRESHOLDS:
        if k != PRIMARY and k in side.columns:
            side[f"delta_{k}_minus_{PRIMARY}"] = (side[k] - side[PRIMARY]).round(DEC)

    flow.to_csv(OUT_DIR / "vrsens_cohort_flow.csv", index=False)
    cv_all.to_csv(OUT_DIR / "vrsens_cv_results.csv", index=False)
    firth_all.to_csv(OUT_DIR / "vrsens_firth_lrt.csv", index=False)
    ret.to_csv(OUT_DIR / "vrsens_returning_positives.csv", index=False)
    zero.to_csv(OUT_DIR / "vrsens_zero_signal_counts.csv", index=False)
    twobytwo.to_csv(OUT_DIR / "vrsens_nulliparity_2x2.csv", index=False)
    side.to_csv(OUT_DIR / "vrsens_side_by_side.csv", index=False)

    print("\n=== COHORT FLOW ===");     print(flow.T.to_string())
    print("\n=== RETURNING POSITIVES (parity) ===")
    if len(ret):
        print(ret.groupby(["threshold", "direction", "parity_group"]).size().to_string())
    print("\n=== ZERO-SIGNAL COUNTS ==="); print(zero.to_string(index=False))
    print("\n=== NULLIPARITY 2x2 ==="); print(twobytwo.to_string(index=False))
    print("\n=== FIRTH LRT ===")
    print(firth_all[["threshold", "checkpoint_minutes", "maternal_raw_p", "maternal_holm_p",
                    "parity_raw_p", "parity_holm_p", "parity_or_per_additional_birth",
                    "all_converged", "ctg_cols_dropped"]].to_string(index=False))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
