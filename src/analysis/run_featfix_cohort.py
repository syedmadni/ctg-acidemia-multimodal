"""Step 0.5: rerun the classical pipeline with fixed CTG features and risk-set scoring.

Spec frozen in
outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value/FEATURE_FIX_SPEC.md

Writes a NEW cohort folder. Nothing existing is modified.

Risk set: at checkpoint t, only records with true recording duration >= t are scored.
Minimum duration in this cohort is 60.00 min, so the risk set is the full cohort at
10/20/30/45/60 and restricts only at 90 min (32 records, 4 positives). Any change at
10-60 is therefore attributable to the feature fix alone.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.analysis.significance import delong_test, paired_permutation_test
from src.data.fixed_ctg_features import ctg_summary_features_v2
from src.training.train_cv import (
    aggregate_repeated_oof_anytime,
    ctg_summary_features,
    crop_ctg_array,
    load_cohort_arrays,
    precompute_quality_splits,
    train_sklearn_quality_fold,
)

SUFFIX = "_705_anytime_vr060"
OLD_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060")
NEW_DIR = Path("outputs/quality_cohorts/primary_705_anytime_vr060_featfix")
CAP64_DIR = Path("outputs/quality_cohorts/anytime_cap64")
AUDIT = Path("outputs/audit/primary_705_valid_ratio_audit_records.csv")
CHECKPOINTS = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0]
MODELS = ["logistic_regression", "random_forest"]
FEATURE_SETS = ["ctg_only_features", "maternal_only_features", "both"]
SEED = 42
PERM_ITERS = 2000
MIN_POSITIVES = 10


def build_features(feature_set, X_ctg, X_mat, ck, legacy):
    if legacy:
        ctg = ctg_summary_features(crop_ctg_array(X_ctg, ck, 1))
    else:
        ctg = ctg_summary_features_v2(X_ctg, checkpoint_minutes=ck)
    if feature_set == "ctg_only_features":
        return ctg
    if feature_set == "maternal_only_features":
        return X_mat.astype(np.float32)
    return np.concatenate([ctg, X_mat], axis=1)


def durations(record_ids):
    aud = pd.read_csv(AUDIT)
    aud["record_id"] = aud["record_id"].astype(str)
    rid = record_ids.copy()
    rid["record_id"] = rid["record_id"].astype(str)
    merged = rid.merge(aud[["record_id", "full_record_duration_minutes"]], on="record_id", how="left")
    d = merged["full_record_duration_minutes"].to_numpy()
    if np.isnan(d).any():
        raise SystemExit("ABORT: duration missing for some cohort records; cannot form risk set.")
    return d


def score(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y_true, y_proba), average_precision_score(y_true, y_proba)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerun classical models with fixed CTG features.")
    ap.add_argument("--dry-run", action="store_true", help="single checkpoint (90 min), LR only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if NEW_DIR.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"ABORT: {NEW_DIR} exists. Pass --overwrite to replace.")

    X_ctg, X_mat, y, record_ids, _ = load_cohort_arrays(Path("data/processed"), SUFFIX)
    splits = precompute_quality_splits(y, 5, 3, SEED, 0.2)
    dur = durations(record_ids)
    rid_str = record_ids["record_id"].astype(str).to_numpy()
    dur_by_rid = dict(zip(rid_str, dur))

    cks = [90.0] if args.dry_run else CHECKPOINTS
    models = ["logistic_regression"] if args.dry_run else MODELS
    scratch = Path(tempfile.mkdtemp(prefix="featfix_"))

    print(f"cohort n={len(y)} positives={int(y.sum())} | duration min={dur.min():.2f} max={dur.max():.2f}")
    print(f"checkpoints={cks} models={models} dry_run={args.dry_run}\n")

    oof_rows = []
    try:
        for legacy in (False,) if args.dry_run else (False,):
            for ck in cks:
                for fs in FEATURE_SETS:
                    feats = build_features(fs, X_ctg, X_mat, ck, legacy=False)
                    for model in models:
                        for sp in splits:
                            _, _, tp, _ = train_sklearn_quality_fold(model, feats, y, sp, SEED, scratch)
                            for j, ix in enumerate(np.asarray(sp["outer_test_idx"], dtype=int)):
                                oof_rows.append({
                                    "cohort_name": fs, "checkpoint_minutes": ck, "model": model,
                                    "record_id": rid_str[ix], "y_true": int(y[ix]),
                                    "y_proba": float(tp[j]),
                                })
                        print(f"  done ck={ck:.0f} {fs} {model}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    avg = aggregate_repeated_oof_anytime(pd.DataFrame(oof_rows))
    avg["duration"] = avg["record_id"].map(dur_by_rid)

    # ---------------- results table ----------------
    rows = []
    for ck in cks:
        rs_mask_n = int((dur >= ck).sum())
        for fs in FEATURE_SETS:
            for model in models:
                g = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == fs) & (avg.model == model)]
                if g.empty:
                    continue
                full_auc, full_pr = score(g["y_true"].to_numpy(), g["y_proba"].to_numpy())
                rs = g[g["duration"] >= ck]
                rs_auc, rs_pr = score(rs["y_true"].to_numpy(), rs["y_proba"].to_numpy())
                npos = int(rs["y_true"].sum())
                rows.append({
                    "checkpoint_minutes": ck, "feature_set": fs, "model": model,
                    "n_full": len(g), "positives_full": int(g["y_true"].sum()),
                    "roc_auc_full": round(full_auc, 4), "pr_auc_full": round(full_pr, 4),
                    "n_risk_set": len(rs), "positives_risk_set": npos,
                    "roc_auc_risk_set": round(rs_auc, 4), "pr_auc_risk_set": round(rs_pr, 4),
                    "reliably_evaluable": bool(npos >= MIN_POSITIVES),
                })
    results = pd.DataFrame(rows)

    # ---------------- ablation significance, risk set ----------------
    sig_rows = []
    for ck in cks:
        for model in models:
            a = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == "both") & (avg.model == model)]
            b = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == "ctg_only_features") & (avg.model == model)]
            if a.empty or b.empty:
                continue
            a = a[a["duration"] >= ck].sort_values("record_id")
            b = b[b["duration"] >= ck].sort_values("record_id")
            if len(a) == 0 or a["y_true"].nunique() < 2:
                continue
            assert (a["record_id"].to_numpy() == b["record_id"].to_numpy()).all()
            yt = a["y_true"].to_numpy()
            d = delong_test(yt, a["y_proba"].to_numpy(), b["y_proba"].to_numpy())
            pm = paired_permutation_test(yt, a["y_proba"].to_numpy(), b["y_proba"].to_numpy(),
                                         n_permutations=PERM_ITERS, seed=SEED)
            sig_rows.append({
                "checkpoint_minutes": ck, "model": model, "n_risk_set": len(a),
                "positives_risk_set": int(yt.sum()),
                "auc_both": round(d["auc_a"], 4), "auc_ctg_only": round(d["auc_b"], 4),
                "gain": round(d["auc_a"] - d["auc_b"], 4),
                "delong_p": round(d["p_value"], 4), "permutation_p": round(pm["p_value"], 4),
                "reliably_evaluable": bool(int(yt.sum()) >= MIN_POSITIVES),
            })
    sig = pd.DataFrame(sig_rows)

    # ---------------- ensemble: fixed LR + existing CNN OOF ----------------
    ens_rows = []
    cap = CAP64_DIR / "oof_predictions.csv"
    if cap.exists():
        cnn = aggregate_repeated_oof_anytime(pd.read_csv(cap))
        cnn = cnn[cnn.model == "ctg_only"].copy()
        cnn["record_id"] = cnn["record_id"].astype(str)
        for ck in cks:
            lr = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == "both")
                     & (avg.model == "logistic_regression")][["record_id", "y_true", "y_proba", "duration"]]
            c = cnn[cnn.checkpoint_minutes == ck][["record_id", "y_true", "y_proba"]]
            m = lr.merge(c, on="record_id", suffixes=("_lr", "_cnn"))
            if m.empty:
                continue
            assert (m["y_true_lr"] == m["y_true_cnn"]).all()
            m["blend"] = 0.5 * m["y_proba_lr"] + 0.5 * m["y_proba_cnn"]
            f_auc, f_pr = score(m["y_true_lr"].to_numpy(), m["blend"].to_numpy())
            rs = m[m["duration"] >= ck]
            r_auc, r_pr = score(rs["y_true_lr"].to_numpy(), rs["blend"].to_numpy())
            ens_rows.append({
                "checkpoint_minutes": ck, "model": "ensemble_cnn_plus_fixedLR",
                "n_full": len(m), "roc_auc_full": round(f_auc, 4), "pr_auc_full": round(f_pr, 4),
                "n_risk_set": len(rs), "positives_risk_set": int(rs["y_true_lr"].sum()),
                "roc_auc_risk_set": round(r_auc, 4), "pr_auc_risk_set": round(r_pr, 4),
                "reliably_evaluable": bool(int(rs["y_true_lr"].sum()) >= MIN_POSITIVES),
            })
    ens = pd.DataFrame(ens_rows)

    if args.dry_run:
        print("\n--- DRY RUN: results ---")
        print(results.to_string(index=False))
        print("\n--- DRY RUN: ablation significance ---")
        print(sig.to_string(index=False) if len(sig) else "(none)")
        print("\n--- DRY RUN: ensemble ---")
        print(ens.to_string(index=False) if len(ens) else "(none)")
        print("\nDry run complete. Nothing written.")
        return

    NEW_DIR.mkdir(parents=True, exist_ok=True)
    avg.drop(columns=["duration"]).to_csv(NEW_DIR / "oof_predictions_averaged.csv", index=False)
    results.to_csv(NEW_DIR / "featfix_results.csv", index=False)
    sig.to_csv(NEW_DIR / "featfix_ablation_significance.csv", index=False)
    ens.to_csv(NEW_DIR / "featfix_ensemble.csv", index=False)

    # ---------------- old vs fixed ----------------
    old = pd.read_csv(OLD_DIR / "modality_ablation_results.csv")
    cmp_rows = []
    for _, r in results.iterrows():
        o = old[(old.checkpoint_minutes == r.checkpoint_minutes)
                & (old.feature_set == r.feature_set) & (old.model == r.model)]
        old_auc = float(o.iloc[0]["roc_auc"]) if len(o) else np.nan
        old_pr = float(o.iloc[0]["pr_auc"]) if len(o) else np.nan
        cmp_rows.append({
            "checkpoint_minutes": r.checkpoint_minutes, "feature_set": r.feature_set, "model": r.model,
            "old_n": 345, "old_positives": 23, "old_roc_auc": old_auc, "old_pr_auc": old_pr,
            "fixed_n_risk_set": r.n_risk_set, "fixed_positives_risk_set": r.positives_risk_set,
            "fixed_roc_auc_risk_set": r.roc_auc_risk_set, "fixed_pr_auc_risk_set": r.pr_auc_risk_set,
            "fixed_roc_auc_full_cohort": r.roc_auc_full,
            "delta_roc_auc_risk_set": round(r.roc_auc_risk_set - old_auc, 4) if old_auc == old_auc else np.nan,
            "delta_roc_auc_featurefix_only": round(r.roc_auc_full - old_auc, 4) if old_auc == old_auc else np.nan,
            "reliably_evaluable": r.reliably_evaluable,
        })
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(NEW_DIR / "old_vs_fixed_comparison.csv", index=False)

    (NEW_DIR / "cohort_note.json").write_text(json.dumps({
        "cohort_name": "primary_705_anytime_vr060_featfix",
        "supersedes": "primary_705_anytime_vr060",
        "suffix": SUFFIX, "n": int(len(y)), "positives": int(y.sum()),
        "change": "fixed CTG summary features (no padding, zeros treated as missing) + risk-set evaluation",
        "spec": str(OLD_DIR / "maternal_value" / "FEATURE_FIX_SPEC.md"),
        "deep_models": "NOT retrained; existing OOF re-evaluated on risk set only",
    }, indent=2), encoding="utf-8")

    print("\n=== RESULTS (fixed features) ===")
    print(results.to_string(index=False))
    print("\n=== ABLATION, RISK SET ===")
    print(sig.to_string(index=False))
    print("\n=== ENSEMBLE ===")
    print(ens.to_string(index=False))
    print(f"\nWrote {NEW_DIR}")


if __name__ == "__main__":
    main()
