"""Maternal-value investigation: ROBUSTNESS CHECKS ADDED AFTER SEEING STEP 1-3 RESULTS.

Every output of this script carries that label. These checks were requested by the user
on 2026-09-20 after reading the Step 1-3 tables, so they are NOT pre-specified and must
never be reported as though they were. They exist to probe four things the Step 1-3
results raised:

  1. Firth-vs-MLE decomposition. Step 1's Firth test (df 8) and standard-MLE sensitivity
     (df 7, pre-eclampsia removed) disagreed. Two things differ at once there: the
     estimator (penalized vs not) AND the variable set. This check runs Firth with
     pre-eclampsia removed (df 7), isolating the estimator.
  2. Permutation LRT for the maternal block, as a distribution-free complement to the
     chi-square reference used in Step 1.
  3. Parity adjusted for age, since parity and age correlate at r=0.34 (Step 2d).
  4. Parity fragility, since the 2x2 OR rests on a single parous positive case.

Conventions match Steps 1-3 exactly: fixed CTG features, whole-cohort median imputation
for these single-fit analyses, checkpoints 10-60 (90 excluded per Amendment 1), Holm
across the 5 checkpoints, logistftest restricted-model convention for every Firth LRT.

Outputs: outputs/quality_cohorts/primary_705_anytime_vr060/maternal_value/robust_*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.firth import firth_lrt  # noqa: E402
from src.analysis.run_maternal_value_steps123 import (  # noqa: E402
    CHECKPOINTS,
    MATERNAL_NAMES,
    OUT_DIR,
    REPORT_DECIMALS_FILE,
    SEED,
    build_ctg_block,
    holm,
    impute_maternal,
    load_data,
    odds_ratio_2x2,
    precompute_quality_splits,
)

LABEL = "ROBUSTNESS CHECK ADDED AFTER SEEING STEP 1-3 RESULTS (not pre-specified)"
N_PERMUTATIONS = 2000


# ---------------------------------------------------------------------------
# 1. Firth LRT with pre-eclampsia removed (df 7) -- isolates estimator from variable set
# ---------------------------------------------------------------------------

def check1_firth_no_preeclampsia(X_ctg, X_maternal, y) -> pd.DataFrame:
    Xmat_imp = impute_maternal(X_maternal)
    pe_idx = MATERNAL_NAMES.index("preeclampsia")
    keep_mat = [i for i in range(8) if i != pe_idx]
    rows = []
    for ck in CHECKPOINTS:
        ctg, ctg_names, _ = build_ctg_block(X_ctg, ck)
        X = np.column_stack([np.ones(len(y)), ctg, Xmat_imp[:, keep_mat]])
        maternal_idx = list(range(1 + len(ctg_names), 1 + len(ctg_names) + 7))
        lrt = firth_lrt(X, y, test_idx=maternal_idx)
        rows.append({
            "label": LABEL,
            "checkpoint_minutes": ck,
            "test": "Firth LRT, M0(CTG) vs M1minusPE(CTG + 7 maternal), pre-eclampsia removed",
            "lr_stat": round(lrt.lr_stat, REPORT_DECIMALS_FILE),
            "df": lrt.df,
            "raw_p": lrt.p_value,
            "converged_full": lrt.full_fit.converged,
            "converged_restricted": lrt.restricted_fit.converged,
        })
    df = pd.DataFrame(rows)
    df["holm_p"] = holm(df["raw_p"].tolist())
    return df


# ---------------------------------------------------------------------------
# 2. Permutation LRT for the maternal block
# ---------------------------------------------------------------------------

def check2_permutation_lrt(X_ctg, X_maternal, y) -> pd.DataFrame:
    Xmat_imp = impute_maternal(X_maternal)
    n = len(y)
    rows = []
    for ck in CHECKPOINTS:
        ctg, ctg_names, _ = build_ctg_block(X_ctg, ck)
        ones = np.ones(n)
        maternal_idx = list(range(1 + len(ctg_names), 1 + len(ctg_names) + 8))

        X_obs = np.column_stack([ones, ctg, Xmat_imp])
        observed = firth_lrt(X_obs, y, test_idx=maternal_idx).lr_stat

        rng = np.random.default_rng(SEED)
        count_ge = 0
        null_stats = []
        for _ in range(N_PERMUTATIONS):
            perm = rng.permutation(n)
            X_perm = np.column_stack([ones, ctg, Xmat_imp[perm]])
            stat = firth_lrt(X_perm, y, test_idx=maternal_idx).lr_stat
            null_stats.append(stat)
            if stat >= observed:
                count_ge += 1
        null_stats = np.asarray(null_stats)
        p_perm = (count_ge + 1) / (N_PERMUTATIONS + 1)

        rows.append({
            "label": LABEL,
            "checkpoint_minutes": ck,
            "test": "Permutation LRT, maternal block (8 vars) permuted JOINTLY as whole rows",
            "observed_lr_stat": round(observed, REPORT_DECIMALS_FILE),
            "null_mean": round(float(null_stats.mean()), REPORT_DECIMALS_FILE),
            "null_median": round(float(np.median(null_stats)), REPORT_DECIMALS_FILE),
            "null_p95": round(float(np.percentile(null_stats, 95)), REPORT_DECIMALS_FILE),
            "n_null_ge_observed": count_ge,
            "n_permutations": N_PERMUTATIONS,
            "permutation_raw_p": p_perm,
            "chisq_reference_p_for_comparison": float(stats.chi2.sf(observed, 8)),
            "note": "Rows of the maternal block are permuted together, so the correlation "
                    "AMONG the 8 maternal variables is preserved. This also breaks any "
                    "maternal-CTG correlation, not only the maternal-outcome association, "
                    "so the null is 'maternal block unrelated to both outcome and CTG', "
                    "which is broader than the LRT's nominal null.",
        })
    df = pd.DataFrame(rows)
    df["holm_p"] = holm(df["permutation_raw_p"].tolist())
    return df


# ---------------------------------------------------------------------------
# 3. Parity adjusted for age
# ---------------------------------------------------------------------------

def check3_parity_adjusted_for_age(X_ctg, X_maternal, y) -> pd.DataFrame:
    Xmat_imp = impute_maternal(X_maternal)
    parity = Xmat_imp[:, MATERNAL_NAMES.index("parity")]
    age = Xmat_imp[:, MATERNAL_NAMES.index("age")]
    rows = []
    for ck in CHECKPOINTS:
        ctg, ctg_names, _ = build_ctg_block(X_ctg, ck)
        X = np.column_stack([np.ones(len(y)), ctg, age, parity])
        parity_col = X.shape[1] - 1
        lrt = firth_lrt(X, y, test_idx=[parity_col])
        beta = float(lrt.full_fit.beta[parity_col])
        se = float(lrt.full_fit.se[parity_col])
        rows.append({
            "label": LABEL + " -- parity remains POST HOC",
            "checkpoint_minutes": ck,
            "test": "Firth LRT, M0(CTG + age) vs M0 + parity",
            "lr_stat": round(lrt.lr_stat, REPORT_DECIMALS_FILE),
            "df": lrt.df,
            "raw_p": lrt.p_value,
            "parity_coef": round(beta, REPORT_DECIMALS_FILE),
            "parity_se": round(se, REPORT_DECIMALS_FILE),
            "parity_odds_ratio": round(float(np.exp(beta)), REPORT_DECIMALS_FILE),
            "parity_or_95ci_low": round(float(np.exp(beta - 1.96 * se)), REPORT_DECIMALS_FILE),
            "parity_or_95ci_high": round(float(np.exp(beta + 1.96 * se)), REPORT_DECIMALS_FILE),
        })
    df = pd.DataFrame(rows)
    df["holm_p"] = holm(df["raw_p"].tolist())
    return df


# ---------------------------------------------------------------------------
# 4. Parity fragility
# ---------------------------------------------------------------------------

def check4_parity_fragility(X_maternal, y) -> pd.DataFrame:
    Xmat_imp = impute_maternal(X_maternal)
    nulli = (Xmat_imp[:, MATERNAL_NAMES.index("parity")] == 0).astype(int)
    a = int(((nulli == 1) & (y == 1)).sum())
    b = int(((nulli == 1) & (y == 0)).sum())
    c = int(((nulli == 0) & (y == 1)).sum())
    d = int(((nulli == 0) & (y == 0)).sum())

    scenarios = [
        ("observed (as analysed in Step 2a)", a, b, c, d),
        ("one parous NEGATIVE recoded POSITIVE", a, b, c + 1, d - 1),
        ("the single parous POSITIVE recoded NEGATIVE", a, b, c - 1, d + 1),
    ]
    rows = []
    for name, aa, bb, cc, dd in scenarios:
        fisher_or, fisher_p = stats.fisher_exact(np.array([[aa, bb], [cc, dd]]))
        or_, log_se, used_ha = odds_ratio_2x2(aa, bb, cc, dd)
        rows.append({
            "label": LABEL + " -- descriptive only, no inference",
            "scenario": name,
            "n_nulliparous_positive": aa, "n_nulliparous_negative": bb,
            "n_parous_positive": cc, "n_parous_negative": dd,
            "fisher_exact_p": float(fisher_p),
            "odds_ratio": round(float(or_), REPORT_DECIMALS_FILE),
            "odds_ratio_95ci_low": round(float(np.exp(np.log(or_) - 1.96 * log_se)), REPORT_DECIMALS_FILE),
            "odds_ratio_95ci_high": round(float(np.exp(np.log(or_) + 1.96 * log_se)), REPORT_DECIMALS_FILE),
            "haldane_anscombe_correction_used": used_ha,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. EPV provenance
# ---------------------------------------------------------------------------

def check5_epv_provenance(y, splits) -> pd.DataFrame:
    rows = []
    for sp in splits:
        outer_train = np.asarray(sp["outer_train_idx"], dtype=int)
        inner_train = np.asarray(sp["inner_train_idx"], dtype=int)
        val = np.asarray(sp["val_idx"], dtype=int)
        outer_test = np.asarray(sp["outer_test_idx"], dtype=int)
        rows.append({
            "fold_name": sp["fold_name"],
            "n_outer_train": len(outer_train), "pos_outer_train": int(y[outer_train].sum()),
            "n_inner_train": len(inner_train), "pos_inner_train": int(y[inner_train].sum()),
            "n_val": len(val), "pos_val": int(y[val].sum()),
            "n_outer_test": len(outer_test), "pos_outer_test": int(y[outer_test].sum()),
        })
    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "label": LABEL,
        "question": "Why 14.4 train-fold positives and not ~18.4?",
        "answer": "An inner validation split IS carved out of every outer training fold, "
                  "and the classical models are fit on the inner-train part ONLY.",
        "code_precompute_split": "src/training/train_cv.py:645-650 -- "
                                 "inner_train_idx, val_idx = train_test_split(outer_train_idx, "
                                 "test_size=val_size, random_state=seed+split_number, "
                                 "stratify=y[outer_train_idx])",
        "code_lr_fit": "src/training/train_cv.py:826 -- "
                       "model.fit(X_tabular[inner_train_idx], y[inner_train_idx])",
        "val_size": 0.2,
        "mean_pos_outer_train": round(float(df["pos_outer_train"].mean()), REPORT_DECIMALS_FILE),
        "mean_pos_inner_train": round(float(df["pos_inner_train"].mean()), REPORT_DECIMALS_FILE),
        "mean_pos_val": round(float(df["pos_val"].mean()), REPORT_DECIMALS_FILE),
        "arithmetic": "23 positives x 4/5 (outer CV) = 18.4 in outer-train; "
                      "18.4 x 0.8 (inner split) = 14.72 expected in inner-train; "
                      "observed mean 14.4 (stratified rounding per fold).",
        "consequence": "EPV per fold should be computed on inner-train positives (14.4), "
                       "not outer-train (18.4), because inner-train is what the model sees. "
                       "The val split is scored but never trained on, and for LR/RF nothing "
                       "is selected from it (fixed hyperparameters), so those 4.0 positives "
                       "per fold are spent for fold-comparability with the deep models.",
    }])
    return df, summary


# ---------------------------------------------------------------------------
# 6. Binary nulliparity coding (the coding the HANDOFF work order specified for Step 2)
# ---------------------------------------------------------------------------

NULLI_LABEL = (
    "Binary nulliparous (0/1) coding -- this CODING is the one the HANDOFF Step 2 work "
    "order specified (the 2x2). The decision to study PARITY AT ALL remains POST HOC, "
    "selected after the 2026-08-12 diagnostic showed 22/23 positives were nulliparous. "
    "These model fits were run after seeing the Step 1-3 results."
)


def check6_binary_nulliparity(X_ctg, X_maternal, y, record_ids, splits) -> tuple:
    Xmat_imp = impute_maternal(X_maternal)
    parity = Xmat_imp[:, MATERNAL_NAMES.index("parity")]
    age = Xmat_imp[:, MATERNAL_NAMES.index("age")]
    nulli = (parity == 0).astype(float)  # 1 = nulliparous, 0 = parous

    rows = []
    for adjust_age in (False, True):
        for ck in CHECKPOINTS:
            ctg, ctg_names, _ = build_ctg_block(X_ctg, ck)
            blocks = [np.ones(len(y)), ctg] + ([age] if adjust_age else []) + [nulli]
            X = np.column_stack(blocks)
            nulli_col = X.shape[1] - 1
            lrt = firth_lrt(X, y, test_idx=[nulli_col])
            beta = float(lrt.full_fit.beta[nulli_col])
            se = float(lrt.full_fit.se[nulli_col])
            rows.append({
                "label": NULLI_LABEL,
                "model": "CTG + age + nulliparous" if adjust_age else "CTG + nulliparous",
                "age_adjusted": adjust_age,
                "checkpoint_minutes": ck,
                "test": ("Firth LRT, M0(CTG + age) vs M0 + nulliparous" if adjust_age
                        else "Firth LRT, M0(CTG) vs M0 + nulliparous"),
                "lr_stat": round(lrt.lr_stat, REPORT_DECIMALS_FILE),
                "df": lrt.df,
                "raw_p": lrt.p_value,
                "or_direction": "nulliparous vs parous (>1 = nulliparity raises odds of acidemia)",
                "odds_ratio_nulliparous_vs_parous": round(float(np.exp(beta)), REPORT_DECIMALS_FILE),
                "or_95ci_low": round(float(np.exp(beta - 1.96 * se)), REPORT_DECIMALS_FILE),
                "or_95ci_high": round(float(np.exp(beta + 1.96 * se)), REPORT_DECIMALS_FILE),
                "coef": round(beta, REPORT_DECIMALS_FILE),
                "se": round(se, REPORT_DECIMALS_FILE),
            })
    lrt_df = pd.DataFrame(rows)
    # Holm within each family (unadjusted / age-adjusted) separately, 5 checkpoints each
    lrt_df["holm_p"] = np.nan
    for adj in (False, True):
        m = lrt_df["age_adjusted"] == adj
        lrt_df.loc[m, "holm_p"] = holm(lrt_df.loc[m, "raw_p"].tolist())

    # ---- CV discrimination: CTG + nulliparous vs CTG-only, same folds ----
    import shutil
    import tempfile

    from sklearn.metrics import roc_auc_score

    from src.analysis.significance import delong_test, paired_permutation_test
    from src.analysis.run_maternal_value_steps123 import (
        PERM_ITERS, aggregate_repeated_oof_anytime, build_ctg_block_raw, train_sklearn_quality_fold,
    )

    nulli_raw = (X_maternal[:, MATERNAL_NAMES.index("parity")] == 0).astype(float)
    scratch = Path(tempfile.mkdtemp(prefix="nulliparity_"))
    oof_rows = []
    try:
        for ck in CHECKPOINTS:
            ctg_raw = build_ctg_block_raw(X_ctg, ck)
            feats = {
                "ctg_only_features": ctg_raw,
                "ctg_plus_nulliparous": np.column_stack([ctg_raw, nulli_raw]),
            }
            for fs_name, X_fs in feats.items():
                for sp in splits:
                    _, _, tp, _ = train_sklearn_quality_fold("logistic_regression", X_fs, y, sp, SEED, scratch)
                    for j, ix in enumerate(np.asarray(sp["outer_test_idx"], dtype=int)):
                        oof_rows.append({
                            "checkpoint_minutes": ck, "cohort_name": fs_name, "model": "logistic_regression",
                            "record_id": record_ids.iloc[ix]["record_id"], "y_true": int(y[ix]),
                            "y_proba": float(tp[j]),
                        })
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    avg = aggregate_repeated_oof_anytime(pd.DataFrame(oof_rows))
    cv_rows = []
    for ck in CHECKPOINTS:
        a = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == "ctg_plus_nulliparous")].sort_values("record_id")
        b = avg[(avg.checkpoint_minutes == ck) & (avg.cohort_name == "ctg_only_features")].sort_values("record_id")
        assert (a["record_id"].to_numpy() == b["record_id"].to_numpy()).all()
        yt = a["y_true"].to_numpy()
        d = delong_test(yt, a["y_proba"].to_numpy(), b["y_proba"].to_numpy())
        pm = paired_permutation_test(yt, a["y_proba"].to_numpy(), b["y_proba"].to_numpy(),
                                     n_permutations=PERM_ITERS, seed=SEED)
        cv_rows.append({
            "label": NULLI_LABEL,
            "checkpoint_minutes": ck,
            "auc_ctg_plus_nulliparous": round(float(roc_auc_score(yt, a["y_proba"])), REPORT_DECIMALS_FILE),
            "auc_ctg_only": round(float(roc_auc_score(yt, b["y_proba"])), REPORT_DECIMALS_FILE),
            "gain": round(float(roc_auc_score(yt, a["y_proba"]) - roc_auc_score(yt, b["y_proba"])), REPORT_DECIMALS_FILE),
            "delong_p": float(d["p_value"]), "permutation_p": float(pm["p_value"]),
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df["delong_holm_p"] = holm(cv_df["delong_p"].tolist())
    cv_df["permutation_holm_p"] = holm(cv_df["permutation_p"].tolist())
    return lrt_df, cv_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_ctg, X_maternal, y, record_ids = load_data()
    splits = precompute_quality_splits(y, 5, 3, SEED, 0.2)

    print("=== 1. Firth LRT, pre-eclampsia removed (df 7) ===")
    c1 = check1_firth_no_preeclampsia(X_ctg, X_maternal, y)
    c1.to_csv(OUT_DIR / "robust1_firth_no_preeclampsia.csv", index=False)
    print(c1.to_string(index=False))

    print(f"\n=== 2. Permutation LRT, maternal block ({N_PERMUTATIONS} permutations) ===")
    c2 = check2_permutation_lrt(X_ctg, X_maternal, y)
    c2.drop(columns=["note"]).to_csv(OUT_DIR / "robust2_permutation_lrt.csv", index=False)
    pd.DataFrame([{"label": LABEL, "note": c2.iloc[0]["note"]}]).to_csv(
        OUT_DIR / "robust2_permutation_lrt_note.csv", index=False)
    print(c2.drop(columns=["note"]).to_string(index=False))

    print("\n=== 3. Parity adjusted for age ===")
    c3 = check3_parity_adjusted_for_age(X_ctg, X_maternal, y)
    c3.to_csv(OUT_DIR / "robust3_parity_adjusted_for_age.csv", index=False)
    print(c3.to_string(index=False))

    print("\n=== 4. Parity fragility ===")
    c4 = check4_parity_fragility(X_maternal, y)
    c4.to_csv(OUT_DIR / "robust4_parity_fragility.csv", index=False)
    print(c4.to_string(index=False))

    print("\n=== 5. EPV provenance ===")
    per_fold, summary = check5_epv_provenance(y, splits)
    per_fold.to_csv(OUT_DIR / "robust5_epv_per_fold.csv", index=False)
    summary.to_csv(OUT_DIR / "robust5_epv_explanation.csv", index=False)
    print(per_fold.head().to_string(index=False))
    print("...")
    print(summary.T.to_string())

    print("\n=== 6. Binary nulliparity coding ===")
    c6_lrt, c6_cv = check6_binary_nulliparity(X_ctg, X_maternal, y, record_ids, splits)
    c6_lrt.to_csv(OUT_DIR / "robust6_nulliparity_lrt.csv", index=False)
    c6_cv.to_csv(OUT_DIR / "robust6_nulliparity_cv.csv", index=False)
    print(c6_lrt.drop(columns=["label", "test", "or_direction"]).to_string(index=False))
    print(c6_cv.drop(columns=["label"]).to_string(index=False))

    print(f"\nWrote robustness outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
